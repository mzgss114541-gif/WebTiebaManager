# -*- coding: utf-8 -*-
"""
OCR Review Plugin for WebTiebaManager
======================================
- Registers "ImageOCRHit" rule condition via Conditions.register()
- Automatically runs OCR when WTM processes posts with images
- Results cached per-pid to avoid re-OCR
- Keywords configured in rule condition's text field (user types in rule editor)
- Dry-run by default: returns true/false, deletion handled by WTM's normal
  rule operations (user configures Delete/Confirm/Ignore as usual)

API:
  GET  /ocr-review                  — management page (view results, manual scan)
  GET  /api/plugin/ocr-review/list  — list scan results
  POST /api/plugin/ocr-review/scan/{pid} — manual scan (debug)
  GET  /api/plugin/ocr-review/config — get OCR settings
  PUT  /api/plugin/ocr-review/config — update OCR settings
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import re
import time
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Literal

import aiohttp
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.api.server import BaseResponse, app
from src.core.constants import BASE_DIR
from src.db import Database
from src.rule.condition import Conditions
from src.rule.template import TextCondition, TextOptions
from src.schemas.process import ProcessObject
from src.schemas.tieba import Content
from src.utils.logging import system_logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR = BASE_DIR / "plugin_data" / "ocr_review"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = CACHE_DIR / "config.json"
RESULTS_DIR = CACHE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MAX_IMAGE_SIZE = 10 * 1024 * 1024
DOWNLOAD_TIMEOUT = 15
OCR_TIMEOUT = 30
MAX_IMAGES = 10
MAX_CACHE_ENTRIES = 500           # auto-prune threshold

ALLOWED_IMAGE_HOSTS = {
    "tiebapic.baidu.com",
    "imgsrc.baidu.com",
    "hiphotos.baidu.com",
    "gss0.baidu.com",
    "gss0.bdstatic.com",
    "gss1.bdstatic.com",
    "gss2.bdstatic.com",
    "gss3.bdstatic.com",
}

BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("ff00::/8"),
]


def _is_internal_host(hostname: str) -> bool:
    if hostname in ("localhost", "0.0.0.0", "::1"):
        return True
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return any(addr in net for net in BLOCKED_NETWORKS)


def _validate_image_url(url: str) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if _is_internal_host(host):
        return False
    if host not in ALLOWED_IMAGE_HOSTS:
        if not any(host == h or host.endswith("." + h) for h in ALLOWED_IMAGE_HOSTS):
            return False
    return True

# ---------------------------------------------------------------------------
# Lazy OCR engine
# ---------------------------------------------------------------------------

_ocr_engine: Any = None
_ocr_lock = asyncio.Lock()


async def _get_engine():
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    async with _ocr_lock:
        if _ocr_engine is not None:
            return _ocr_engine
        from rapidocr_onnxruntime import RapidOCR
        loop = asyncio.get_running_loop()
        _ocr_engine = await loop.run_in_executor(None, RapidOCR)
        system_logger.info("[ocr_review] OCR engine ready")
        return _ocr_engine


# ---------------------------------------------------------------------------
# Config (global OCR settings, NOT per-rule keywords)
# ---------------------------------------------------------------------------

class OCRConfig(BaseModel):
    max_images: int = MAX_IMAGES
    enabled: bool = True


def _load_config() -> OCRConfig:
    if CONFIG_FILE.exists():
        try:
            return OCRConfig(**json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return OCRConfig()


def _save_config(cfg: OCRConfig) -> None:
    CONFIG_FILE.write_text(cfg.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Log sanitizer
# ---------------------------------------------------------------------------

_SENSITIVE = [
    (r"BDUSS=[A-Za-z0-9+/=]+", "BDUSS=***"),
    (r"bduss=[A-Za-z0-9+/=]+", "bduss=***"),
    (r"Cookie:\s*[^\n]+", "Cookie: ***"),
    (r"stoken=[A-Za-z0-9+/=]+", "stoken=***"),
    (r"\b[A-Za-z0-9+/=]{150,}\b", "[TOKEN]"),
]


def _sanitize(msg: str) -> str:
    for pat, rep in _SENSITIVE:
        msg = re.sub(pat, rep, msg)
    return msg


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

async def _download(url: str) -> bytes | None:
    if not _validate_image_url(url):
        system_logger.debug(f"[ocr_review] blocked url: {_sanitize(url)}")
        return None
    try:
        t = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
        async with aiohttp.ClientSession(timeout=t) as s:
            async with s.get(url) as resp:
                if resp.status != 200:
                    return None
                ct = resp.headers.get("Content-Type", "")
                if not ct.startswith("image/"):
                    return None
                cl = resp.headers.get("Content-Length")
                if cl and int(cl) > MAX_IMAGE_SIZE:
                    return None
                data = await resp.read()
                if len(data) > MAX_IMAGE_SIZE:
                    return None
                return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _img_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _ocr_sync(engine: Any, img: bytes):
    result, _ = engine(img)
    if result is None:
        return []
    return [(item[1], item[2]) for item in result if item[1]]


async def _ocr(engine: Any, img: bytes) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        pairs = await asyncio.wait_for(
            loop.run_in_executor(None, _ocr_sync, engine, img),
            timeout=OCR_TIMEOUT,
        )
        return [t for t, _ in pairs]
    except asyncio.TimeoutError:
        return []


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------

class HitItem(BaseModel):
    image_url: str
    image_hash: str
    text_lines: list[str]


class ImgDetail(BaseModel):
    url: str
    hash: str
    width: int
    height: int
    text: str           # OCR text from this image
    text_length: int


class ScanResult(BaseModel):
    pid: int
    tid: int
    forum_name: str
    scan_time: float
    total_images: int
    ocr_images: int
    ocr_text: str       # all OCR text concatenated
    images: list[ImgDetail]  # per-image detail
    trigger: str = ""   # "manual" or "auto"


def _result_path(pid: int) -> Path:
    return RESULTS_DIR / f"{pid}.json"


def _load_cache(pid: int) -> ScanResult | None:
    p = _result_path(pid)
    if not p.exists():
        return None
    try:
        return ScanResult(**json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return None


def _prune_cache(keep: int = MAX_CACHE_ENTRIES) -> int:
    files = sorted(RESULTS_DIR.glob("*.json"), key=os.path.getmtime)
    if len(files) <= keep:
        return 0
    removed = 0
    for f in files[:-keep]:
        f.unlink()
        removed += 1
    return removed


def _save_cache(r: ScanResult) -> None:
    _result_path(r.pid).write_text(
        r.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _prune_cache()


# ---------------------------------------------------------------------------
# OCR a post's images, return all text
# ---------------------------------------------------------------------------

async def _ocr_post(pid: int, content: Content | None = None) -> str:
    """Run OCR on all images of a post. Returns concatenated OCR text. Cached."""
    cached = _load_cache(pid)
    if cached is not None:
        return cached.ocr_text

    if content is None:
        content = await Database.get_full_content_by_pid(pid)
    if content is None or content.pid == 0:
        return ""

    cfg = _load_config()
    if not cfg.enabled:
        return ""
    images = content.images[: cfg.max_images]

    engine = await _get_engine()
    all_texts: list[str] = []
    details: list[ImgDetail] = []
    total = len(images)
    n_ocr = 0

    for img in images:
        url = img.src
        if not url:
            continue
        data = await _download(url)
        if data is None:
            continue
        h = _img_hash(data)
        lines = await _ocr(engine, data)
        if not lines:
            continue
        n_ocr += 1
        txt = "\n".join(lines)
        all_texts.append(txt)
        details.append(ImgDetail(
            url=_sanitize(url), hash=h,
            width=img.width, height=img.height,
            text=txt, text_length=len(txt),
        ))

    full_text = "\n".join(all_texts)

    result = ScanResult(
        pid=content.pid, tid=content.tid, forum_name=content.fname,
        scan_time=time.time(), total_images=total, ocr_images=n_ocr,
        ocr_text=full_text, images=details, trigger="",
    )
    if total:
        _save_cache(result)

    system_logger.info(
        f"[ocr_review] auto-ocr pid={pid} tid={content.tid} "
        f"imgs={n_ocr}/{total} text_len={len(full_text)}"
    )
    return full_text


# ---------------------------------------------------------------------------
# Rule Condition: ImageOCRHitCondition
#
# Inherits from TextCondition so the WTM rule editor shows a text input.
# User types a keyword (e.g. "微信") in the rule editor.
# When WTM processes a post, get_value() runs OCR on its images and
# returns the OCR text. The parent TextCondition.check() then matches
# the user's keyword against the OCR text.
# ---------------------------------------------------------------------------

@Conditions.register(
    name="图片OCR命中关键词",
    category="帖子内容",
    description="自动OCR帖子图片，匹配用户输入的关键词。首次OCR会耗时，结果缓存后秒级判断",
)
class ImageOCRHitCondition(TextCondition):
    type: Literal["ImageOCRHitCondition"] = "ImageOCRHitCondition"  # type: ignore[assignment]
    options: TextOptions
    priority: int = 10  # low priority — OCR is expensive, run last

    async def get_value(self, obj: ProcessObject) -> str:
        pid = getattr(obj.content, "pid", 0)
        if not pid:
            return ""
        return await _ocr_post(pid, obj.content)


# ---------------------------------------------------------------------------
# Management page (vanilla HTML/JS, no CDN)
# ---------------------------------------------------------------------------

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>OCR Review</title>
<style>
:root{--bg:#f5f7fa;--card:#fff;--t1:#303133;--t2:#606266;--t3:#909399;--bd:#dcdfe6;--pri:#409eff;--dng:#f56c6c;--suc:#67c23a;--war:#e6a23c;--rad:8px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Helvetica Neue",Helvetica,"PingFang SC","Microsoft YaHei",Arial,sans-serif;background:var(--bg);color:var(--t1);line-height:1.6}
.app{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;font-weight:600;margin-bottom:4px}
h2{font-size:16px;font-weight:600;margin-bottom:16px}
.card{background:var(--card);border-radius:var(--rad);padding:20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.06);border:1px solid var(--bd)}
.row{display:flex;gap:12px;align-items:flex-end;flex-wrap:wrap;margin-bottom:12px}
.col{flex:1;min-width:200px}
label{display:block;font-size:13px;color:var(--t2);margin-bottom:4px;font-weight:500}
input,textarea,select{width:100%;padding:8px 12px;border:1px solid var(--bd);border-radius:6px;font-size:14px;font-family:inherit;color:var(--t1);background:#fff;outline:none}
input:focus,textarea:focus,select:focus{border-color:var(--pri)}
textarea{resize:vertical;min-height:80px}
button{padding:8px 20px;border:none;border-radius:6px;font-size:14px;font-weight:500;cursor:pointer;transition:opacity .2s}
.btn-pri{background:var(--pri);color:#fff}.btn-pri:hover{opacity:.85}
.btn-suc{background:var(--suc);color:#fff}.btn-suc:hover{opacity:.85}
.btn-out{background:#fff;color:var(--t1);border:1px solid var(--bd)}.btn-out:hover{border-color:var(--pri);color:var(--pri)}
.btn-sm{padding:4px 12px;font-size:12px}
.btn-sm{padding:4px 12px;font-size:12px}
.tag-inline{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;margin:2px}
.tag-info{background:#ecf5ff;color:var(--pri)}.tag-dng{background:#fef0f0;color:var(--dng)}.tag-suc{background:#f0f9eb;color:var(--suc)}.tag-war{background:#fdf6ec;color:var(--war)}
.flex{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.mt8{margin-top:8px}.mt12{margin-top:12px}.mt16{margin-top:16px}
.result{border-bottom:1px solid var(--bd);padding:16px 0}.result:last-child{border-bottom:none}
.empty{text-align:center;color:var(--t3);padding:40px 0}
.spin{width:16px;height:16px;border:2px solid var(--bd);border-top-color:var(--pri);border-radius:50%;animation:sp .6s linear infinite;display:inline-block;vertical-align:middle;margin-right:6px}
@keyframes sp{to{transform:rotate(360deg)}}
.toast{position:fixed;top:16px;right:16px;padding:10px 20px;border-radius:6px;font-size:14px;z-index:999;display:none;box-shadow:0 4px 12px rgba(0,0,0,.15)}
.toast.ok{background:#f0f9eb;color:var(--suc);border:1px solid #c2e7b0;display:block}
.toast.err{background:#fef0f0;color:var(--dng);border:1px solid #fbc4c4;display:block}
pre.ocr-text{background:#f8f9fa;padding:12px;border-radius:6px;font-size:13px;white-space:pre-wrap;max-height:200px;overflow-y:auto;margin-top:8px;border:1px solid var(--bd)}
</style>
</head>
<body>
<div class="app">
<h1>OCR Review</h1>
<div style="color:var(--t3);font-size:13px;margin-bottom:20px">
  规则条件: <b>图片OCR命中关键词</b> — 在 WTM 规则编辑器中添加此条件，输入关键词即可自动 OCR 匹配
</div>

<div class="card">
  <h2>调试: 手动扫描 PID</h2>
  <div class="row">
    <div class="col"><label>帖子 PID</label><input id="pid" type="number" placeholder="手动扫描仅用于调试"></div>
    <div><label>&nbsp;</label><button class="btn-pri" id="btn-scan" onclick="doScan()">扫描</button></div>
  </div>
  <div id="status" class="mt8"></div>
</div>

<div class="card">
  <h2>设置</h2>
  <div class="row">
    <div><label>每帖最多 OCR 图片数</label><input id="maximg" type="number" min="1" max="50" value="10" style="width:100px"></div>
    <div style="display:flex;align-items:flex-end;padding-bottom:1px"><label><input type="checkbox" id="enabled" checked> 启用 OCR</label></div>
    <div><label>&nbsp;</label><button class="btn-suc" id="btn-save" onclick="saveConfig()">保存</button></div>
  </div>
  <div class="mt8">
    <button class="btn-dng btn-sm" id="btn-clear" onclick="clearCache()">清除所有缓存</button>
  </div>
</div>

<div class="card">
  <div class="flex"><h2 style="margin-bottom:0">OCR 缓存记录</h2><button class="btn-out btn-sm" onclick="loadResults()">刷新</button></div>
  <div id="results" class="mt8"><div class="empty">加载中...</div></div>
</div>
</div>
<div id="toast" class="toast"></div>
<script>
var B="/api/plugin/ocr-review";
function toast(m,c){var e=document.getElementById("toast");e.textContent=m;e.className="toast "+c;setTimeout(function(){e.className="toast"},3000)}
function api(m,p,b){var o={method:m,headers:{}};if(b){o.headers["Content-Type"]="application/json";o.body=JSON.stringify(b)}return fetch(B+p,o).then(function(r){return r.json().then(function(d){if(!r.ok)throw new Error(d.detail||r.statusText);return d})})}
function loadConfig(){api("GET","/config").then(function(d){document.getElementById("maximg").value=d.data.max_images;document.getElementById("enabled").checked=d.data.enabled}).catch(function(e){toast(e.message,"err")})}
function saveConfig(){var b=document.getElementById("btn-save");b.disabled=true;b.textContent="保存中...";var mx=parseInt(document.getElementById("maximg").value)||10;var en=document.getElementById("enabled").checked;api("PUT","/config",{max_images:mx,enabled:en}).then(function(){toast("已保存","ok")}).catch(function(e){toast(e.message,"err")}).finally(function(){b.disabled=false;b.textContent="保存"})}
function doScan(){var p=document.getElementById("pid").value.trim();if(!p)return toast("请输入PID","err");var b=document.getElementById("btn-scan"),s=document.getElementById("status");b.disabled=true;b.textContent="扫描中...";s.textContent="";var sp=document.createElement("span");sp.className="spin";s.appendChild(sp);s.appendChild(document.createTextNode(" 扫描 PID "+esc(p)+" ..."));api("POST","/scan/"+p+"?force=true").then(function(d){var r=d.data;s.innerHTML="";var ok=document.createElement("span");ok.className="tag-inline tag-suc";ok.textContent="完成";s.appendChild(ok);s.appendChild(document.createTextNode(" PID:"+r.pid+" TID:"+r.tid+" "+esc(r.forum_name)+" - "+r.ocr_images+"/"+r.total_images+" 张已OCR，共 "+r.ocr_text.length+" 字"));loadResults()}).catch(function(e){s.innerHTML="";var er=document.createElement("span");er.className="tag-inline tag-dng";er.textContent="失败";s.appendChild(er);s.appendChild(document.createTextNode(" "+esc(e.message)))}).finally(function(){b.disabled=false;b.textContent="扫描"})}
function esc(s){if(typeof s!=="string")return"";return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;")}
function loadResults(){var e=document.getElementById("results");api("GET","/list").then(function(d){if(!d.data.length){e.innerHTML='<div class="empty">暂无记录（规则触发或手动扫描后会出现）</div>';return}var f=document.createDocumentFragment();d.data.forEach(function(r){var dt=new Date(r.scan_time*1000).toLocaleString("zh-CN");var src=r.trigger=="manual"?"手动":"自动";var sc=r.trigger=="manual"?"tag-war":"tag-suc";var div=document.createElement("div");div.className="result";var flex=document.createElement("div");flex.className="flex";var ls=document.createElement("span");var pt=document.createElement("span");pt.className="tag-inline tag-info";pt.textContent="PID "+r.pid;ls.appendChild(pt);ls.appendChild(document.createTextNode(" "));var ot=document.createElement("span");ot.className="tag-inline tag-war";ot.textContent=r.ocr_images+"/"+r.total_images+" OCR";ls.appendChild(ot);ls.appendChild(document.createTextNode(" "));var st=document.createElement("span");st.className="tag-inline "+sc;st.textContent=src;ls.appendChild(st);flex.appendChild(ls);var ds=document.createElement("span");ds.style.cssText="font-size:12px;color:#909399";ds.textContent=dt;flex.appendChild(ds);div.appendChild(flex);var meta=document.createElement("div");meta.style.cssText="font-size:12px;color:#606266;margin-top:4px";meta.textContent="TID:"+r.tid+" | "+esc(r.forum_name)+" | 共 "+r.ocr_text.length+" 字";div.appendChild(meta);var ld=document.createElement("div");ld.style.cssText="margin-top:2px";var a=document.createElement("a");a.href="https://tieba.baidu.com/p/"+r.tid;a.target="_blank";a.style.cssText="font-size:12px;color:#409eff;text-decoration:none";a.textContent="打开帖子";ld.appendChild(a);div.appendChild(ld);(r.images||[]).forEach(function(img){var det=document.createElement("div");det.style.cssText="margin-top:8px;border:1px solid #ebeef5;border-radius:6px;overflow:hidden";var hdr=document.createElement("div");hdr.style.cssText="background:#f8f9fa;padding:6px 12px;font-size:12px;color:#909399;display:flex;justify-content:space-between";var ds2=document.createElement("span");ds2.textContent=img.width+"x"+img.height+" | "+img.text_length+"字";hdr.appendChild(ds2);var us=document.createElement("span");us.style.cssText="word-break:break-all;max-width:70%";us.textContent=img.url;hdr.appendChild(us);det.appendChild(hdr);var pre=document.createElement("pre");pre.className="ocr-text";pre.style.cssText="margin:0;border:none;border-radius:0;max-height:120px";pre.textContent=img.text;det.appendChild(pre);div.appendChild(det)});f.appendChild(div)});e.innerHTML="";e.appendChild(f)}).catch(function(err){e.innerHTML='<div class="empty">加载失败</div>'})}function clearCache(){if(!confirm("确定清除所有 OCR 缓存？此操作不可恢复。"))return;var b=document.getElementById("btn-clear");b.disabled=true;b.textContent="清除中...";api("POST","/cache/clear").then(function(){toast("缓存已清除","ok");loadResults()}).catch(function(e){toast(e.message,"err")}).finally(function(){b.disabled=false;b.textContent="清除所有缓存"})}
loadConfig();loadResults();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/ocr-review", response_class=HTMLResponse)
async def ocr_page():
    return PAGE


router = APIRouter(prefix="/api/plugin/ocr-review", tags=["ocr-review"])


@router.get("/list")
async def list_results() -> BaseResponse[list[ScanResult]]:
    results: list[ScanResult] = []
    for f in sorted(RESULTS_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            results.append(ScanResult(**json.loads(f.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return BaseResponse(data=results)


@router.post("/scan/{pid}")
async def scan_post(pid: int, force: bool = False) -> BaseResponse[ScanResult]:
    if force:
        p = _result_path(pid)
        if p.exists():
            p.unlink()
    try:
        text = await _ocr_post(pid)
        cached = _load_cache(pid)
        if cached is None:
            raise HTTPException(status_code=404, detail=f"Post {pid} not found")
        cached.trigger = "manual"
        return BaseResponse(data=cached)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/config")
async def get_config() -> BaseResponse[OCRConfig]:
    return BaseResponse(data=_load_config())


@router.put("/config")
async def update_config(req: OCRConfig) -> BaseResponse[OCRConfig]:
    if not 1 <= req.max_images <= 50:
        raise HTTPException(status_code=400, detail="max_images must be 1-50")
    _save_config(req)
    system_logger.info(f"[ocr_review] config: max_images={req.max_images}")
    return BaseResponse(data=req, message="saved")


@router.post("/cache/clear")
async def clear_cache() -> BaseResponse[int]:
    count = 0
    for f in RESULTS_DIR.glob("*.json"):
        f.unlink()
        count += 1
    system_logger.info(f"[ocr_review] cache cleared: {count} files")
    return BaseResponse(data=count, message=f"Deleted {count} cache files")


app.include_router(router)
system_logger.info("[ocr_review] loaded, condition: ImageOCRHitCondition, page: /ocr-review")