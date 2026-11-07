"""HeyGen-v3-compatible HTTP API for HeyGem12 lip-sync.

Implements the /v3/lipsyncs and /v3/assets subset of HeyGen's API
(https://developers.heygen.com), backed by the local HeyGem12 pipeline.
"""
import base64
import hashlib
import hmac
import json
import logging
import re
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .scheduler import Scheduler
from .store import Store, new_id

log = logging.getLogger("heygem12_api")

DOC_URL = "https://github.com/wizd/HeyGem12/blob/main/docs/API.md"
MEDIA_EXT = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
    ".m4v": "video/x-m4v", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".flac": "audio/flac",
}
IDEM_RE = re.compile(r"^[A-Za-z0-9_:.\-]{1,255}$")


# ------------------------------------------------------------------ errors
class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, param: Optional[str] = None):
        self.status, self.code, self.message, self.param = status, code, message, param


def error_body(code: str, message: str, param: Optional[str] = None) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message, "param": param, "doc_url": DOC_URL}}


# ----------------------------------------------------------- signed URLs
def _sign(kind: str, ident: str, expires: int) -> str:
    return hmac.new(settings.url_signing_secret, f"{kind}:{ident}:{expires}".encode(),
                    hashlib.sha256).hexdigest()[:32]


def signed_url(kind: str, ident: str, base_url: Optional[str]) -> str:
    exp = int(time.time()) + settings.url_ttl_s
    base = settings.public_url or (base_url or "").rstrip("/")
    return f"{base}/v3/files/{kind}/{ident}?expires={exp}&signature={_sign(kind, ident, exp)}"


def _base_url(request: Request) -> str:
    return settings.public_url or str(request.base_url).rstrip("/")


# --------------------------------------------------------------- app state
settings.ensure_dirs()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(settings.logs_dir / "server.log", encoding="utf-8")])
store = Store(str(settings.db_path))
scheduler = Scheduler(store, signed_url)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    keys = settings.api_keys()
    log.info("API keys loaded from %s (%d key(s))", settings.data_dir / "api_keys.txt", len(keys))
    scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(title="HeyGem12 API (HeyGen v3 lipsync-compatible)", version="1.0.0", lifespan=lifespan)


@app.exception_handler(ApiError)
async def _api_error(_req: Request, exc: ApiError):
    return JSONResponse(error_body(exc.code, exc.message, exc.param), status_code=exc.status)


@app.exception_handler(RequestValidationError)
async def _validation_error(_req: Request, exc: RequestValidationError):
    e = exc.errors()[0] if exc.errors() else {}
    param = ".".join(str(x) for x in e.get("loc", [])[1:]) or None
    return JSONResponse(error_body("invalid_parameter", e.get("msg", "invalid request"), param),
                        status_code=400)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_req: Request, exc: StarletteHTTPException):
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "error")
    return JSONResponse(error_body(code, str(exc.detail)), status_code=exc.status_code)


OPEN_PATHS = ("/health", "/v3/files/", "/docs", "/openapi.json", "/redoc")


@app.middleware("http")
async def _auth(request: Request, call_next):
    path = request.url.path
    if not any(path == p or (p.endswith("/") and path.startswith(p)) for p in OPEN_PATHS):
        key = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        if not key and auth.lower().startswith("bearer "):
            key = auth[7:].strip()
        if not key or not any(hmac.compare_digest(key, k) for k in settings.api_keys()):
            return JSONResponse(error_body("unauthorized", "missing or invalid API key "
                                           "(send x-api-key or Authorization: Bearer)"), status_code=401)
    return await call_next(request)


# --------------------------------------------------------------- helpers
async def _json_body(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        raise ApiError(400, "invalid_json", "request body must be valid JSON")
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_parameter", "request body must be a JSON object")
    return body


def _enc_token(offset: int) -> str:
    return base64.urlsafe_b64encode(f"o:{offset}".encode()).decode().rstrip("=")


def _dec_token(token: Optional[str]) -> int:
    if not token:
        return 0
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
        assert raw.startswith("o:")
        return max(0, int(raw[2:]))
    except Exception:
        raise ApiError(400, "invalid_parameter", "invalid pagination token", "token")


def _limit(v: Optional[str], default: int) -> int:
    try:
        n = int(v) if v is not None else default
    except ValueError:
        raise ApiError(400, "invalid_parameter", "limit must be an integer", "limit")
    if not 1 <= n <= 100:
        raise ApiError(400, "invalid_parameter", "limit must be between 1 and 100", "limit")
    return n


def _input_spec(v: Any, param: str) -> Dict[str, str]:
    if not isinstance(v, dict) or v.get("type") not in ("url", "asset_id"):
        raise ApiError(400, "invalid_parameter",
                       f"{param} must be {{type: 'url', url}} or {{type: 'asset_id', asset_id}}", param)
    if v["type"] == "url":
        url = v.get("url")
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            raise ApiError(400, "invalid_parameter", f"{param}.url must be an http(s) URL", f"{param}.url")
        return {"type": "url", "url": url}
    aid = v.get("asset_id")
    if not isinstance(aid, str) or not store.get_asset(aid):
        raise ApiError(400, "asset_not_found", f"{param}.asset_id does not exist", f"{param}.asset_id")
    return {"type": "asset_id", "asset_id": aid}


def _validate_lipsync(body: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    p = lambda name: f"{prefix}{name}"  # noqa: E731
    for name in ("video", "audio"):
        if name not in body:
            raise ApiError(400, "invalid_parameter", f"{p(name)} is required", p(name))
    req: Dict[str, Any] = {"video": _input_spec(body["video"], p("video")),
                           "audio": _input_spec(body["audio"], p("audio"))}
    mode = body.get("mode", "speed") or "speed"
    if mode not in ("speed", "precision"):
        raise ApiError(400, "invalid_parameter", "mode must be 'speed' or 'precision'", p("mode"))
    req["mode"] = mode
    for name in ("title", "callback_id", "callback_url", "folder_id", "fps_mode"):
        v = body.get(name)
        if v is not None and not isinstance(v, str):
            raise ApiError(400, "invalid_parameter", f"{p(name)} must be a string", p(name))
        req[name] = v
    if req["callback_url"] and not re.match(r"^https?://", req["callback_url"]):
        raise ApiError(400, "invalid_parameter", "callback_url must be an http(s) URL", p("callback_url"))
    if req["folder_id"]:
        raise ApiError(404, "not_found", "folders are not supported by this server", p("folder_id"))
    if req["fps_mode"] not in (None, "", "cfr", "passthrough", "vfr"):
        raise ApiError(400, "invalid_parameter", "fps_mode must be 'vfr', 'cfr' or 'passthrough'", p("fps_mode"))
    dyn = body.get("enable_dynamic_duration", True)
    if not isinstance(dyn, bool):
        raise ApiError(400, "invalid_parameter", "enable_dynamic_duration must be a boolean",
                       p("enable_dynamic_duration"))
    req["enable_dynamic_duration"] = dyn
    # Accepted for compatibility, no effect: captions are never produced; the source
    # resolution is always kept.
    for name in ("enable_caption", "keep_the_same_format"):
        if body.get(name) not in (None, True, False):
            raise ApiError(400, "invalid_parameter", f"{p(name)} must be a boolean", p(name))
    # Features HeyGen offers that this server cannot do: refuse instead of ignoring.
    for name in ("disable_music_track", "enable_speech_enhancement", "enable_watermark"):
        if body.get(name):
            raise ApiError(400, "unsupported_parameter", f"{p(name)}=true is not supported by this server",
                           p(name))
    for name in ("start_time", "end_time"):
        if body.get(name) is not None:
            raise ApiError(400, "unsupported_parameter", f"{p(name)} (partial lipsync) is not supported",
                           p(name))
    return req


def _lipsync_view(row: Dict[str, Any]) -> Dict[str, Any]:
    done = row["status"] == "completed"
    return {
        "id": row["id"], "title": row["title"], "status": row["status"],
        "callback_id": row["callback_id"], "created_at": row["created_at"],
        "duration": row["duration"],
        "video_url": signed_url("output", row["id"], row["base_url"]) if done else None,
        "caption_url": None,
        "failure_code": row["failure_code"] if row["status"] == "failed" else None,
        "failure_message": row["failure_message"] if row["status"] == "failed" else None,
    }


STATUS_V2 = {"pending": "queued", "running": "processing", "completed": "completed", "failed": "failed"}


async def _idempotent(request: Request, scope: str, fn):
    """Run fn() -> (status, body) honoring an optional Idempotency-Key header."""
    key = request.headers.get("idempotency-key")
    if not key:
        status, body = await fn()
        return JSONResponse(body, status_code=status)
    if not IDEM_RE.match(key):
        raise ApiError(400, "invalid_parameter", "Idempotency-Key must be 1-255 chars of [A-Za-z0-9_:.-]",
                       "Idempotency-Key")
    prev = store.idem_get(key, scope)
    if prev:
        if prev["status_code"] is None:
            raise ApiError(409, "request_in_progress", "a request with this Idempotency-Key is in progress")
        return JSONResponse(json.loads(prev["response"]), status_code=prev["status_code"])
    if not store.idem_reserve(key, scope):
        raise ApiError(409, "request_in_progress", "a request with this Idempotency-Key is in progress")
    try:
        status, body = await fn()
    except Exception:
        store.idem_release(key, scope)
        raise
    store.idem_store(key, scope, status, body)
    return JSONResponse(body, status_code=status)


# ================================================================= routes
@app.get("/health")
def health():
    return {"status": "ok" if scheduler.state != "error" else "degraded",
            "worker": scheduler.state, "queue_length": store.queue_length(),
            "running": scheduler.current is not None}


# ---------------------------------------------------------------- assets
@app.post("/v3/assets")
async def upload_asset(request: Request, file: UploadFile = File(...)):
    async def go():
        ext = Path(file.filename or "").suffix.lower()
        if ext not in MEDIA_EXT:
            raise ApiError(400, "unsupported_file_type",
                           f"unsupported file type {ext or '(none)'}; allowed: {', '.join(sorted(MEDIA_EXT))}",
                           "file")
        aid = new_id()
        dest = settings.assets_dir / f"{aid}{ext}"
        limit = settings.max_upload_mb * 1024 * 1024
        size = 0
        with open(dest, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise ApiError(413, "file_too_large", f"file exceeds {settings.max_upload_mb} MB", "file")
                f.write(chunk)
        if size == 0:
            dest.unlink(missing_ok=True)
            raise ApiError(400, "invalid_parameter", "file is empty", "file")
        mime = MEDIA_EXT[ext]
        store.add_asset(aid, file.filename, str(dest), mime, size)
        return 200, {"data": {"asset_id": aid, "url": signed_url("asset", aid, _base_url(request)),
                              "mime_type": mime, "size_bytes": size}}
    return await _idempotent(request, "POST /v3/assets", go)


@app.get("/v3/assets/{asset_id}")
def get_asset(asset_id: str, request: Request):
    a = store.get_asset(asset_id)
    if not a:
        raise ApiError(404, "not_found", "asset not found", "asset_id")
    return {"data": {"id": a["id"], "asset_id": a["id"], "name": a["filename"],
                     "file_type": a["mime_type"], "mime_type": a["mime_type"],
                     "size_bytes": a["size_bytes"], "created_at": a["created_at"],
                     "url": signed_url("asset", a["id"], _base_url(request))}}


@app.delete("/v3/assets/{asset_id}")
def delete_asset(asset_id: str):
    a = store.get_asset(asset_id)
    if not a:
        raise ApiError(404, "not_found", "asset not found", "asset_id")
    store.delete_asset(asset_id)
    Path(a["path"]).unlink(missing_ok=True)
    return {"data": {"asset_id": asset_id}}


# -------------------------------------------------------------- lipsyncs
@app.post("/v3/lipsyncs")
async def create_lipsync(request: Request):
    async def go():
        req = _validate_lipsync(await _json_body(request))
        lid = store.add_lipsync(req, _base_url(request))
        scheduler.notify()
        return 200, {"data": {"lipsync_id": lid}}
    return await _idempotent(request, "POST /v3/lipsyncs", go)


@app.get("/v3/lipsyncs")
def list_lipsyncs(limit: Optional[str] = None, token: Optional[str] = None):
    n, off = _limit(limit, 10), _dec_token(token)
    rows = store.list_lipsyncs(n, off)
    more = len(rows) > n
    return {"data": [_lipsync_view(r) for r in rows[:n]], "has_more": more,
            "next_token": _enc_token(off + n) if more else None}


@app.get("/v3/lipsyncs/statuses")
def lipsync_statuses(lipsync_ids: Optional[str] = None, batch_ids: Optional[str] = None):
    ids = [x for x in (lipsync_ids or "").split(",") if x.strip()]
    bids = [x for x in (batch_ids or "").split(",") if x.strip()]
    if not ids and not bids:
        raise ApiError(400, "invalid_parameter", "provide lipsync_ids and/or batch_ids", "lipsync_ids")
    items: List[Dict[str, Any]] = []
    for lid in ids:
        r = store.get_lipsync(lid.strip())
        items.append({"video_id": lid.strip(), "status": STATUS_V2[r["status"]] if r else "not_found",
                      "error": _err(r)})
    for bid in bids:
        bid = bid.strip()
        if not store.get_batch(bid):
            items.append({"video_id": None, "batch_id": bid, "status": "not_found"})
            continue
        for r in store.batch_items(bid, 10_000, 0):
            items.append({"video_id": r["id"], "batch_id": bid, "item_index": r["item_index"],
                          "status": STATUS_V2[r["status"]], "error": _err(r)})
    if len(items) > 100:
        raise ApiError(400, "invalid_parameter", "at most 100 statuses per request")
    return {"data": items, "has_more": False, "next_token": None}


def _err(r):
    if r and r["status"] == "failed":
        return {"code": r["failure_code"], "message": r["failure_message"]}
    return None


@app.get("/v3/lipsyncs/{lipsync_id}")
def get_lipsync(lipsync_id: str):
    r = store.get_lipsync(lipsync_id)
    if not r:
        raise ApiError(404, "not_found", "lipsync not found", "lipsync_id")
    return {"data": _lipsync_view(r)}


@app.patch("/v3/lipsyncs/{lipsync_id}")
async def update_lipsync(lipsync_id: str, request: Request):
    body = await _json_body(request)
    if not isinstance(body.get("title"), str):
        raise ApiError(400, "invalid_parameter", "title is required", "title")
    if not store.get_lipsync(lipsync_id):
        raise ApiError(404, "not_found", "lipsync not found", "lipsync_id")
    store.update_lipsync(lipsync_id, title=body["title"])
    return {"data": _lipsync_view(store.get_lipsync(lipsync_id))}


@app.delete("/v3/lipsyncs/{lipsync_id}")
def delete_lipsync(lipsync_id: str):
    r = store.get_lipsync(lipsync_id)
    if not r:
        raise ApiError(404, "not_found", "lipsync not found", "lipsync_id")
    if r["status"] == "running":
        raise ApiError(409, "lipsync_in_progress", "lipsync is processing; delete it after it finishes")
    store.delete_lipsync(lipsync_id)
    if r["output_path"]:
        Path(r["output_path"]).unlink(missing_ok=True)
    return {"data": {"id": lipsync_id}}


# --------------------------------------------------------------- batches
@app.post("/v3/lipsyncs/batches", status_code=202)
async def create_batch(request: Request):
    async def go():
        body = await _json_body(request)
        items = body.get("lipsyncs")
        if not isinstance(items, list) or not 1 <= len(items) <= 100:
            raise ApiError(400, "invalid_parameter", "lipsyncs must be a list of 1-100 items", "lipsyncs")
        for name in ("title", "callback_url"):
            if body.get(name) is not None and not isinstance(body[name], str):
                raise ApiError(400, "invalid_parameter", f"{name} must be a string", name)
        reqs = [_validate_lipsync(it if isinstance(it, dict) else {}, f"lipsyncs[{i}].")
                for i, it in enumerate(items)]
        base = _base_url(request)
        bid = store.add_batch(body.get("title"), body.get("callback_url"), base, len(reqs))
        for i, req in enumerate(reqs):
            store.add_lipsync(req, base, batch_id=bid, item_index=i)
        scheduler.notify()
        return 202, {"data": {"batch_id": bid}}
    return await _idempotent(request, "POST /v3/lipsyncs/batches", go)


@app.get("/v3/lipsyncs/batches/{batch_id}")
def get_batch(batch_id: str, limit: Optional[str] = None, token: Optional[str] = None):
    b = store.get_batch(batch_id)
    if not b:
        raise ApiError(404, "not_found", "batch not found", "batch_id")
    n, off = _limit(limit, 100), _dec_token(token)
    rows = store.batch_items(batch_id, n, off)
    more = len(rows) > n
    raw = store.batch_counts(batch_id)
    counts: Dict[str, int] = {}
    for k, v in raw.items():
        counts[STATUS_V2[k]] = counts.get(STATUS_V2[k], 0) + v
    terminal = counts.get("completed", 0) + counts.get("failed", 0)
    if terminal < b["total_items"]:
        status = "processing"
    else:
        status = "failed" if counts.get("failed", 0) == b["total_items"] else "completed"
    return {"data": {
        "batch_id": batch_id, "title": b["title"], "status": status, "total_items": b["total_items"],
        "counts_by_status": counts, "created_at": b["created_at"],
        "items": [{"item_index": r["item_index"], "status": STATUS_V2[r["status"]], "video_id": r["id"],
                   "error": _err(r)} for r in rows[:n]],
        "has_more": more, "next_token": _enc_token(off + n) if more else None}}


# ------------------------------------------------- signed file downloads
@app.get("/v3/files/{kind}/{ident}")
def download_file(kind: str, ident: str, expires: int = 0, signature: str = ""):
    if kind not in ("output", "asset") or expires < time.time() or \
            not hmac.compare_digest(signature, _sign(kind, ident, expires)):
        raise ApiError(403, "forbidden", "invalid or expired link")
    if kind == "output":
        r = store.get_lipsync(ident)
        path = r and r["output_path"]
        name, mime = f"{ident}.mp4", "video/mp4"
    else:
        a = store.get_asset(ident)
        path = a and a["path"]
        name, mime = (a["filename"] if a else ident), (a["mime_type"] if a else None)
    if not path or not Path(path).exists():
        raise ApiError(404, "not_found", "file not found")
    return FileResponse(path, media_type=mime, filename=name)
