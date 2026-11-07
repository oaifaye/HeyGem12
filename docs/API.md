# HeyGem12 API server

A self-hosted HTTP API for HeyGem12 lip-sync, compatible with the **lipsync and
asset endpoints of HeyGen's v3 API** (`/v3/lipsyncs`, `/v3/assets`). A client
written against HeyGen's lipsync API should only need a different base URL and
API key.

```
POST /v3/assets                      upload a video/audio file   -> asset_id
POST /v3/lipsyncs                    start a job (video + audio) -> lipsync_id
GET  /v3/lipsyncs/{id}               poll: pending | running | completed | failed (+ video_url)
```

## Running

```bat
uv pip install -r requirements-api.txt        (into the same venv as HeyGem12)
python -m heygem12_api
```

`ffmpeg` and `ffprobe` must be on PATH. On first start the server creates
`data\api_keys.txt` with one generated key. Add or remove keys there (one per line;
the file is read on every request, so no restart is needed).

| Environment variable | Default | Meaning |
|---|---|---|
| `HEYGEM12_API_HOST` / `_PORT` | `0.0.0.0` / `8800` | Listen address |
| `HEYGEM12_API_PUBLIC_URL` | (request host) | Base URL used in `video_url`s and webhooks, e.g. `https://lipsync.example.com` |
| `HEYGEM12_API_DATA` | `.\data` | Database, uploaded assets, outputs, logs, secrets |
| `HEYGEM12_API_DEVICE` | `cuda` | torch device (`cuda` also covers ROCm GPUs) |
| `HEYGEM12_API_MAX_UPLOAD_MB` | `1024` | Upload / URL-download size limit |
| `HEYGEM12_API_MAX_DURATION` | `600` | Longest output in seconds (frames are held in RAM, ~6 GB per minute at 720p) |
| `HEYGEM12_API_ALLOW_PRIVATE_URLS` | `0` | Allow `url` inputs and `callback_url`s on private or loopback addresses |
| `HEYGEM12_API_JOB_TIMEOUT_BASE` / `_PER_FRAME` | `300` / `0.5` | Watchdog budget per job; on overrun the GPU worker is killed and restarted |
| `HEYGEM12_API_URL_TTL` | `604800` | Lifetime of signed download links (seconds) |
| `HEYGEM12_API_WARMUP` | `1` | Warm up the GPU when the worker starts (~16 s) so the first job isn't slow |

## Conventions (same as HeyGen v3)

- **Auth:** `x-api-key: <key>` or `Authorization: Bearer <key>`.
- **Success:** `{"data": ...}`. **Errors:** `{"error": {"code", "message", "param", "doc_url"}}`
  with HTTP 400/401/403/404/409/413.
- **Idempotency:** optional `Idempotency-Key` header on POSTs (1–255 chars of `[A-Za-z0-9_:.-]`).
  A repeat within 24 h replays the original response; a concurrent repeat gets 409 `request_in_progress`.
- **Pagination:** `limit` (1–100) and an opaque `token`; responses carry `has_more` and `next_token`.
- **Download URLs** (`video_url`, asset `url`) are signed and expire (default 7 days). They need no API key.

## Endpoints

### `POST /v3/assets`: upload a file
`multipart/form-data` with a `file` field. Types: mp4 mov webm mkv m4v mp3 wav m4a aac ogg flac.
```json
{"data": {"asset_id": "…", "url": "…signed…", "mime_type": "video/mp4", "size_bytes": 8360134}}
```
`GET /v3/assets/{asset_id}` returns metadata and a fresh signed `url`. `DELETE /v3/assets/{asset_id}` removes the asset.

### `POST /v3/lipsyncs`: create a lipsync job
```jsonc
{
  "video": {"type": "asset_id", "asset_id": "…"},         // or {"type": "url", "url": "https://…"}
  "audio": {"type": "url", "url": "https://…/voice.mp3"},  // or {"type": "asset_id", …}
  "title": "optional",
  "mode": "speed",                     // "precision" = GFPGAN face restoration before lip-sync (~2x slower)
  "enable_dynamic_duration": true,     // see below
  "callback_url": "https://…", "callback_id": "your-id"
}
```
Response: `{"data": {"lipsync_id": "…"}}`

**Output length.** With `enable_dynamic_duration: true` (the default) the output matches the
audio length. If the audio is longer than the video, the video is extended by
ping-pong looping (forward, reversed, forward …), which avoids a visible jump at the
loop point; if the audio is shorter, the video is trimmed. With `false`, the output
stops at the shorter of the video and the audio.

**Differences from HeyGen.**
- Accepted but no effect: `enable_caption` and `keep_the_same_format`. No captions are
  produced (`caption_url` is always null), and the source resolution is always kept.
- Rejected with 400 `unsupported_parameter`: `disable_music_track`,
  `enable_speech_enhancement` or `enable_watermark` set to true, and `start_time`/`end_time`.
- A non-empty `folder_id` gets 404.

### `GET /v3/lipsyncs/{lipsync_id}`
```json
{"data": {"id": "…", "title": "…", "status": "completed", "callback_id": "…",
          "created_at": 1794067000, "duration": 63.36,
          "video_url": "…signed…", "caption_url": null,
          "failure_code": null, "failure_message": null}}
```
`status` goes `pending` → `running` → `completed` or `failed`. Jobs run one at a time on the GPU, in submission order.

Also available:
- `GET /v3/lipsyncs`: list jobs, newest first.
- `PATCH /v3/lipsyncs/{id}`: rename (`{"title"}`).
- `DELETE /v3/lipsyncs/{id}`: delete a job and its output (409 while it is running).
- `GET /v3/lipsyncs/statuses?lipsync_ids=a,b&batch_ids=c`: statuses
  `queued | processing | completed | failed | not_found`.

### Batches
`POST /v3/lipsyncs/batches` returns **202** `{"data": {"batch_id"}}`.
```jsonc
{"title": "optional",
 "callback_url": "optional; called once when every item has finished",
 "lipsyncs": [ {…same shape as POST /v3/lipsyncs…}, … ]}   // 1–100 items
```
`GET /v3/lipsyncs/batches/{batch_id}` returns `status` (`processing | completed | failed`),
`counts_by_status`, and `items[]` with `item_index`, `status`, `video_id` (= lipsync id) and `error`.

### Webhooks
When a job with a `callback_url` finishes, the server POSTs JSON to it (up to 4 attempts):
```json
{"event_type": "avatar_video.success",
 "event_data": {"video_id": "…", "lipsync_id": "…", "url": "…signed…", "duration": 63.36, "callback_id": "…"}}
{"event_type": "avatar_video.fail",
 "event_data": {"video_id": "…", "lipsync_id": "…", "msg": "…", "failure_code": "…", "callback_id": "…"}}
```
A batch-level `callback_url` receives `{"event_type": "lipsync_batch.complete", "event_data":
{"batch_id", "status", "counts_by_status"}}` once. Items get their own webhook only if
they set their own `callback_url`.

Every webhook request carries a `Signature` header: the hex HMAC-SHA256 of the raw body,
keyed with the secret in `data\webhook.secret`. The event names follow HeyGen's
`avatar_video.*` events. HeyGen does not publish a lipsync-specific event name, so this
part of the format is HeyGem12's own.

### Other
- `GET /health` (no auth): `{"status", "worker", "queue_length", "running"}`.
- Interactive docs at `/docs`.

## Failure codes
`invalid_video`, `invalid_audio`, `invalid_media`, `download_failed`, `invalid_url`,
`file_too_large`, `duration_too_long`, `asset_not_found`, `processing_failed`
(pipeline error, e.g. no face found), `timeout` (the watchdog killed the GPU worker),
`worker_crashed`, `internal_error`.

## Example (curl)
```bash
KEY=hg12_…; API=http://server:8800
V=$(curl -s -H "x-api-key: $KEY" -F file=@face.mp4  $API/v3/assets | jq -r .data.asset_id)
A=$(curl -s -H "x-api-key: $KEY" -F file=@voice.wav $API/v3/assets | jq -r .data.asset_id)
ID=$(curl -s -H "x-api-key: $KEY" -H 'Content-Type: application/json' $API/v3/lipsyncs \
  -d "{\"video\":{\"type\":\"asset_id\",\"asset_id\":\"$V\"},\"audio\":{\"type\":\"asset_id\",\"asset_id\":\"$A\"}}" \
  | jq -r .data.lipsync_id)
curl -s -H "x-api-key: $KEY" $API/v3/lipsyncs/$ID      # poll until status is "completed", then download video_url
```

## Operational notes
- One GPU worker process keeps the models loaded. If a job exceeds its time budget,
  the worker is killed and restarted and the job fails with `timeout`. A hard GPU
  driver hang can make the worker unkillable; the machine then needs a reboot.
- Jobs left `running` when the server stops are re-queued on the next start.
- Outputs and assets are kept until they are deleted through the API.
- URL inputs and callbacks to private or loopback addresses are refused unless
  `HEYGEM12_API_ALLOW_PRIVATE_URLS=1`.
- The server handles one job at a time; a long job blocks the queue behind it.
