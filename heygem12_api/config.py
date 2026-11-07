"""Service configuration, read from environment variables (HEYGEM12_API_*)."""
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HEYGEM12_API_{name}", default)


@dataclass
class Settings:
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8800")))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA", str(REPO_ROOT / "data"))))
    # Base URL clients use to reach this server (for video_url / webhooks). If empty,
    # derived from the incoming request's Host header.
    public_url: str = field(default_factory=lambda: _env("PUBLIC_URL", "").rstrip("/"))
    checkpoints: str = field(default_factory=lambda: _env("CHECKPOINTS", str(REPO_ROOT / "checkpoints")))
    device: str = field(default_factory=lambda: _env("DEVICE", "cuda"))
    ffmpeg: str = field(default_factory=lambda: _env("FFMPEG", "ffmpeg"))
    ffprobe: str = field(default_factory=lambda: _env("FFPROBE", "ffprobe"))
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "1024")))
    # Longest output accepted (seconds). All frames are held in RAM during synthesis.
    max_duration_s: float = field(default_factory=lambda: float(_env("MAX_DURATION", "600")))
    allow_private_urls: bool = field(default_factory=lambda: _env("ALLOW_PRIVATE_URLS", "0") == "1")
    # Job watchdog: budget = base + per_frame * frames; on overrun the GPU worker is killed.
    job_timeout_base_s: float = field(default_factory=lambda: float(_env("JOB_TIMEOUT_BASE", "300")))
    job_timeout_per_frame_s: float = field(default_factory=lambda: float(_env("JOB_TIMEOUT_PER_FRAME", "0.5")))
    url_ttl_s: int = field(default_factory=lambda: int(_env("URL_TTL", str(7 * 24 * 3600))))
    warmup: bool = field(default_factory=lambda: _env("WARMUP", "1") == "1")

    @property
    def assets_dir(self) -> Path: return self.data_dir / "assets"
    @property
    def outputs_dir(self) -> Path: return self.data_dir / "outputs"
    @property
    def work_dir(self) -> Path: return self.data_dir / "work"
    @property
    def cache_dir(self) -> Path: return self.data_dir / "cache"
    @property
    def logs_dir(self) -> Path: return self.data_dir / "logs"
    @property
    def db_path(self) -> Path: return self.data_dir / "heygem12_api.sqlite3"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.assets_dir, self.outputs_dir, self.work_dir,
                  self.cache_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # --- secrets kept in the data dir -------------------------------------------------
    def _secret_file(self, name: str) -> str:
        p = self.data_dir / name
        if not p.exists():
            p.write_text(secrets.token_hex(32) + "\n", encoding="ascii")
        return p.read_text(encoding="ascii").strip()

    @property
    def url_signing_secret(self) -> bytes:
        return self._secret_file("url_signing.secret").encode()

    @property
    def webhook_secret(self) -> bytes:
        return self._secret_file("webhook.secret").encode()

    def api_keys(self) -> set:
        """API keys: one per line in data/api_keys.txt ('#' comments allowed).
        A key is generated on first start."""
        p = self.data_dir / "api_keys.txt"
        if not p.exists():
            p.write_text("# one API key per line\n" + "hg12_" + secrets.token_urlsafe(32) + "\n",
                         encoding="ascii")
        keys = set()
        for line in p.read_text(encoding="ascii").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                keys.add(line)
        return keys


settings = Settings()
