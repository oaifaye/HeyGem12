"""Start the API server:  python -m heygem12_api"""
import uvicorn

from .config import settings


def main() -> None:
    # Single process on purpose: the GPU scheduler lives in this process.
    uvicorn.run("heygem12_api.app:app", host=settings.host, port=settings.port, workers=1,
                log_level="info")


if __name__ == "__main__":
    main()
