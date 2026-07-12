from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from sam3_service.config import Settings
from sam3_service.main import create_app


class CorsTest(unittest.TestCase):
    def settings(self, root: Path, origins: tuple[str, ...] = ()) -> Settings:
        return Settings(
            data_dir=root,
            database_path=root / "sam3.sqlite3",
            static_dir=root / "static",
            segmenter="mock",
            checkpoint_path=None,
            offline=False,
            poll_interval_seconds=0.01,
            max_upload_bytes=20 * 1024 * 1024,
            max_video_duration_seconds=10,
            max_prompts=3,
            chunk_size_bytes=1024 * 1024,
            result_chunk_seconds=1,
            default_redetect_interval_frames=1,
            default_max_detections_per_frame=13,
            max_detections_per_frame_limit=64,
            default_dedupe_iou_threshold=0.6,
            cors_allow_origins=origins,
        )

    def test_cors_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            app = create_app(self.settings(Path(temporary)))
            response = _asgi_request(
                app,
                "GET",
                "/api/v1/health/live",
                {"origin": "http://localhost:5173"},
            )

        self.assertEqual(response["status"], 200)
        self.assertNotIn("access-control-allow-origin", response["headers"])

    def test_configured_origin_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            app = create_app(
                self.settings(Path(temporary), ("http://localhost:5173",))
            )
            response = _asgi_request(
                app,
                "OPTIONS",
                "/api/v1/health/live",
                {
                    "origin": "http://localhost:5173",
                    "access-control-request-method": "GET",
                },
            )

        self.assertEqual(response["status"], 200)
        self.assertEqual(
            response["headers"]["access-control-allow-origin"],
            "http://localhost:5173",
        )


def _asgi_request(
    app: Any,
    method: str,
    path: str,
    headers: dict[str, str],
) -> dict[str, Any]:
    import asyncio

    sent: list[dict[str, Any]] = []
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (name.lower().encode(), value.encode()) for name, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    start = next(message for message in sent if message["type"] == "http.response.start")
    return {
        "status": start["status"],
        "headers": {
            name.decode(): value.decode()
            for name, value in start.get("headers", [])
        },
    }


if __name__ == "__main__":
    unittest.main()
