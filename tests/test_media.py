from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sam3_service.media import probe_video


class MediaProbeTest(unittest.TestCase):
    @patch("sam3_service.media.subprocess.run")
    def test_reported_video_frame_count_wins_over_longer_container_duration(
        self, run: Mock
    ) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "width": 1278,
                            "height": 720,
                            "codec_name": "h264",
                            "avg_frame_rate": "30/1",
                            "nb_frames": "536",
                            "duration": "17.866667",
                        }
                    ],
                    "format": {"duration": "17.900000"},
                }
            ),
            stderr="",
        )

        metadata = probe_video(Path("normalized.mp4"))

        self.assertEqual(metadata["frame_count"], 536)
        self.assertEqual(metadata["duration_ms"], 17867)

    @patch("sam3_service.media.subprocess.run")
    def test_duration_estimates_frame_count_when_stream_count_is_absent(
        self, run: Mock
    ) -> None:
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "width": 1280,
                            "height": 720,
                            "codec_name": "h264",
                            "avg_frame_rate": "30000/1001",
                            "nb_frames": "N/A",
                            "duration": "10.010000",
                        }
                    ],
                    "format": {"duration": "10.050000"},
                }
            ),
            stderr="",
        )

        metadata = probe_video(Path("normalized.mp4"))

        self.assertEqual(metadata["frame_count"], 300)
        self.assertEqual(metadata["duration_ms"], 10010)


if __name__ == "__main__":
    unittest.main()
