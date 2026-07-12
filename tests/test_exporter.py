from __future__ import annotations

import unittest

from sam3_service.exporter import (
    Centerline,
    ExportOptions,
    _degree_labels,
)


class ExporterTest(unittest.TestCase):
    def test_computes_degree_label_for_each_target_paddle(self) -> None:
        reference = Centerline(
            record={"prompt_id": "boat", "instance_id": "boat:1"},
            line=(0, 0, 100, 0),
            color=(255, 255, 255, 255),
        )
        targets = [
            Centerline(
                record={"prompt_id": "paddle", "instance_id": f"paddle:{index}"},
                line=(10.0 * index, 0, 10.0 * index, 50),
                color=(53, 194, 255, 255),
            )
            for index in range(1, 5)
        ]

        labels = _degree_labels(
            [reference, *targets],
            ExportOptions(
                reference_prompt_id="boat",
                target_prompt_ids=("paddle",),
            ),
        )

        self.assertEqual(len(labels), 4)
        self.assertEqual([label.degree for label in labels], [90, 90, 90, 90])


if __name__ == "__main__":
    unittest.main()
