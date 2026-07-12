from __future__ import annotations

import unittest

from sam3_service.exporter import (
    Centerline,
    DegreeLabel,
    ExportOptions,
    _degree_label_entries,
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

    def test_degree_label_entries_use_short_text_and_highlight_outlier(self) -> None:
        labels = [
            DegreeLabel(
                instance_id=f"paddle:{index}",
                degree=degree,
                line=(0, 0, 1, 1),
                color=(53, 194, 255, 255),
            )
            for index, degree in enumerate([40, 42, 41, 75], start=1)
        ]

        entries = _degree_label_entries(labels)

        self.assertEqual(
            [entry.text for entry in entries],
            ["1: 40°", "2: 42°", "3: 41°", "4: 75°"],
        )
        self.assertEqual(entries[3].text_color, (255, 82, 96, 255))
        self.assertTrue(all(entry.text_color != (255, 82, 96, 255) for entry in entries[:3]))


if __name__ == "__main__":
    unittest.main()
