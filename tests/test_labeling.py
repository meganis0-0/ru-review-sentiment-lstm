import json

from src.data.labeling import (
    annotation_label,
    apply_pilot_export,
    select_pilot,
    write_labels,
    write_pilot,
)


def _rows() -> list[dict[str, str]]:
    rows = []
    index = 1
    for source in ("bank", "geo"):
        for label in ("negative", "neutral", "positive"):
            for copy in range(4):
                rows.append(
                    {
                        "review_id": str(index),
                        "text": f"{source}-{label}-{copy}",
                        "label": label,
                        "source": source,
                    }
                )
                index += 1
    return rows


def test_select_pilot_covers_each_cell() -> None:
    pilot = select_pilot(_rows(), per_cell=2, extra=2, seed=42)
    cells = {(row["source"], row["label"]) for row in pilot}

    assert len(pilot) == 14
    assert len(cells) == 6
    assert len({row["review_id"] for row in pilot}) == 14


def test_apply_pilot_export_replaces_disagreements(tmp_path) -> None:
    labels = [
        {
            "review_id": "1",
            "text": "плохо",
            "label": "positive",
            "source": "bank",
            "label_origin": "source",
        },
        {
            "review_id": "2",
            "text": "хорошо",
            "label": "positive",
            "source": "bank",
            "label_origin": "source",
        },
    ]
    tasks = [
        {
            "data": {"review_id": "1"},
            "annotations": [
                {
                    "result": [
                        {
                            "from_name": "sentiment",
                            "value": {"choices": ["negative"]},
                        }
                    ]
                }
            ],
        },
        {
            "data": {"review_id": "2"},
            "annotations": [
                {
                    "result": [
                        {
                            "from_name": "sentiment",
                            "value": {"choices": ["positive"]},
                        }
                    ]
                }
            ],
        },
    ]

    stats = apply_pilot_export(labels, tasks)

    assert stats == {"compared": 2, "matched": 1, "updated": 2}
    assert labels[0]["label"] == "negative"
    assert labels[0]["label_origin"] == "manual"
    assert labels[1]["label_origin"] == "manual"


def test_annotation_label_ignores_empty_task() -> None:
    assert annotation_label({"annotations": []}) is None


def test_write_pilot_and_labels_roundtrip(tmp_path) -> None:
    rows = _rows()[:1]
    rows[0]["label_origin"] = "source"
    labels_path = tmp_path / "labels.csv"
    pilot_path = tmp_path / "pilot.json"

    write_labels(labels_path, rows)
    write_pilot(pilot_path, rows)
    tasks = json.loads(pilot_path.read_text(encoding="utf-8"))

    assert "review_id,text,label,source,label_origin" in labels_path.read_text(encoding="utf-8")
    assert tasks[0]["data"]["source_name"] == "отзывы о банках"
    assert "predictions" not in tasks[0]
