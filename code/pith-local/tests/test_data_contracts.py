from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from racpith.config import load_config, resolve_runtime_paths
from racpith.data.crop import (
    CropBox,
    build_crop_annotation,
    crop_pith_coordinates,
    overlapping_crop_boxes,
)
from racpith.data.split import make_grouped_split
from racpith.data.urudendro4 import (
    RingPolygon,
    UruDendro4Sample,
    load_labelme_rings,
    parse_sample_code,
)
from racpith.provenance import read_json_object, read_jsonl


def test_strict_json_rejects_decoder_overflow_and_non_object_root(tmp_path) -> None:
    overflow = tmp_path / "overflow.json"
    overflow.write_text('{"value": 1e999}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON number"):
        read_json_object(overflow)

    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="root is not an object"):
        read_json_object(non_object)


def test_strict_jsonl_reports_line_for_nonfinite_value(tmp_path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text('{"ok": 1}\n{"bad": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: invalid strict JSON"):
        read_jsonl(source)


def test_tree_id_keeps_treatment_and_block() -> None:
    code = parse_sample_code("T2_B3_N17_ADAP.jpg")
    assert code.section_id == "T2_B3_N17_ADAP"
    assert code.tree_id == "T2_B3_N17"


def test_runtime_paths_are_resolved_from_the_main_config() -> None:
    project_root = Path(__file__).resolve().parents[1]
    frozen = load_config(project_root / "configs" / "racpith_v1.json")
    paths = resolve_runtime_paths(frozen, project_root=project_root)
    assert paths.dataset_root == Path(
        "/root/2026/dataset/UruDendro4/UruDendro4"
    )
    assert paths.output_root == (project_root / "outputs").resolve()
    assert paths.prepared_root == (project_root / "outputs" / "prepared").resolve()
    assert paths.development_root == (
        project_root / "outputs" / "development"
    ).resolve()


def _labelme_payload(
    *,
    width: object,
    height: object,
    image_path: object = "T0_B1_N27_A.jpg",
) -> dict[str, object]:
    return {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "ring",
                "points": [[10, 10], [70, 10], [70, 50], [10, 50]],
                "shape_type": "polygon",
                "flags": {},
            }
        ],
        "imagePath": image_path,
        "imageData": None,
        "imageWidth": width,
        "imageHeight": height,
    }


def test_labelme_blank_dimensions_use_decoded_image_size(tmp_path) -> None:
    annotation = tmp_path / "T0_B1_N27_A.json"
    annotation.write_text(
        json.dumps(_labelme_payload(width="", height="", image_path="")),
        encoding="utf-8",
    )
    rings = load_labelme_rings(
        annotation,
        expected_section_id="T0_B1_N27_A",
        expected_image_size_px=(80, 60),
    )
    assert len(rings) == 1


def test_labelme_nonblank_image_path_must_match_section(tmp_path) -> None:
    annotation = tmp_path / "T0_B1_N27_A.json"
    annotation.write_text(
        json.dumps(
            _labelme_payload(
                width=80,
                height=60,
                image_path="another_section.jpg",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="imagePath does not match"):
        load_labelme_rings(
            annotation,
            expected_section_id="T0_B1_N27_A",
            expected_image_size_px=(80, 60),
        )


def test_labelme_nonblank_dimensions_must_match_image(tmp_path) -> None:
    annotation = tmp_path / "T0_B1_N27_A.json"
    annotation.write_text(
        json.dumps(_labelme_payload(width="81", height=60)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="declared dimensions"):
        load_labelme_rings(
            annotation,
            expected_section_id="T0_B1_N27_A",
            expected_image_size_px=(80, 60),
        )


def test_crop_annotation_is_gt_free_and_does_not_create_rectangle_edges(tmp_path) -> None:
    points = np.asarray([[100, 20], [180, 100], [100, 180], [20, 100]], dtype=float)
    ring = RingPolygon(
        ring_id="ring_000",
        ring_order=0,
        source_index=0,
        source_label="ring",
        points_xy=points,
        area_px2=12800.0,
        centroid_xy=(100.0, 100.0),
        bbox_xyxy=(20.0, 20.0, 180.0, 180.0),
    )
    sample = UruDendro4Sample(
        code=parse_sample_code("T0_B1_N1_A"),
        image_path=tmp_path / "T0_B1_N1_A.jpg",
        annotation_path=tmp_path / "T0_B1_N1_A.json",
        image_size_px=(200, 200),
        pith_source_px=(100.0, 100.0),
        rings=(ring,),
    )
    annotation = build_crop_annotation(
        sample,
        CropBox(0, 0, 100, 100),
        crop_id="fixture_crop",
        split="train",
        minimum_fragment_length_px=1.0,
    )
    assert not {"pith_source_px", "pith_crop_px", "pith_inside_crop"} & set(annotation)
    assert annotation["metadata"]["artificial_crop_boundary_edges_added"] is False
    for ring_payload in annotation["rings"]:
        for arc in ring_payload["arcs"]:
            xy = np.asarray(arc["points_crop_px"], dtype=float)
            assert np.all((xy[:, 0] >= 0) & (xy[:, 0] < 100))
            assert np.all((xy[:, 1] >= 0) & (xy[:, 1] < 100))
            assert len(arc["source_s_px"]) == len(arc["points_crop_px"])


def test_crop_grid_has_overlap() -> None:
    boxes = overlapping_crop_boxes(
        (200, 200), [0.5], stride_fraction_of_crop=0.35
    )
    first = boxes[0][2]
    second = boxes[1][2]
    assert second.x0 < first.x1


def test_crop_pith_coordinates_use_half_open_crop_bounds() -> None:
    box = CropBox(10, 20, 110, 220)
    assert crop_pith_coordinates((10.0, 20.0), box) == ((0.0, 0.0), True)
    assert crop_pith_coordinates((110.0, 20.0), box) == ((100.0, 0.0), False)


def test_grouped_split_never_separates_sections_of_one_tree() -> None:
    rows = []
    for tree_number in range(1, 5):
        for height in ("A", "B"):
            code = parse_sample_code(f"T0_B1_N{tree_number}_{height}")
            rows.append(code.as_dict())
    plan = make_grouped_split(
        rows,
        {"train": 0.5, "sealed_test": 0.5},
        random_seed=7,
        search_trials=10,
    )
    sections = plan.section_manifest_rows(rows)
    by_tree: dict[str, set[str]] = {}
    for row in sections:
        by_tree.setdefault(row["tree_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_tree.values())
    assert plan.audit["leakage_free"] is True
