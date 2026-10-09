from oapith.data.manifest import RingCurve, SampleRecord, dump_manifest, load_manifest


def test_manifest_metadata_roundtrip_does_not_nest(tmp_path) -> None:
    record = SampleRecord(
        sample_id="s",
        image="image.png",
        group_id="tree-1",
        curves=[
            RingCurve(
                ring_id=1,
                arc_id="a",
                annotator_id="expert",
                points_px=[[0.0, 0.0], [1.0, 0.1], [2.0, 0.0]],
                metadata={"closed": False},
            )
        ],
        metadata={"tree_id": "tree-1"},
    )
    path = tmp_path / "manifest.jsonl"
    dump_manifest([record], path)
    loaded = load_manifest(path)[0]
    assert loaded.metadata == {"tree_id": "tree-1"}
    assert loaded.curves[0].metadata == {"closed": False}
