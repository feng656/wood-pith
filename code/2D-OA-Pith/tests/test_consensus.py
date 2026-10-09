import torch

from oapith.data.consensus import consensus_curve
from oapith.data.manifest import RingCurve


def test_consensus_estimates_centerline_and_nonzero_normal_uncertainty() -> None:
    x = torch.linspace(0, 20, 21)
    curves = []
    for annotator, offset in (("a", -1.0), ("b", 0.0), ("c", 1.0)):
        points = torch.stack([x, torch.full_like(x, offset)], -1).tolist()
        curves.append(
            RingCurve(
                ring_id=4,
                arc_id="r4-a",
                annotator_id=annotator,
                points_px=points,
                sigma_px=0.2,
            )
        )
    result = consensus_curve(curves, samples=41, minimum_sigma_px=0.1)
    points = torch.tensor(result.points_px)
    sigma = torch.tensor(result.sigma_px)
    torch.testing.assert_close(points[:, 1], torch.zeros(41), atol=1e-10, rtol=0)
    assert bool((sigma > 0.5).all())


def test_two_rater_consensus_uses_midpoint_not_lower_order_statistic() -> None:
    x = torch.linspace(0, 10, 11)
    curves = [
        RingCurve(
            ring_id=2,
            arc_id="r2",
            annotator_id=name,
            points_px=torch.stack([x, torch.full_like(x, offset)], -1).tolist(),
            sigma_px=0.1,
        )
        for name, offset in (("left", -1.0), ("right", 1.0))
    ]
    result = consensus_curve(curves, samples=21)
    torch.testing.assert_close(
        torch.tensor(result.points_px)[:, 1], torch.zeros(21), atol=1e-10, rtol=0
    )
