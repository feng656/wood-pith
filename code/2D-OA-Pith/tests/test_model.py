import torch

from oapith.models import OAPithNet


def test_model_output_contract() -> None:
    model = OAPithNet(base_channels=8, mixture_components=3)
    image = torch.randn(2, 3, 128, 128)
    output = model(image, torch.ones(2, 1, 128, 128))
    assert output["ring_logits"].shape == (2, 1, 128, 128)
    assert output["distance"].shape == (2, 1, 128, 128)
    assert output["log_variance"].shape == (2, 1, 128, 128)
    assert output["orientation"].shape == (2, 2, 128, 128)
    assert output["embedding"].shape == (2, 8, 128, 128)
    assert output["defect_logits"].shape == (2, 4, 128, 128)
    assert output["near_mean"].shape == (2, 3, 2)
    assert output["near_cholesky"].shape == (2, 3, 2, 2)
    assert output["far_direction"].shape == (2, 3, 2)
    assert output["state_logits"].shape == (2, 4)


def test_variance_objective_does_not_update_shared_trunk_by_default() -> None:
    model = OAPithNet(base_channels=4)
    output = model(torch.randn(1, 3, 64, 64))
    output["log_variance"].mean().backward()
    shared_modules = [model.stem, model.encoder, model.decoder, model.dense_head]
    trunk_gradients = [
        parameter.grad
        for module in shared_modules
        for parameter in module.parameters()
    ]
    variance_gradients = [parameter.grad for parameter in model.variance_head.parameters()]
    assert all(value is None for value in trunk_gradients)
    assert any(
        value is not None
        and bool(torch.isfinite(value).all())
        and bool((value != 0).any())
        for value in variance_gradients
    )
