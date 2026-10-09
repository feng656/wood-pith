from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import torch
from torch import nn

from oapith.geometry.curves import estimate_tangents, resample_polyline
from oapith.geometry.disturbance import LatentMorphologyField
from oapith.types import ArcObservation, PithState


class RefinementMode(str, Enum):
    NEAR = "near"
    FAR = "far"
    INFINITY = "infinity"


@dataclass
class RefinerConfig:
    sample_spacing: float = 0.015
    minimum_sigma: float = 0.002
    student_df: float = 4.0
    outlier_scale: float = 8.0
    tangent_scale: float = 0.15
    switch_radius: float = 2.0
    near_max_radius: float = 4.0
    far_allow_affine: bool = False
    maximum_harmonic: int = 5
    nesting_margin: float = 0.002
    warmup_steps: int = 35
    stage_steps: int = 50
    em_rounds: int = 3
    max_modes: int = 6
    mode_loss_window: float = 12.0
    deduplication_distance: float = 0.08
    deduplication_angle: float = 0.02
    deduplication_inverse_distance: float = 0.02
    stationarity_tolerance: float = 1e-4
    disturbance_source_options: tuple[int, ...] = (0, 1)
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.sample_spacing <= 0 or self.minimum_sigma <= 0:
            raise ValueError("sampling and uncertainty scales must be positive")
        if self.student_df <= 0 or self.outlier_scale <= 0 or self.tangent_scale <= 0:
            raise ValueError("likelihood scales must be positive")
        if self.switch_radius <= 0 or self.near_max_radius < self.switch_radius:
            raise ValueError("near_max_radius must overlap and exceed switch_radius")
        if self.maximum_harmonic < 0 or self.nesting_margin <= 0:
            raise ValueError("invalid growth parameterization")
        if not isinstance(self.maximum_harmonic, int):
            raise ValueError("maximum_harmonic must be an integer")
        if min(self.warmup_steps, self.stage_steps, self.em_rounds) < 1:
            raise ValueError("optimization step and EM counts must be positive")
        if self.mode_loss_window < 0:
            raise ValueError("mode_loss_window must be nonnegative")
        if self.max_modes < 1 or not self.disturbance_source_options:
            raise ValueError("at least one mode and disturbance option are required")
        if min(
            self.deduplication_distance,
            self.deduplication_angle,
            self.deduplication_inverse_distance,
            self.stationarity_tolerance,
        ) <= 0:
            raise ValueError("deduplication scales must be positive")
        if any(not isinstance(value, int) or value < 0 for value in self.disturbance_source_options):
            raise ValueError("disturbance source counts must be nonnegative")


@dataclass
class PackedArcs:
    points: torch.Tensor
    tangents: torch.Tensor
    sigma: torch.Tensor
    arc_index: torch.Tensor
    ring_index: torch.Tensor
    arc_ids: list[str]
    ring_ids: list[int]
    prior_probability: torch.Tensor
    counts: torch.Tensor

    @property
    def number_of_arcs(self) -> int:
        return len(self.arc_ids)


@dataclass
class RefinementResult:
    mode: RefinementMode
    state: PithState
    center: torch.Tensor | None
    direction: torch.Tensor | None
    inverse_distance: float | None
    loss: float
    reliabilities: dict[str, float]
    model_state: dict[str, torch.Tensor]
    evidence_score: float
    diagnostics: dict[str, float | int | bool | str] = field(default_factory=dict)


def pack_arcs(arcs: list[ArcObservation], config: RefinerConfig) -> PackedArcs:
    identifiers = [arc.arc_id for arc in arcs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("arc_id values must be unique")
    points_list, tangent_list, sigma_list, arc_indices, ring_values = [], [], [], [], []
    arc_ids, ring_ids = [], sorted({arc.ring_id for arc in arcs})
    ring_lookup = {value: index for index, value in enumerate(ring_ids)}
    priors, counts = [], []
    for arc_number, arc in enumerate(arcs):
        arc.validate()
        points = resample_polyline(arc.points.to(dtype=config.dtype), config.sample_spacing)
        tangents = estimate_tangents(points)
        if arc.sigma.ndim == 0:
            sigma = arc.sigma.to(dtype=config.dtype).expand(points.shape[0])
        else:
            # Re-sample uncertainty by true source arclength, not vertex index;
            # annotation polylines are rarely uniformly sampled.
            source_points = arc.points.to(device=points.device, dtype=config.dtype)
            source = torch.cat(
                [
                    source_points.new_zeros(1),
                    torch.cumsum(
                        torch.linalg.vector_norm(
                            source_points[1:] - source_points[:-1], dim=-1
                        ),
                        dim=0,
                    ),
                ]
            )
            source = source / source[-1].clamp_min(1e-8)
            target = torch.linspace(0, 1, points.shape[0], device=points.device, dtype=config.dtype)
            right = torch.searchsorted(source, target, right=True).clamp(1, source.shape[0] - 1)
            left = right - 1
            alpha = (target - source[left]) / (source[right] - source[left]).clamp_min(1e-8)
            sigma = arc.sigma.to(dtype=config.dtype)[left] * (1 - alpha) + arc.sigma.to(
                dtype=config.dtype
            )[right] * alpha
        sigma = sigma.clamp_min(1e-8)
        points_list.append(points)
        tangent_list.append(tangents)
        sigma_list.append(sigma)
        arc_indices.append(torch.full((points.shape[0],), arc_number, device=points.device))
        ring_values.append(
            torch.full(
                (points.shape[0],), ring_lookup[arc.ring_id], device=points.device, dtype=torch.long
            )
        )
        arc_ids.append(arc.arc_id)
        priors.append(arc.prior_reliability)
        counts.append(points.shape[0])
    if not points_list:
        raise ValueError("at least one arc is required")
    device = points_list[0].device
    return PackedArcs(
        points=torch.cat(points_list),
        tangents=torch.cat(tangent_list),
        sigma=torch.cat(sigma_list),
        arc_index=torch.cat(arc_indices).long(),
        ring_index=torch.cat(ring_values),
        arc_ids=arc_ids,
        ring_ids=ring_ids,
        prior_probability=torch.tensor(priors, device=device, dtype=config.dtype),
        counts=torch.tensor(counts, device=device, dtype=config.dtype),
    )


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value > 30.0, value, torch.log(torch.expm1(value).clamp_min(1e-12)))


def _affine_from_parameters(parameters: torch.Tensor) -> torch.Tensor:
    """Symmetric positive definite affine with determinant exactly one."""
    a, b = parameters.unbind()
    generator = torch.stack([torch.stack([a, b]), torch.stack([b, -a])])
    return torch.matrix_exp(generator)


def _fourier(theta: torch.Tensor, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
    if stop < start:
        empty = theta.new_zeros(theta.shape[0], 0)
        return empty, empty
    orders = torch.arange(start, stop + 1, device=theta.device, dtype=theta.dtype)
    angles = theta[:, None] * orders[None, :]
    basis = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    derivative = torch.cat(
        [-orders[None, :] * torch.sin(angles), orders[None, :] * torch.cos(angles)], dim=-1
    )
    return basis, derivative


def _helmert_contrast(number: int, like: torch.Tensor) -> torch.Tensor:
    """Orthonormal ring contrasts spanning the exact zero-mean subspace."""
    if number <= 1:
        return like.new_zeros(number, 0)
    result = like.new_zeros(number, number - 1)
    for column in range(number - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        result[: column + 1, column] = 1.0 / denominator
        result[column + 1, column] = -(column + 1) / denominator
    return result


class _ProbabilisticLayerModel(nn.Module):
    def __init__(self, data: PackedArcs, config: RefinerConfig) -> None:
        super().__init__()
        self.number_of_rings = len(data.ring_ids)
        self.number_of_arcs = data.number_of_arcs
        self.config = config

    def point_residuals(
        self, data: PackedArcs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def prior_loss(self) -> torch.Tensor:
        raise NotImplementedError

    def pith_chart(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        raise NotImplementedError

    def _log_likelihoods(self, data: PackedArcs) -> tuple[torch.Tensor, torch.Tensor]:
        radial, tangent, radial_log_scale = self.point_residuals(data)
        student = torch.distributions.StudentT(df=self.config.student_df)
        inlier_point = (
            student.log_prob(radial)
            - radial_log_scale
            + student.log_prob(tangent)
            - math.log(self.config.tangent_scale)
        )
        # The contamination component is uniform over a broad standardized residual
        # window and therefore independent of the candidate geometry. Making it a
        # broad Gaussian of the residual would let "outliers" pull the center too.
        # Same base measure as the inlier: physical radial residual F and
        # sin(delta_theta).  The latter has support width two.
        outlier_point = radial.new_full(
            radial.shape, -math.log(2.0 * self.config.outlier_scale) - math.log(2.0)
        )
        inlier_arc = radial.new_zeros(self.number_of_arcs)
        outlier_arc = radial.new_zeros(self.number_of_arcs)
        inlier_arc.scatter_add_(0, data.arc_index, inlier_point)
        outlier_arc.scatter_add_(0, data.arc_index, outlier_point)
        # Limit how strongly dense samples classify a whole arc while retaining the
        # genuine information advantage of a longer independent arc.
        cap = torch.minimum(data.counts, data.counts.new_full(data.counts.shape, 24.0))
        factor = cap / data.counts.clamp_min(1.0)
        return inlier_arc * factor, outlier_arc * factor

    def posterior_reliability(self, data: PackedArcs) -> torch.Tensor:
        inlier, outlier = self._log_likelihoods(data)
        prior = data.prior_probability.clamp(1e-5, 1 - 1e-5)
        logit = torch.logit(prior) + inlier - outlier
        return torch.sigmoid(logit).detach()

    def weighted_objective(
        self, data: PackedArcs, reliability: torch.Tensor, include_prior: bool = True
    ) -> torch.Tensor:
        inlier_arc, outlier_arc = self._log_likelihoods(data)
        # Exact expected complete-data NLL for the same capped arc mixture used in
        # the E step. The outlier term is constant in geometry but retained so the
        # optimization and reported objective have one explicit probabilistic scale.
        data_loss = -(
            reliability * inlier_arc + (1.0 - reliability) * outlier_arc
        ).sum()
        return data_loss + (self.prior_loss() if include_prior else 0.0)

    def marginal_objective(self, data: PackedArcs, include_prior: bool = True) -> torch.Tensor:
        inlier, outlier = self._log_likelihoods(data)
        prior = data.prior_probability.clamp(1e-5, 1 - 1e-5)
        terms = torch.stack([torch.log(prior) + inlier, torch.log1p(-prior) + outlier], -1)
        data_loss = -torch.logsumexp(terms, dim=-1).sum()
        return data_loss + (self.prior_loss() if include_prior else 0.0)

    def forward(
        self,
        data: PackedArcs,
        reliability: torch.Tensor | None = None,
        marginal: bool = False,
        include_prior: bool = True,
    ) -> torch.Tensor:
        if marginal:
            return self.marginal_objective(data, include_prior)
        if reliability is None:
            raise ValueError("reliability is required for the weighted objective")
        return self.weighted_objective(data, reliability, include_prior)


class NearGrowthModel(_ProbabilisticLayerModel):
    def __init__(
        self,
        data: PackedArcs,
        center: torch.Tensor,
        config: RefinerConfig,
        number_disturbance_sources: int = 0,
    ) -> None:
        super().__init__(data, config)
        self.target = nn.Parameter(center.to(dtype=config.dtype).clone())
        self.affine_parameters = nn.Parameter(center.new_zeros(2, dtype=config.dtype))
        initial_radii = []
        for ring in range(self.number_of_rings):
            distance = torch.linalg.vector_norm(data.points[data.ring_index == ring] - self.target, dim=-1)
            initial_radii.append(distance.median().clamp_min(0.02))
        initial_radii_tensor = torch.stack(initial_radii)
        ring_order = torch.argsort(initial_radii_tensor.detach())
        self.register_buffer("ring_order", ring_order)
        ring_position = torch.empty_like(ring_order)
        ring_position[ring_order] = torch.arange(self.number_of_rings, device=ring_order.device)
        self.register_buffer("ring_position", ring_position)
        ordered_radii = initial_radii_tensor[ring_order]
        self.inner_base_raw = nn.Parameter(_inverse_softplus(ordered_radii[0] - 0.005))
        gap_count = max(0, self.number_of_rings - 1)
        initial_gaps = (
            (ordered_radii[1:] - ordered_radii[:-1] - config.nesting_margin).clamp_min(1e-4)
            if gap_count
            else ordered_radii.new_zeros(0)
        )
        self.gap_base_raw = nn.Parameter(_inverse_softplus(initial_gaps))
        inner_width = max(0, 2 * (config.maximum_harmonic - 2))
        gap_width = 2 * config.maximum_harmonic if gap_count else 0
        self.inner_high_coefficients = nn.Parameter(
            center.new_zeros(inner_width, dtype=config.dtype)
        )
        self.gap_shared_coefficients = nn.Parameter(
            center.new_zeros(gap_width, dtype=config.dtype)
        )
        self.register_buffer(
            "gap_contrast", _helmert_contrast(gap_count, center.to(config.dtype))
        )
        self.gap_deviation_coefficients = nn.Parameter(
            center.new_zeros(max(0, gap_count - 1), gap_width, dtype=config.dtype)
        )
        self.disturbance = LatentMorphologyField(
            number_disturbance_sources,
            self.number_of_rings,
            data.points,
            data.ring_index,
            self.ring_order,
        )

    def _radius_and_derivative(
        self, theta: torch.Tensor, ring_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Positive inner radius plus strictly positive ordered growth increments."""
        inner_basis, inner_derivative_basis = _fourier(
            theta, 3, self.config.maximum_harmonic
        )
        inner_input = self.inner_base_raw.expand(theta.shape[0])
        inner_input_derivative = theta.new_zeros(theta.shape[0])
        if inner_basis.shape[1]:
            inner_input = inner_input + inner_basis @ self.inner_high_coefficients
            inner_input_derivative = (
                inner_derivative_basis @ self.inner_high_coefficients
            )
        inner_radius = 0.005 + torch.nn.functional.softplus(inner_input)
        inner_derivative = torch.sigmoid(inner_input) * inner_input_derivative
        gap_count = self.number_of_rings - 1
        if gap_count == 0:
            return inner_radius, inner_derivative

        gap_basis, gap_derivative_basis = _fourier(
            theta, 1, self.config.maximum_harmonic
        )
        shared = gap_basis @ self.gap_shared_coefficients
        shared_derivative = gap_derivative_basis @ self.gap_shared_coefficients
        deviations = self.gap_contrast @ self.gap_deviation_coefficients
        gap_input = (
            self.gap_base_raw[:, None]
            + shared[None, :]
            + deviations @ gap_basis.T
        )
        gap_input_derivative = (
            shared_derivative[None, :] + deviations @ gap_derivative_basis.T
        )
        gaps = self.config.nesting_margin + torch.nn.functional.softplus(gap_input)
        gap_derivatives = torch.sigmoid(gap_input) * gap_input_derivative
        zero = gaps.new_zeros(1, theta.shape[0])
        cumulative = torch.cat([zero, torch.cumsum(gaps, dim=0)], dim=0)
        cumulative_derivative = torch.cat(
            [zero, torch.cumsum(gap_derivatives, dim=0)], dim=0
        )
        position = self.ring_position[ring_index]
        sample = torch.arange(theta.shape[0], device=theta.device)
        return (
            inner_radius + cumulative[position, sample],
            inner_derivative + cumulative_derivative[position, sample],
        )

    def _curve_quantities(
        self, data: PackedArcs
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        affine = _affine_from_parameters(self.affine_parameters)
        inverse = torch.linalg.inv(affine)
        disturbance, jacobian = self.disturbance.field_and_jacobian(
            data.points, data.ring_index
        )
        corrected_points = data.points - disturbance
        inverse_warp_jacobian = (
            torch.eye(2, device=data.points.device, dtype=data.points.dtype)[None] - jacobian
        )
        corrected_tangents = torch.nn.functional.normalize(
            torch.einsum("nij,nj->ni", inverse_warp_jacobian, data.tangents), dim=-1
        )
        relative = corrected_points - self.target
        y = relative @ inverse.T
        radius = torch.linalg.vector_norm(y, dim=-1).clamp_min(1e-8)
        theta = torch.atan2(y[:, 1], y[:, 0])
        predicted_radius, derivative_radius = self._radius_and_derivative(
            theta, data.ring_index
        )
        unit = torch.stack([torch.cos(theta), torch.sin(theta)], -1)
        angular = torch.stack([-torch.sin(theta), torch.cos(theta)], -1)
        predicted_tangent = (derivative_radius[:, None] * unit + predicted_radius[:, None] * angular)
        predicted_tangent = torch.nn.functional.normalize(predicted_tangent @ affine.T, dim=-1)
        return (
            radius,
            predicted_radius,
            derivative_radius,
            predicted_tangent,
            inverse,
            inverse_warp_jacobian,
            corrected_tangents,
            y,
        )

    def point_residuals(
        self, data: PackedArcs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            radius,
            predicted_radius,
            derivative_radius,
            predicted_tangent,
            inverse,
            inverse_warp_jacobian,
            corrected_tangents,
            y,
        ) = self._curve_quantities(data)
        unit = y / radius[:, None]
        angular = torch.stack([-unit[:, 1], unit[:, 0]], -1)
        gradient_y = unit - (derivative_radius / radius)[:, None] * angular
        gradient_x = torch.einsum(
            "ni,nij->nj", gradient_y @ inverse, inverse_warp_jacobian
        )
        observed_normal = torch.stack([-data.tangents[:, 1], data.tangents[:, 0]], -1)
        normal_gradient = (gradient_x * observed_normal).sum(-1).abs().clamp_min(1e-4)
        distance_sigma = torch.sqrt(data.sigma.square() + self.config.minimum_sigma**2)
        signed_normal_distance = (radius - predicted_radius) / normal_gradient
        radial = signed_normal_distance / distance_sigma
        tangent_cross = (
            corrected_tangents[:, 0] * predicted_tangent[:, 1]
            - corrected_tangents[:, 1] * predicted_tangent[:, 0]
        )
        tangent = tangent_cross / self.config.tangent_scale
        return radial, tangent, torch.log(distance_sigma)

    def prior_loss(self) -> torch.Tensor:
        harmonic = (
            1e-2 * self.inner_high_coefficients.square().sum()
            + 1e-2 * self.gap_shared_coefficients.square().sum()
            + 5e-2 * self.gap_deviation_coefficients.square().sum()
        )
        affine = 2e-2 * self.affine_parameters.square().sum()
        gap_smoothness = (
            1e-3 * (self.gap_base_raw[1:] - self.gap_base_raw[:-1]).square().mean()
            if self.gap_base_raw.numel() > 1
            else self.inner_base_raw.new_zeros(())
        )
        return harmonic + affine + gap_smoothness + self.disturbance.regularization()

    def pith_chart(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direction = torch.nn.functional.normalize(self.target, dim=-1, eps=1e-8)
        inverse_distance = 1.0 / torch.linalg.vector_norm(self.target).clamp_min(1e-8)
        return self.target, direction, inverse_distance


class FarLayerModel(_ProbabilisticLayerModel):
    """Numerically continuous circular/elliptic layer model at inverse distance zero."""

    def __init__(
        self,
        data: PackedArcs,
        direction: torch.Tensor,
        inverse_distance: float,
        config: RefinerConfig,
        *,
        fixed_infinity: bool = False,
        number_disturbance_sources: int = 0,
    ) -> None:
        super().__init__(data, config)
        angle = torch.atan2(direction[1], direction[0]).to(dtype=config.dtype)
        maximum_inverse = 1.0 / config.switch_radius
        if fixed_infinity:
            self.target = nn.Parameter(angle.reshape(1))
        else:
            self.target = nn.Parameter(
                torch.stack(
                    [angle, angle.new_tensor(min(max(inverse_distance, 0.0), maximum_inverse))]
                ).to(dtype=config.dtype)
            )
        self.fixed_infinity = fixed_infinity
        self.allow_affine = bool(config.far_allow_affine)
        self.affine_parameters = nn.Parameter(direction.new_zeros(2, dtype=config.dtype))
        self.affine_parameters.requires_grad_(self.allow_affine)
        n = self.direction()
        rho = self.inverse_distance()
        transformed = data.points
        base_factor, _ = self._base_factor_and_gradient(transformed, n, rho)
        offsets = []
        for ring in range(self.number_of_rings):
            offsets.append(base_factor[data.ring_index == ring].median())
        initial_offsets = torch.stack(offsets)
        ring_order = torch.argsort(initial_offsets.detach())
        self.register_buffer("ring_order", ring_order)
        ring_position = torch.empty_like(ring_order)
        ring_position[ring_order] = torch.arange(self.number_of_rings, device=ring_order.device)
        self.register_buffer("ring_position", ring_position)
        ordered_offsets = initial_offsets[ring_order]
        self.inner_offset = nn.Parameter(ordered_offsets[0].clone())
        initial_gaps = (
            (ordered_offsets[1:] - ordered_offsets[:-1] - config.nesting_margin).clamp_min(1e-4)
            if self.number_of_rings > 1
            else ordered_offsets.new_zeros(0)
        )
        self.offset_gap_raw = nn.Parameter(_inverse_softplus(initial_gaps))
        # Quadratic tangential deformation is intentionally excluded: at rho=0 its
        # score is collinear with inverse distance/curvature. Cubic and quartic terms
        # retain broad low-frequency morphology without destroying the rho gauge.
        self.higher_deformation = nn.Parameter(direction.new_zeros(2, dtype=config.dtype))
        self.disturbance = LatentMorphologyField(
            number_disturbance_sources,
            self.number_of_rings,
            data.points,
            data.ring_index,
            self.ring_order,
        )

    def _ring_offsets(self) -> torch.Tensor:
        if self.number_of_rings == 1:
            ordered = self.inner_offset.reshape(1)
        else:
            gaps = self.config.nesting_margin + torch.nn.functional.softplus(
                self.offset_gap_raw
            )
            ordered = self.inner_offset + torch.cat(
                [gaps.new_zeros(1), torch.cumsum(gaps, dim=0)]
            )
        return ordered[self.ring_position]

    def direction(self) -> torch.Tensor:
        angle = self.target[0]
        return torch.stack([torch.cos(angle), torch.sin(angle)])

    def inverse_distance(self) -> torch.Tensor:
        if self.fixed_infinity:
            return self.target.new_zeros(())
        # Direct rho keeps the residual/Hessian regular at rho=0. A smooth residual
        # plus a one-sided prior is preferable to a sigmoid whose derivative vanishes
        # exactly where distance observability is weakest.
        return self.target[1]

    @staticmethod
    def _base_factor_and_gradient(
        y: torch.Tensor, direction: torch.Tensor, inverse_distance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rationalized circle factor, finite and differentiable at rho=0."""
        z = inverse_distance * y - direction
        norm_z = torch.linalg.vector_norm(z, dim=-1).clamp_min(1e-10)
        numerator = inverse_distance * y.square().sum(-1) - 2.0 * (y * direction).sum(-1)
        denominator = norm_z + 1.0
        factor = numerator / denominator
        gradient_numerator = 2.0 * inverse_distance * y - 2.0 * direction
        gradient_denominator = inverse_distance * z / norm_z[:, None]
        gradient = (
            gradient_numerator * denominator[:, None]
            - numerator[:, None] * gradient_denominator
        ) / denominator[:, None].square()
        return factor, gradient

    def point_residuals(
        self, data: PackedArcs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        affine = _affine_from_parameters(self.affine_parameters)
        inverse = torch.linalg.inv(affine)
        disturbance, jacobian = self.disturbance.field_and_jacobian(
            data.points, data.ring_index
        )
        corrected_points = data.points - disturbance
        inverse_warp_jacobian = (
            torch.eye(2, device=data.points.device, dtype=data.points.dtype)[None] - jacobian
        )
        corrected_tangents = torch.nn.functional.normalize(
            torch.einsum("nij,nj->ni", inverse_warp_jacobian, data.tangents), dim=-1
        )
        y = corrected_points @ inverse.T
        # The target chart is always expressed in the original normalized image.
        # Convert that pith ray into the shared-affine chart without allowing the
        # nuisance affine to redefine the reported pith.
        direction_x, rho_x = self.direction(), self.inverse_distance()
        direction_y_raw = direction_x @ inverse.T
        direction_scale = torch.linalg.vector_norm(direction_y_raw).clamp_min(1e-10)
        direction_y = direction_y_raw / direction_scale
        rho_y = rho_x / direction_scale
        factor, gradient_y = self._base_factor_and_gradient(y, direction_y, rho_y)
        tangent_axis = torch.stack([-direction_y[1], direction_y[0]])
        local_coordinate = y @ tangent_axis
        deformation = (
            self.higher_deformation[0] * local_coordinate.pow(3)
            + self.higher_deformation[1] * local_coordinate.pow(4)
        )
        predicted = self._ring_offsets()[data.ring_index] + deformation
        deformation_gradient = (
            3.0 * self.higher_deformation[0] * local_coordinate.square()
            + 4.0 * self.higher_deformation[1] * local_coordinate.pow(3)
        )[:, None] * tangent_axis
        normal_y = gradient_y - deformation_gradient
        gradient_x = torch.einsum(
            "ni,nij->nj", normal_y @ inverse, inverse_warp_jacobian
        )
        observed_normal = torch.stack([-data.tangents[:, 1], data.tangents[:, 0]], -1)
        normal_gradient = (gradient_x * observed_normal).sum(-1).abs().clamp_min(1e-4)
        distance_sigma = torch.sqrt(data.sigma.square() + self.config.minimum_sigma**2)
        signed_normal_distance = (factor - predicted) / normal_gradient
        radial = signed_normal_distance / distance_sigma
        normal_x = torch.nn.functional.normalize(normal_y @ inverse, dim=-1, eps=1e-8)
        predicted_tangent = torch.stack([-normal_x[:, 1], normal_x[:, 0]], -1)
        tangent_cross = (
            corrected_tangents[:, 0] * predicted_tangent[:, 1]
            - corrected_tangents[:, 1] * predicted_tangent[:, 0]
        )
        tangent = tangent_cross / self.config.tangent_scale
        return radial, tangent, torch.log(distance_sigma)

    def prior_loss(self) -> torch.Tensor:
        rho = self.inverse_distance()
        maximum_inverse = 1.0 / self.config.switch_radius
        range_penalty = (
            1e4 * torch.nn.functional.relu(-rho).square()
            + 1e4 * torch.nn.functional.relu(rho - maximum_inverse).square()
        )
        gap_smoothness = (
            1e-3 * (self.offset_gap_raw[1:] - self.offset_gap_raw[:-1]).square().mean()
            if self.offset_gap_raw.numel() > 1
            else self.inner_offset.new_zeros(())
        )
        return (
            2e-2 * self.affine_parameters.square().sum()
            + 1e-1 * self.higher_deformation.square().sum()
            + range_penalty
            + gap_smoothness
            + self.disturbance.regularization()
        )

    def pith_chart(self) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        direction_x, rho = self.direction(), self.inverse_distance().clamp_min(0.0)
        if float(rho.detach()) < 1e-8:
            return None, direction_x, rho
        center_x = direction_x / rho
        return center_x, direction_x, rho


class ProbabilisticGeometryRefiner:
    def __init__(self, config: RefinerConfig | None = None) -> None:
        self.config = config or RefinerConfig()

    @staticmethod
    def _set_stage(model: _ProbabilisticLayerModel, stage: int) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.target.requires_grad_(True)
        for name in ("inner_base_raw", "gap_base_raw"):
            if hasattr(model, name):
                getattr(model, name).requires_grad_(True)
        for name in ("inner_offset", "offset_gap_raw"):
            if hasattr(model, name):
                getattr(model, name).requires_grad_(True)
        if stage >= 1:
            if not isinstance(model, FarLayerModel) or model.allow_affine:
                model.affine_parameters.requires_grad_(True)
        if stage >= 2:
            for name in (
                "inner_high_coefficients",
                "gap_shared_coefficients",
                "gap_deviation_coefficients",
                "higher_deformation",
            ):
                if hasattr(model, name):
                    getattr(model, name).requires_grad_(True)
            for parameter in model.disturbance.parameters():
                parameter.requires_grad_(True)

    def _lbfgs(
        self,
        model: _ProbabilisticLayerModel,
        data: PackedArcs,
        reliability: torch.Tensor,
        steps: int,
        *,
        marginal: bool = False,
    ) -> None:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=0.8,
            max_iter=steps,
            max_eval=max(steps * 2, 20),
            tolerance_grad=1e-8,
            tolerance_change=1e-10,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                data,
                None if marginal else reliability,
                marginal=marginal,
                include_prior=True,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite geometry objective")
            loss.backward()
            return loss

        optimizer.step(closure)

    def _fit_one(
        self,
        data: PackedArcs,
        start: torch.Tensor,
        mode: RefinementMode,
        number_disturbance_sources: int,
    ) -> tuple[_ProbabilisticLayerModel, RefinementResult]:
        if mode is RefinementMode.NEAR:
            model: _ProbabilisticLayerModel = NearGrowthModel(
                data, start, self.config, number_disturbance_sources
            )
        else:
            distance = float(torch.linalg.vector_norm(start).clamp_min(self.config.switch_radius))
            direction = torch.nn.functional.normalize(start, dim=-1)
            model = FarLayerModel(
                data,
                direction,
                1.0 / distance,
                self.config,
                fixed_infinity=mode is RefinementMode.INFINITY,
                number_disturbance_sources=number_disturbance_sources,
            )
        reliability = data.prior_probability.detach().clone()
        self._set_stage(model, 0)
        self._lbfgs(model, data, reliability, self.config.warmup_steps)
        for stage in (1, 2):
            self._set_stage(model, stage)
            for _ in range(self.config.em_rounds):
                self._lbfgs(model, data, reliability, self.config.stage_steps)
                reliability = model.posterior_reliability(data)
        # Finish at the actual marginalized MAP used by the observed Hessian. EM is
        # the stable optimizer/cold-start mechanism, not a frozen-weight endpoint.
        self._set_stage(model, 2)
        self._lbfgs(
            model,
            data,
            reliability,
            max(12, self.config.stage_steps // 2),
            marginal=True,
        )
        reliability = model.posterior_reliability(data)
        if isinstance(model, FarLayerModel) and not model.fixed_infinity:
            rho = float(model.inverse_distance().detach())
            if not (0.0 < rho <= 1.0 / self.config.switch_radius):
                raise FloatingPointError("finite far mode left its valid inverse-distance chart")
        if isinstance(model, NearGrowthModel):
            radius = float(torch.linalg.vector_norm(model.target).detach())
            if radius > self.config.near_max_radius:
                raise FloatingPointError("near mode left its finite Cartesian chart")
        marginal = float(model(data, marginal=True, include_prior=True).detach())
        data_nll = float(model(data, marginal=True, include_prior=False).detach())
        center, direction, inverse_distance = model.pith_chart()
        state = {
            RefinementMode.NEAR: PithState.NEAR,
            RefinementMode.FAR: PithState.FAR,
            RefinementMode.INFINITY: PithState.INFINITY,
        }[mode]
        number_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        effective_sample_size = float(
            torch.minimum(data.counts, data.counts.new_full(data.counts.shape, 24.0)).sum()
        )
        # This is deliberately labelled a selection heuristic, not a Laplace
        # marginal likelihood: the likelihood is arc-capped, so raw resampled point
        # count would make model weights change merely with sample_spacing.
        selection_bic = 2.0 * data_nll + number_parameters * math.log(
            max(2.0, effective_sample_size)
        )
        result = RefinementResult(
            mode=mode,
            state=state,
            center=None if center is None else center.detach().clone(),
            direction=None if direction is None else direction.detach().clone(),
            inverse_distance=None if inverse_distance is None else float(inverse_distance.detach()),
            loss=marginal,
            reliabilities={
                arc_id: float(value)
                for arc_id, value in zip(data.arc_ids, reliability.detach().cpu())
            },
            model_state={key: value.detach().clone() for key, value in model.state_dict().items()},
            evidence_score=-0.5 * selection_bic,
            diagnostics={
                "number_points": data.points.shape[0],
                "number_arcs": data.number_of_arcs,
                "number_rings": len(data.ring_ids),
                "number_parameters": number_parameters,
                "number_disturbance_sources": number_disturbance_sources,
                "effective_sample_size": effective_sample_size,
                "selection_data_nll": data_nll,
                "selection_criterion": "effective-sample BIC heuristic",
            },
        )
        return model, result

    def refine(
        self,
        arcs: list[ArcObservation],
        candidates: Iterable[torch.Tensor],
        *,
        modes: tuple[RefinementMode, ...] = (
            RefinementMode.NEAR,
            RefinementMode.FAR,
            RefinementMode.INFINITY,
        ),
        state_probabilities: torch.Tensor | None = None,
    ) -> tuple[list[RefinementResult], list[_ProbabilisticLayerModel], PackedArcs]:
        if not modes or len(set(modes)) != len(modes):
            raise ValueError("modes must be a non-empty tuple of unique charts")
        data = pack_arcs(arcs, self.config)
        candidates = [candidate.to(data.points) for candidate in candidates]
        if not candidates:
            raise ValueError("at least one initialization candidate is required")
        fitted: list[tuple[_ProbabilisticLayerModel, RefinementResult]] = []
        for mode in modes:
            starts = candidates
            if mode is RefinementMode.NEAR:
                starts = [
                    value
                    for value in candidates
                    if torch.linalg.vector_norm(value) <= self.config.near_max_radius
                ]
            elif mode in (RefinementMode.FAR, RefinementMode.INFINITY):
                starts = [value for value in candidates if torch.linalg.vector_norm(value) >= 1.0]
            if len(starts) > 16:
                indices = torch.linspace(0, len(starts) - 1, 16).round().long().tolist()
                starts = [starts[index] for index in indices]
            for number_sources in self.config.disturbance_source_options:
                for start in starts:
                    try:
                        item = self._fit_one(data, start, mode, number_sources)
                        if state_probabilities is not None:
                            probability = state_probabilities[int(item[1].state)].clamp_min(1e-8)
                            item[1].evidence_score += float(torch.log(probability))
                            item[1].diagnostics["state_gating_probability"] = float(probability)
                        fitted.append(item)
                    except (RuntimeError, FloatingPointError):
                        continue
        if not fitted:
            raise RuntimeError("all geometry optimization starts failed")
        fitted.sort(key=lambda item: item[1].evidence_score, reverse=True)
        kept: list[tuple[_ProbabilisticLayerModel, RefinementResult]] = []
        best_evidence = fitted[0][1].evidence_score

        def eligible(item: tuple[_ProbabilisticLayerModel, RefinementResult]) -> bool:
            return item[1].evidence_score >= best_evidence - self.config.mode_loss_window

        def duplicate_of_kept(result: RefinementResult) -> bool:
            for _, previous in kept:
                if result.mode is not previous.mode:
                    continue
                if result.mode is RefinementMode.NEAR:
                    assert result.center is not None and previous.center is not None
                    if bool(
                        torch.linalg.vector_norm(result.center - previous.center)
                        < self.config.deduplication_distance
                    ):
                        return True
                elif result.mode is RefinementMode.FAR:
                    assert result.direction is not None and previous.direction is not None
                    angle_a = torch.atan2(result.direction[1], result.direction[0])
                    angle_b = torch.atan2(previous.direction[1], previous.direction[0])
                    angle_difference = torch.atan2(
                        torch.sin(angle_a - angle_b), torch.cos(angle_a - angle_b)
                    ).abs()
                    rho_difference = abs(
                        (result.inverse_distance or 0.0)
                        - (previous.inverse_distance or 0.0)
                    )
                    if bool(angle_difference < self.config.deduplication_angle) and (
                        rho_difference < self.config.deduplication_inverse_distance
                    ):
                        return True
                else:
                    assert result.direction is not None and previous.direction is not None
                    if bool((result.direction * previous.direction).sum() > 0.9998):
                        return True
            return False

        # Preserve at least one eligible representative of every fitted chart when
        # the budget permits.  If max_modes is smaller than the number of charts,
        # select chart representatives by evidence instead of fixed enum order.
        chart_representatives = []
        for mode in modes:
            candidate = next(
                (item for item in fitted if item[1].mode is mode and eligible(item)), None
            )
            if candidate is not None:
                chart_representatives.append(candidate)
        chart_representatives.sort(
            key=lambda item: item[1].evidence_score, reverse=True
        )
        kept.extend(chart_representatives[: self.config.max_modes])

        for model, result in fitted:
            if len(kept) >= self.config.max_modes:
                break
            item = (model, result)
            if not eligible(item):
                continue
            if not duplicate_of_kept(result):
                kept.append(item)
            if len(kept) >= self.config.max_modes:
                break
        return [item[1] for item in kept], [item[0] for item in kept], data

    def restore_model(
        self, result: RefinementResult, data: PackedArcs
    ) -> _ProbabilisticLayerModel:
        if result.mode is RefinementMode.NEAR:
            assert result.center is not None
            model: _ProbabilisticLayerModel = NearGrowthModel(
                data,
                result.center,
                self.config,
                int(result.diagnostics.get("number_disturbance_sources", 0)),
            )
        else:
            assert result.direction is not None
            model = FarLayerModel(
                data,
                result.direction,
                result.inverse_distance or 0.0,
                self.config,
                fixed_infinity=result.mode is RefinementMode.INFINITY,
                number_disturbance_sources=int(
                    result.diagnostics.get("number_disturbance_sources", 0)
                ),
            )
        model.load_state_dict(copy.deepcopy(result.model_state))
        return model
