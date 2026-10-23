import json
from itertools import pairwise
from math import exp, log

import numpy as np
import pytest
from scipy.integrate import quad

from chemworld.data.correlation_calibration import CorrelationFitParameter
from chemworld.data.particle_distribution_calibration import DiameterDistributionCalibration
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.particle_distributions import (
    AnalyticDiameterDistribution,
    BinnedDiameterDistribution,
    CustomDiameterDistribution,
    InterpolatedDiameterDistribution,
    diameter_spacing,
)
from chemworld.physchem.sieve_catalog import SieveCatalog


def source():
    return InputSource(
        DataSourceProvenance("synthetic", 0, "Synthetic PSD fixtures", "1", "Synthetic"),
        "synthetic",
    )


def test_weighted_truncation_moments_and_divergence():
    for family in ("lognormal", "GGS", "RR"):
        model = AnalyticDiameterDistribution(
            family, 1e-4, 0.8 if family == "lognormal" else 2, lower_m=2e-5, upper_m=2e-4
        )
        for order in (0, 2, 3):
            integrated = quad(
                lambda x, model=model, order=order: model.pdf(x * 1e-4, order=order) * 1e-4,
                model.lower_m / 1e-4,
                model.upper_m / 1e-4,
                epsabs=1e-10,
            )[0]
            assert integrated == pytest.approx(1, rel=1e-8)
            assert model.cdf(model.lower_m, order=order) == 0
            assert model.cdf(model.upper_m, order=order) == 1
            for fraction in (0.01, 0.5, 0.99):
                assert model.cdf(
                    model.quantile(fraction, order=order), order=order
                ) == pytest.approx(fraction, abs=1e-9)
    with pytest.raises(ValueError, match="diverge"):
        AnalyticDiameterDistribution("GGS", 1e-4, 2).cdf(5e-5, order=0)
    with pytest.raises(ValueError, match="diverge"):
        AnalyticDiameterDistribution("RR", 1e-4, 2).moment(-3)
    model = AnalyticDiameterDistribution("lognormal", 1e-4, 0.8)
    assert model.mean_size(3, 3) == 1e-4
    assert model.mean_size(3, 2) == pytest.approx(1e-4 * exp(-(0.8**2) / 2))
    assert model.quantile(0.1, order=0) > 0


def test_interpolation_has_bounded_support_and_serializes():
    original = InterpolatedDiameterDistribution([1e-4, 2e-4, 3e-4, 4e-4], [0.4, 0, 0.6])
    model = InterpolatedDiameterDistribution.from_dict(json.loads(json.dumps(original.to_dict())))
    assert model.pdf(5e-5) == model.pdf(5e-4) == 0
    assert model.cdf(5e-5) == 0 and model.cdf(5e-4) == 1
    assert model.interval_fraction(2e-4, 3e-4) == 0
    assert model.interval_fraction(2e-4, 3e-4, order=2) == 0
    values = model.sample_cdf(np.linspace(0, 5e-4, 30))
    assert all(a <= b for a, b in pairwise(values))
    assert sum(model.sample_fractions([1e-4, 2e-4, 3e-4, 4e-4])) == 1
    assert model.sample_curve([2e-4])[0]["diameter_m"] == 2e-4
    with pytest.raises(ValueError, match="diverge"):
        InterpolatedDiameterDistribution([1e-4, 2e-4], [0.3, 0.7]).cdf(1e-4, order=0)


def test_custom_moments_spacing_and_binned_semantics():
    custom = CustomDiameterDistribution(lambda d: 1, scale_m=1, lower_m=1, upper_m=2, order=0)
    assert custom.cdf(1.5) == 0.5
    assert custom.moment(1) == 1.5
    assert custom.mean_size(0, 0) == pytest.approx(exp(2 * log(2) - 1))
    with pytest.raises(ValueError, match="nonnegative"):
        CustomDiameterDistribution(lambda d: -1, scale_m=1, lower_m=1, upper_m=2).cdf(1.5)
    with pytest.raises(ValueError, match="exactly one"):
        diameter_spacing(lower_m=1e-6, upper_m=1e-3, method="R10")
    assert diameter_spacing(lower_m=1e-6, points=11, method="R10")[-1] == pytest.approx(1e-5)
    bins = BinnedDiameterDistribution([1, 2, 4], [2, 3], order=0)
    assert bins.bin_power(0, -1) == log(2)
    assert bins.fractions(0) == pytest.approx((0.4, 0.6))
    assert BinnedDiameterDistribution.from_dict(json.loads(json.dumps(bins.to_dict()))).mean_size(
        3, 2
    ) == bins.mean_size(3, 2)


def test_native_radius_quadrature_preserves_volume_basis_and_json():
    model = AnalyticDiameterDistribution("lognormal", 2e-5, 0.4, lower_m=2e-6, upper_m=1e-4)
    restored = AnalyticDiameterDistribution.from_dict(
        json.loads(json.dumps(model.to_dict(), allow_nan=False))
    )
    points = restored.to_particle_quadrature(lower_m=2e-6, upper_m=1e-4, nodes=64, source=source())
    assert sum(points.volume_weights) == pytest.approx(1)
    mean_diameter = sum(
        2 * r * w for r, w in zip(points.radii_m, points.volume_weights, strict=True)
    )
    assert mean_diameter == pytest.approx(model.moment(1, order=3), rel=1e-8)
    assert points.source == source() and points.source_basis == "volume"
    assert type(points).from_dict(json.loads(json.dumps(points.to_dict()))) == points
    discrete = BinnedDiameterDistribution([1e-5, 2e-5, 4e-5], [0.2, 0.3, 0.5], order=0)
    nodes = discrete.to_particle_quadrature(source=source())
    assert nodes.weights("number") == pytest.approx((0.2, 0.3, 0.5))


def test_sieve_catalog_keeps_source_units_aliases_and_missing_fields():
    columns = (
        QuantityColumn("opening", "mm", "observation", (0, 100), "Nominal aperture"),
        QuantityColumn("wire", "mm", "observation", (0, 100), "Caller wire diameter"),
    )
    rows = tuple(
        ScientificRow(
            str(i),
            "synthetic",
            "TEST",
            f"TEST:{i}",
            "test",
            {
                "opening": InputValue(float(i), "mm", None, "Synthetic"),
                "wire": InputValue(None, "mm", None, "Not available", "missing"),
            },
        )
        for i in (1, 2, 3)
    )
    table = ScientificTable("sieves", "measurements", columns, (source(),), rows)
    model = SieveCatalog(table, series={"odd": ("1", "3")}, aliases={"1": ("fine",)})
    model = SieveCatalog.from_dict(json.loads(json.dumps(model.to_dict())))
    selected = model.lookup("TEST", "fine")
    assert selected.opening_m == 0.001
    assert selected.values["wire"].quality == "missing"
    assert model.spacing("odd", lower_m=0.001, upper_m=0.004) == (0.001, 0.003)
    with pytest.raises(ValueError, match="unknown"):
        model.lookup("ISO", "1")


def test_fitting_uses_only_training_rows_and_original_calibration_contract():
    target = AnalyticDiameterDistribution("GGS", 1e-4, 4)
    columns = (
        QuantityColumn("diameter", "m", "condition", (0, 1), "Diameter"),
        QuantityColumn("cdf", "dimensionless", "observation", (0, 1), "CDF"),
    )
    rows = []
    for i, ratio in enumerate((0.3, 0.5, 0.7, 0.9, 0.6)):
        split = "validation" if i == 4 else "train"
        rows.append(
            ScientificRow(
                str(i),
                "synthetic",
                split,
                str(i),
                split,
                {
                    "diameter": InputValue(ratio * 1e-4, "m", None, "Exact"),
                    "cdf": InputValue(
                        0.9 if i == 4 else target.cdf(ratio * 1e-4),
                        "dimensionless",
                        0.01,
                        "Synthetic weight",
                    ),
                },
            )
        )
    table = ScientificTable("fit", "measurements", columns, (source(),), tuple(rows))
    parameters = (
        CorrelationFitParameter("scale_m", (5e-5, 2e-4), 1.2e-4, "log"),
        CorrelationFitParameter("shape", (0.2, 8), 4.8, "log"),
    )
    calibration = DiameterDistributionCalibration(
        table, AnalyticDiameterDistribution("GGS", 1.2e-4, 4.8), parameters
    )
    report = calibration.fit(max_nfev=80)
    assert report["status"] == "fitted"
    assert report["estimate"] == pytest.approx({"scale_m": 1e-4, "shape": 4}, rel=1e-6)
    assert report["validation"]["weighted_rmse"] > 70
    assert all(run["split"] == "train" for run in report["runs"] if run["stage"] == "fit")
    assert report["resources"]["jacobian_calls"] > 0
    with pytest.raises(ValueError, match="fresh"):
        calibration.fit()
