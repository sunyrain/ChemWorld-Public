"""Mixture identity, explicit pressure references and actual heat consumption."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem import (
    ComponentSpec,
    MixtureSpec,
    PropertyEvaluation,
    TransportPropertyReport,
    compressed_liquid_conductivity,
    compressed_liquid_conductivity_derivatives,
    compressed_liquid_conductivity_report,
    dippr9h_conductivity,
    dippr9i_conductivity,
    filippov_conductivity,
    lindsay_bromley_conductivity,
    mixture_conductivity_ledger,
    wassiljewa_herning_zipperer_conductivity,
)
from chemworld.physchem.transport import internal_heat_transfer_coefficient, jacket_heat_transfer
from chemworld.physchem.transport_properties import (
    liquid_mixture_thermal_conductivity_dippr9h_ledger,
)


def mixture(x, phase):
    return MixtureSpec.from_mole_fractions(
        (
            ComponentSpec("methane", "CH4", default_phase=phase),
            ComponentSpec("ethane", "C2H6", default_phase=phase),
        ),
        x,
        phase_label=phase,
        temperature_K=300,
        pressure_Pa=101325,
    )


@pytest.mark.parametrize(
    "method,phase,extra",
    [
        ("DIPPR9H", "liquid", {}),
        ("DIPPR9I", "liquid", {"molar_volumes_m3_mol": (60e-6, 100e-6)}),
        ("Filippov", "liquid", {}),
        ("Wassiljewa_Herning_Zipperer", "gas", {}),
        (
            "Lindsay_Bromley",
            "gas",
            {"viscosities_Pa_s": (1e-5, 2e-5), "boiling_temperatures_K": (180, 250)},
        ),
    ],
)
def test_mixture_actual_composition_changes_heat_and_pure_endpoint(method, phase, extra):
    heat = []
    for x in ((1, 0), (0.3, 0.7), (0, 1)):
        state = mixture(x, phase)
        ledger = mixture_conductivity_ledger(state, (0.1, 0.2), method=method, **extra)
        data = json.loads(json.dumps(ledger.to_dict()))
        assert data["phase"] == phase and data["method_family"] == method
        assert data["contributions"]["methane"]["thermal_conductivity_W_m_K"] == 0.1
        if method in {"DIPPR9H", "Filippov"}:
            kernel = dippr9h_conductivity if method == "DIPPR9H" else filippov_conductivity
            assert ledger.mixture_value == pytest.approx(kernel(state.mass_fractions, (0.1, 0.2)))
            if x == (0.3, 0.7):
                assert ledger.mixture_value != pytest.approx(
                    kernel(state.mole_fractions, (0.1, 0.2))
                )
        h = internal_heat_transfer_coefficient(
            nusselt=3.66,
            thermal_conductivity_W_m_K=ledger.mixture_value,
            diameter_m=0.01,
        )
        heat.append(
            jacket_heat_transfer(
                area_m2=0.1,
                overall_u_W_m2_K=h,
                reactor_temperature_K=300,
                jacket_temperature_K=310,
                duration_s=5,
            ).heat_energy_J
        )
    assert heat[0] < heat[1] < heat[2]
    assert heat[2] == pytest.approx(2 * heat[0])


def test_original_dippr9h_ledger_keeps_shared_terms_and_warnings():
    ledger = liquid_mixture_thermal_conductivity_dippr9h_ledger(
        component_mass_fractions={"a": 0.25, "b": 0.75},
        component_thermal_conductivities_W_m_K={"a": 0.1, "b": 0.4},
    )
    assert ledger.mixture_value == dippr9h_conductivity((0.25, 0.75), (0.1, 0.4))
    assert ledger.contributions["a"]["inverse_square_contribution"] == pytest.approx(25)
    assert any("2x" in text for text in ledger.warnings)


def test_molar_volume_is_consumed_and_filippov_order_is_explicit():
    first = dippr9i_conductivity((0.4, 0.6), (50e-6, 100e-6), (0.1, 0.2))
    changed = dippr9i_conductivity((0.4, 0.6), (100e-6, 50e-6), (0.1, 0.2))
    assert changed < first
    assert first == pytest.approx(dippr9i_conductivity((0.6, 0.4), (100e-6, 50e-6), (0.2, 0.1)))
    assert filippov_conductivity((0.258, 0.742), (0.1692, 0.1528)) == pytest.approx(0.159291676288)
    assert filippov_conductivity((0.742, 0.258), (0.1528, 0.1692)) != pytest.approx(0.159291676288)
    assert lindsay_bromley_conductivity(
        323.15,
        (0.23, 0.77),
        (0.01939, 0.01231),
        (1.002e-5, 1.015e-5),
        (248.31, 248.93),
        (46.07, 50.49),
    ) == pytest.approx(0.013902644179693132)


@pytest.mark.parametrize(
    "fractions,ks",
    [
        ((), ()),
        ((0.2, 0.3), (0.1, 0.2)),
        ((1, -0.1), (0.1, 0.2)),
        ((1,), (0,)),
        ((1,), (float("nan"),)),
        ((1,), (True,)),
        ((1, 0), (0.1,)),
    ],
)
def test_invalid_mixture_inputs_reject(fractions, ks):
    with pytest.raises(ValueError):
        dippr9h_conductivity(fractions, ks)


def test_mixture_no_ignored_parameters_or_inconsistent_composition():
    state = mixture((0.3, 0.7), "liquid")
    for method, kwargs in (
        ("unknown", {}),
        ("DIPPR9I", {}),
        ("Lindsay_Bromley", {}),
        ("DIPPR9H", {"viscosities_Pa_s": (1e-5, 2e-5)}),
    ):
        with pytest.raises(ValueError):
            mixture_conductivity_ledger(state, (0.1, 0.2), method=method, **kwargs)
    with pytest.raises(ValueError, match="same composition"):
        mixture_conductivity_ledger(
            replace(state, mass_fractions=(0.3, 0.7)), (0.1, 0.2), method="DIPPR9H"
        )
    with pytest.raises(ValueError, match="two components"):
        filippov_conductivity((1,), (0.1,))
    with pytest.raises(ValueError):
        wassiljewa_herning_zipperer_conductivity((0.3, 0.7), (0.1, 0.2), (20, -40))


PRESSURE = {
    "critical_temperature_K": 591.8,
    "critical_pressure_Pa": 4.1e6,
    "reference_conductivity_W_m_K": 0.129,
}


def reference_report(t=384.67, pressure=101325, unit="W/(m*K)", value=0.129):
    return TransportPropertyReport(
        PropertyEvaluation(
            "thermal_conductivity",
            "synthetic",
            "constant",
            value,
            unit,
            {"temperature": t, "pressure": pressure},
            ("source warning",),
        ),
        "liquid",
        "source",
        "out_of_range",
        reference_reading=("source reference",),
    )


@pytest.mark.parametrize("method", ("DIPPR9G", "Missenard"))
def test_pressure_report_consumes_actual_pressure_reference_units_and_heat(method):
    source = reference_report(unit="mW/(m*K)", value=129)
    heat = []
    for p in (3e7, 2e8):
        report = compressed_liquid_conductivity_report(
            source,
            temperature_K=384.67,
            pressure_Pa=p,
            method=method,
            reference_kind="atmospheric",
            critical_temperature_K=591.8,
            critical_pressure_Pa=4.1e6,
        )
        k = report.evaluation.value
        assert k == pytest.approx(
            compressed_liquid_conductivity(384.67, p, method=method, **PRESSURE)
        )
        assert report.validity_status == "out_of_range"
        assert report.evaluation.warnings == ("source warning",)
        assert json.loads(json.dumps(report.to_dict()))["property"]["inputs"]["pressure"] == p
        heat.append(
            internal_heat_transfer_coefficient(
                nusselt=3.66,
                thermal_conductivity_W_m_K=k,
                diameter_m=0.01,
            )
        )
    assert heat[1] > heat[0]


def test_missenard_table_domain_explicit_clamping_and_derivative_knots():
    assert compressed_liquid_conductivity(
        304, 633e6, method="Missenard", **PRESSURE
    ) == pytest.approx(0.2198375777069657)
    for tr, pr in ((0.4, 10), (0.9, 10), (0.65, 0.5), (0.65, 250)):
        with pytest.raises(ValueError, match="Missenard requires"):
            compressed_liquid_conductivity(tr * 591.8, pr * 4.1e6, method="Missenard", **PRESSURE)
        assert (
            compressed_liquid_conductivity(
                tr * 591.8, pr * 4.1e6, method="Missenard", outside_table="nearest", **PRESSURE
            )
            > 0.129
        )
    for variable, tr, pr in (("temperature", 0.6, 30), ("pressure", 0.65, 10)):
        with pytest.raises(ValueError, match="knot"):
            compressed_liquid_conductivity_derivatives(
                tr * 591.8,
                pr * 4.1e6,
                method="Missenard",
                variable=variable,
                **PRESSURE,
            )
    with pytest.raises(ValueError, match="T <= Tc"):
        compressed_liquid_conductivity(600, 1e7, method="DIPPR9G", **PRESSURE)


@pytest.mark.parametrize(
    "source,t,p,kind",
    [
        (reference_report(t=300), 384.67, 1e7, "atmospheric"),
        (reference_report(pressure=None), 384.67, 1e7, "saturation"),
        (reference_report(pressure=2e5), 384.67, 1e7, "atmospheric"),
        (reference_report(pressure=2e5), 384.67, 1e5, "saturation"),
        (reference_report(), 384.67, 1e7, "unknown"),
    ],
)
def test_pressure_reference_is_explicit(source, t, p, kind):
    with pytest.raises(ValueError):
        compressed_liquid_conductivity_report(
            source,
            temperature_K=t,
            pressure_Pa=p,
            method="DIPPR9G",
            reference_kind=kind,
            critical_temperature_K=591.8,
            critical_pressure_Pa=4.1e6,
        )


def test_saturation_reference_and_explicit_table_extension_remain_visible():
    source = replace(reference_report(t=250, pressure=2e5), validity_status="valid")
    report = compressed_liquid_conductivity_report(
        source,
        temperature_K=250,
        pressure_Pa=2e7,
        method="Missenard",
        reference_kind="saturation",
        critical_temperature_K=591.8,
        critical_pressure_Pa=4.1e6,
        outside_table="nearest",
    )
    assert report.validity_status == "out_of_range"
    assert "clamped" in report.evaluation.warnings[-1]
    assert report.evaluation.inputs["reference_pressure"] == 2e5
