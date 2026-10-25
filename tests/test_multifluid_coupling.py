"""Coupled-potential limits, parameter contracts and existing caloric consumers."""

import json
from dataclasses import replace
from math import log, log1p

import pytest
from tests.test_multifluid import synthetic_model

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.generic_saft import CanonicalSAFTAssociation, MaskedSAFTAssociation
from chemworld.physchem.molecular_consumers import (
    molecular_model_from_dict,
    molecular_model_to_dict,
)
from chemworld.physchem.molecular_eos import molecular_scaled_derivative
from chemworld.physchem.multifluid_coupling import (
    MultifluidActivityModel,
    MultifluidAssociationModel,
)
from chemworld.physchem.saft_association import AssociationSite, SAFTAssociation

POLICY = "explicit_joint_parameterization"


def coupled_model(kind, n=3):
    base = synthetic_model(n, beta=1.0)
    ids = base.component_ids
    b = tuple((30 + 5 * i) * 1e-6 for i in range(n))
    if kind == "activity":
        p = {
            f"lambda_a:{left}|{right}": log(b[j] / b[i]) - 0.1 * (i + j + 1)
            for i, left in enumerate(ids)
            for j, right in enumerate(ids)
            if i != j
        }
        p.update(
            {
                f"lambda_b:{left}|{right}": -20.0 * (i - j)
                for i, left in enumerate(ids)
                for j, right in enumerate(ids)
                if i != j
            }
        )
        return MultifluidActivityModel(
            base,
            ActivityModelSpec("synthetic-Wilson", ids, "wilson", p),
            b,
            1.17,
            "wilson_volume_residual",
            "synthetic joint coupling",
            POLICY,
        )
    sites = tuple(AssociationSite(cid, label, 1) for cid in ids for label in ("e", "H"))
    mask = tuple(
        tuple(a.component_id != z.component_id and a.site_id != z.site_id for z in sites)
        for a in sites
    )
    spec = SAFTAssociation(
        sites,
        ((400.0,) * len(sites),) * len(sites),
        ((100e-30,) * len(sites),) * len(sites),
        "synthetic bonds",
    )
    kernel = CanonicalSAFTAssociation(ids, b, spec, "synthetic contact", "Carnahan-Starling")
    return MultifluidAssociationModel(
        base, MaskedSAFTAssociation(kernel, mask), "synthetic", POLICY
    )


@pytest.mark.parametrize("kind", ("activity", "association"))
@pytest.mark.parametrize("index", (0, 1, 2))
def test_pure_limits_preserve_all_temperature_density_derivatives(kind, index):
    model = coupled_model(kind)
    xs = tuple(float(i == index) for i in range(3))
    for i in range(4):
        for j in range(4 - i):
            args = {"inverse_temperature_order": i, "density_order": j}
            assert molecular_scaled_derivative(model, 400.0, 700.0, xs, **args) == pytest.approx(
                molecular_scaled_derivative(model.base, 400.0, 700.0, xs, **args), abs=1e-12
            )
    # Absent molecules still have a nonzero insertion response: do not early-return
    # a constant zero just because the primal composition is pure.
    other = (index + 1) % 3

    def increment(values):
        fractions = list(xs)
        fractions[index], fractions[other] = 1 - values["x"], values["x"]
        return model.alphar(400.0, 700.0, fractions) - model.base.alphar(400.0, 700.0, fractions)

    assert abs(partial_derivative(increment, {"x": 0.0}, ("x",))) > 1e-6


@pytest.mark.parametrize("kind", ("activity", "association"))
def test_codec_caloric_and_thermodynamic_identity(kind):
    model = coupled_model(kind)
    encoded = json.loads(json.dumps(molecular_model_to_dict(model)))
    restored = molecular_model_from_dict(encoded)
    assert restored.to_dict() == model.to_dict()
    state = model.caloric_model().state(400.0, 700.0, (0.2, 0.3, 0.5))
    assert state.helmholtz_energy_J_mol == pytest.approx(
        state.internal_energy_J_mol - 400 * state.entropy_J_mol_K
    )
    assert state.gibbs_energy_J_mol == pytest.approx(
        state.enthalpy_J_mol - 400 * state.entropy_J_mol_K
    )
    assert state.residual_helmholtz_J_mol == pytest.approx(
        400 * model.gas_constant_J_mol_K * model.alphar(400.0, 700.0, (0.2, 0.3, 0.5))
    )
    assert state.pressure_Pa != model.base.state(400.0, 700.0, (0.2, 0.3, 0.5)).pressure_Pa
    payload = model.to_dict()
    payload["ignored"] = 1
    with pytest.raises(ValueError, match="exactly"):
        type(model).from_dict(payload)


@pytest.mark.parametrize("kind", ("activity", "association"))
def test_actual_coupled_pressure_controls_source_domain(kind):
    model = coupled_model(kind)
    xs = (0.2, 0.3, 0.5)
    base_pressure = model.base.state(400.0, 700.0, xs).pressure_Pa
    pressure = model.state(400.0, 700.0, xs).pressure_Pa
    limit = (base_pressure + pressure) / 2
    bounded = replace(
        model,
        base=replace(
            model.base,
            fluids=tuple(replace(f, maximum_pressure_Pa=limit) for f in model.base.fluids),
        ),
    )
    if pressure > limit:
        with pytest.raises(ValueError, match="pressure"):
            bounded.state(400.0, 700.0, xs)
    else:
        assert bounded.state(400.0, 700.0, xs).pressure_Pa == pressure
        with pytest.raises(ValueError, match="pressure"):
            bounded.base.state(400.0, 700.0, xs)


@pytest.mark.parametrize("kind", ("activity", "association"))
def test_vacuum_departure_and_invalid_joint_declaration(kind):
    model = coupled_model(kind)
    correction = model.activity_departure if kind == "activity" else model.association_departure
    assert correction(400.0, 0.0, (0.2, 0.3, 0.5)) == pytest.approx(0.0, abs=1e-14)
    with pytest.raises(ValueError, match="explicit_joint"):
        replace(model, combination_policy="inherit_base_qualification")
    with pytest.raises(ValueError, match="source"):
        replace(model, source="")


def test_self_and_asymmetric_association_are_rejected():
    model = coupled_model("association")
    mask = [list(row) for row in model.association.interaction_mask]
    mask[0][1] = mask[1][0] = True
    with pytest.raises(ValueError, match="self association"):
        replace(model, association=replace(model.association, interaction_mask=mask))
    mask = [list(row) for row in model.association.interaction_mask]
    mask[0][3] = False
    with pytest.raises(ValueError, match="reciprocal"):
        replace(model, association=replace(model.association, interaction_mask=mask))


def test_activity_convention_and_reference_parameters_explicit():
    model = coupled_model("activity")
    for changes in ({"u": 0}, {"covolumes_m3_mol": (1.0,)}, {"activity_convention": "automatic"}):
        with pytest.raises(ValueError):
            replace(model, **changes)
    ideal = ActivityModelSpec("explicit-residual-ideal", model.component_ids)
    with pytest.raises(ValueError, match="Wilson"):
        replace(model, activity=ideal)
    declared = replace(model, activity=ideal, activity_convention="declared_residual")
    assert declared.residual_activity(400.0, (0.2, 0.3, 0.5)) == 0


def test_declared_residual_margules_has_analytic_pressure_and_thermal_limit():
    base = synthetic_model(2, ideal=True, beta=1.0)
    activity = ActivityModelSpec(
        "explicit-residual-Margules", base.component_ids, "margules", {"A:A|B": 0.7}
    )
    model = MultifluidActivityModel(
        base, activity, (3e-5, 4e-5), 1.17, "declared_residual", "synthetic GE", POLICY
    )
    t, rho, xs = 400.0, 700.0, (0.2, 0.8)
    bm, ge, scale = (
        sum(x * b for x, b in zip(xs, model.covolumes_m3_mol, strict=True)),
        0.112,
        log1p(1 / 1.17),
    )
    state = model.state(t, rho, xs)
    expected_z = 1 + ge * rho * bm / ((1 + rho * bm) * scale)
    assert state.pressure_Pa == pytest.approx(rho * model.gas_constant_J_mol_K * t * expected_z)
    assert state.residual_internal_energy_J_mol == pytest.approx(0.0, abs=1e-12)
    assert state.residual_helmholtz_J_mol == pytest.approx(
        model.gas_constant_J_mol_K * t * ge * log1p(rho * bm) / scale
    )


@pytest.mark.parametrize("kind", ("activity", "association"))
def test_permuted_component_order_does_not_change_potential(kind):
    model = coupled_model(kind)
    reversed_base = replace(model.base, fluids=model.base.fluids[::-1])
    if kind == "activity":
        permuted = replace(
            model,
            base=reversed_base,
            covolumes_m3_mol=model.covolumes_m3_mol[::-1],
            activity=replace(model.activity, component_ids=model.component_ids[::-1]),
        )
    else:
        kernel = replace(
            model.association.kernel,
            component_ids=model.component_ids[::-1],
            covolumes_m3_mol=model.association.kernel.covolumes_m3_mol[::-1],
        )
        permuted = replace(
            model, base=reversed_base, association=replace(model.association, kernel=kernel)
        )
    assert permuted.alphar(400.0, 700.0, (0.5, 0.3, 0.2)) == pytest.approx(
        model.alphar(400.0, 700.0, (0.2, 0.3, 0.5)), abs=2e-14
    )
