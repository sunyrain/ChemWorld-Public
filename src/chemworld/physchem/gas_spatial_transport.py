"""Explicit isothermal/thermal spatial declaration and original isothermal closure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import fsum, isfinite, sqrt
from typing import Any

from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.gas_polynomial_transport import GasTransportPolynomialSpec
from chemworld.physchem.gas_transport import (
    GasTransportSpec,
    gas_transport_from_dict,
    gas_transport_response,
)
from chemworld.physchem.gas_transport_fluxes import gas_transport_fluxes
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network import ReactionNetworkSpec


@dataclass(frozen=True)
class GasSpatialTransportSpec:
    mesh: DiffusionMesh
    cell_vessel_ids: tuple[str, ...]
    transport: GasTransportSpec | GasTransportPolynomialSpec
    flux_model: str = "stefan-maxwell"
    maximum_mach: float = 0.05
    boundary_mode: str = "sealed"
    energy_equation: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_vessel_ids", tuple(self.cell_vessel_ids))
        ids = self.cell_vessel_ids
        if (
            self.mesh.geometry != "planar"
            or len(ids) != self.mesh.cells
            or len(set(ids)) != len(ids)
            or any(not isinstance(s, str) or not s for s in ids)
        ):
            raise ValueError("spatial gas needs one original vessel per planar mesh cell")
        if self.flux_model not in {"stefan-maxwell", "mixture-averaged"}:
            raise ValueError("unsupported spatial gas flux model")
        if self.boundary_mode not in {"sealed", "finite-flow"}:
            raise ValueError("unsupported spatial gas boundary mode")
        if type(self.energy_equation) is not bool:
            raise ValueError("spatial energy equation must be explicitly boolean")
        if self.transport.thermal_diffusion != "off":
            raise ValueError(
                "dynamic spatial transport requires thermal_diffusion='off'; "
                "Soret is not connected to this consumer"
            )
        if (
            isinstance(self.maximum_mach, bool)
            or not isfinite(self.maximum_mach)
            or not 0 < self.maximum_mach <= 0.05
        ):
            raise ValueError("sealed low-Mach gas requires a Mach ceiling in (0, 0.05]")

    @property
    def topology(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (f"spatial-face:{i}", self.cell_vessel_ids[i - 1], self.cell_vessel_ids[i])
            for i in range(1, self.mesh.cells)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "sealed-thermal-low-mach-v1"
            if self.energy_equation and self.boundary_mode == "sealed"
            else "finite-flow-thermal-low-mach-v1"
            if self.energy_equation
            else "sealed-isothermal-low-mach-v1"
            if self.boundary_mode == "sealed"
            else "finite-flow-isothermal-low-mach-v1",
            "mesh": {
                "geometry": self.mesh.geometry,
                "faces_m": list(self.mesh.faces_m),
                "area_m2": self.mesh.area_m2,
            },
            "cell_vessel_ids": list(self.cell_vessel_ids),
            "transport": self.transport.to_dict(),
            "flux_model": self.flux_model,
            "maximum_mach": self.maximum_mach,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GasSpatialTransportSpec:
        value = dict(payload)
        model = value.pop("model")
        if model not in {
            "sealed-isothermal-low-mach-v1",
            "finite-flow-isothermal-low-mach-v1",
            "sealed-thermal-low-mach-v1",
            "finite-flow-thermal-low-mach-v1",
        }:
            raise ValueError("unsupported spatial gas closure")
        return cls(
            DiffusionMesh(**value["mesh"]),
            tuple(value["cell_vessel_ids"]),
            gas_transport_from_dict(value["transport"]),
            value["flux_model"],
            value["maximum_mach"],
            "sealed" if model.startswith("sealed-") else "finite-flow",
            "-thermal-" in model,
        )


def spatial_gas_faces(
    spec: GasSpatialTransportSpec,
    network: ReactionNetworkSpec,
    temperature_K: float,
    amounts: Sequence[Mapping[str, float]],
    chemical_rates: Sequence[Mapping[str, float]],
    *,
    inlet_mol_s: float = 0.0,
    outlet_mol_s: float = 0.0,
) -> dict[str, Any]:
    """Algebraically eliminate bulk flow from the uniform-density continuity law.

    Thermodynamic pressure changes with the total inventory; it is never an
    independent stock. Molar and mass average velocities are distinct. Boundary
    species fluxes are zero for sealed mode. Finite-flow mode uses declared total
    molar boundary flows in the same continuity equation. Signed faces point right.
    """
    if spec.energy_equation:
        raise ValueError("thermal spatial gas requires the coupled energy/continuity closure")
    ids, mesh = network.species_ids, spec.mesh
    totals = [fsum(n.values()) for n in amounts]
    if len(amounts) != mesh.cells or len(chemical_rates) != mesh.cells or min(totals) <= 0:
        raise ValueError("spatial gas needs positive finite inventory in each declared cell")
    volumes = list(map(float, mesh.cell_volumes_m3))
    total_volume, total_amount = fsum(volumes), fsum(totals)
    density = total_amount / total_volume
    pressure = density * R * temperature_K
    x = [{s: n[s] / total for s in ids} for n, total in zip(amounts, totals, strict=True)]
    production = [fsum(r.values()) for r in chemical_rates]
    total_production = fsum(production)
    net_total_rate = total_production
    if spec.boundary_mode == "finite-flow":
        if any(not isfinite(f) or f < 0 for f in (inlet_mol_s, outlet_mol_s)):
            raise ValueError("spatial boundary flows must be finite and nonnegative")
        net_total_rate += inlet_mol_s - outlet_mol_s
    elif inlet_mol_s != 0 or outlet_mol_s != 0:
        raise ValueError("sealed spatial gas cannot consume boundary flows")
    thermo = network.imported_gas_thermochemistry
    faces: dict[str, dict[str, Any]] = {}
    bulk = inlet_mol_s
    for j, (identity, left, right) in enumerate(spec.topology, 1):
        bulk += production[j - 1] - volumes[j - 1] / total_volume * net_total_rate
        distance = float(mesh.centers_m[j] - mesh.centers_m[j - 1])
        # Linear interpolation to the actual face, including nonuniform meshes.
        weight = float((mesh.faces_m[j] - mesh.centers_m[j - 1]) / distance)
        composition = {s: (1 - weight) * x[j - 1][s] + weight * x[j][s] for s in ids}
        response = gas_transport_response(
            spec.transport,
            network,
            temperature_K=temperature_K,
            pressure_Pa=min(pressure, spec.transport.maximum_pressure_Pa),
            mole_fractions=composition,
        )
        if pressure > spec.transport.maximum_pressure_Pa:
            # Exact 1/P continuation of this declared dilute-gas law for solver
            # trials only. Accepted states stop at/below the transport ceiling.
            scale = response.pressure_Pa / pressure
            response = replace(
                response,
                pressure_Pa=pressure,
                molar_density_mol_m3=response.molar_density_mol_m3 / scale,
                density_kg_m3=response.density_kg_m3 / scale,
                binary_diffusivities_m2_s={
                    s: {k: d * scale for k, d in row.items()}
                    for s, row in response.binary_diffusivities_m2_s.items()
                },
                diffusion_coefficients_m2_s={
                    basis: {s: None if d is None else d * scale for s, d in row.items()}
                    for basis, row in response.diffusion_coefficients_m2_s.items()
                },
            )
        flux = gas_transport_fluxes(
            response,
            mole_fraction_gradients_per_m={s: (x[j][s] - x[j - 1][s]) / distance for s in ids},
            model=spec.flux_model,
            frame="molar",
            thermal_diffusion="off",
        )
        area = float(mesh.face_areas_m2[j])
        diffusion = {s: area * flux.molar_fluxes_mol_m2_s[s] for s in ids}
        upstream = x[j - 1] if bulk >= 0 else x[j]
        convection = {s: bulk * upstream[s] for s in ids}
        total = {s: diffusion[s] + convection[s] for s in ids}
        mass_rate = fsum(response.molecular_weights_kg_mol[s] * total[s] for s in ids)
        mass_velocity = mass_rate / (area * response.density_kg_m3)
        cp = fsum(composition[s] * thermo[s].evaluate(temperature_K).cp_J_mol_K for s in ids)
        mean_mass = response.density_kg_m3 / response.molar_density_mol_m3
        sound_speed = sqrt(cp / (cp - R) * R * temperature_K / mean_mass)
        faces[identity] = {
            "source": left,
            "target": right,
            "bulk_mol_s": bulk,
            "diffusion_mol_s": diffusion,
            "convection_mol_s": convection,
            "amount_rates_mol_s": total,
            "diffusion_enthalpy_W": area * flux.diffusion_enthalpy_flux_W_m2,
            "convection_enthalpy_W": fsum(
                response.partial_enthalpies_J_mol[s] * convection[s] for s in ids
            ),
            "enthalpy_W": fsum(response.partial_enthalpies_J_mol[s] * total[s] for s in ids),
            "molar_velocity_m_s": bulk / (area * density),
            "mass_velocity_m_s": mass_velocity,
            "mach": abs(mass_velocity) / sound_speed,
            "pressure_Pa": pressure,
            "temperature_K": temperature_K,
            "mole_fractions": composition,
            "diffusive_molar_residual_mol_s": fsum(diffusion.values()),
        }
    return {
        "faces": faces,
        "pressure_Pa": pressure,
        "pressure_rate_Pa_s": R * temperature_K * net_total_rate / total_volume,
        "maximum_mach": max((f["mach"] for f in faces.values()), default=0.0),
        "cell_total_rates_mol_s": dict(
            zip(
                spec.cell_vessel_ids,
                (v / total_volume * net_total_rate for v in volumes),
                strict=True,
            )
        ),
        "boundary_species_rates_mol_s": {
            "left": dict.fromkeys(ids, 0.0),
            "right": dict.fromkeys(ids, 0.0),
        },
    }
