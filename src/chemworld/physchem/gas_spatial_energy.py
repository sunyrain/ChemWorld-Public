"""Nonisothermal low-Mach continuity/energy on original finite gas inventories.

For m fixed-volume cells a transient 2m linear system solves m-1 bulk molar
flows, m temperature derivatives and Pdot/P. Upwind states must agree with the
resulting flow signs. Pressure is derived from current n/T, never a new stock.
The original CoupledSystem owns integration, total U, acceptance and events.
"""

from __future__ import annotations

from dataclasses import replace
from math import fsum, isclose, sqrt
from typing import Any

import numpy as np

from chemworld.physchem.coupled_system import (
    DEFAULT_COUPLED_ODE_POLICY,
    CoupledAdvanceResult,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    HeatFlux,
    InternalTransfer,
    MaterialPort,
    ProcessContribution,
    assemble_coupled_rates,
)
from chemworld.physchem.gas_network_energy import (
    _device_domain,
    _NetworkCalorics,
    gas_link_heat_key,
    gas_loss_key,
    gas_reaction_energy_key,
    gas_temperature_key,
)
from chemworld.physchem.gas_polynomial_transport import GasTransportPolynomialSpec
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    _key,
    gas_bath_key,
    gas_enthalpy_key,
    gas_inventory_key,
    gas_material_topology,
    gas_network_coordinates,
    gas_node_pressure,
    gas_node_view,
    gas_transfer_key,
)
from chemworld.physchem.gas_spatial_flow import add_spatial_boundary_faces
from chemworld.physchem.gas_spatial_reactor import (
    spatial_bulk_key,
    spatial_diffusion_heat_key,
    spatial_diffusion_key,
)
from chemworld.physchem.gas_transport import gas_transport_response
from chemworld.physchem.gas_transport_fluxes import gas_transport_fluxes
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reactor_solvers import reaction_network_process
from chemworld.physchem.residual_system import linear_solve


def spatial_fourier_key(face: str) -> str:
    return _key("spatial_fourier_J", face)


class _SpatialEnergyClosure:
    """Immutable model references and transient evaluation; no second state owner."""

    def __init__(
        self,
        spec: GasReactorNetworkSpec,
        network: ReactionNetworkSpec,
        thermal_device: ThermalDeviceSpec | None,
    ) -> None:
        spatial = spec.spatial_transport
        if spatial is None or not spatial.energy_equation or not spec.energy_nodes:
            raise ValueError("nonisothermal spatial gas needs explicit mesh and all energy nodes")
        self.spec, self.network, self.spatial = spec, network, spatial
        self.device = thermal_device
        self.nodes = {n.vessel_id: n for n in spec.nodes}
        self.energy = {e.vessel_id: e for e in spec.energy_nodes}
        self.thermal = _NetworkCalorics(spec, network)
        self.models = self.thermal.local_models
        self.thermo = network.imported_gas_thermochemistry
        if set(self.thermo) != set(network.species_ids) or any(
            s.phase != "gas" for s in network.species
        ):
            raise ValueError("thermal spatial gas requires original gas thermochemistry")
        _device_domain(spec, thermal_device)
        for name, energy in self.energy.items():
            self.models[name].validate_temperature_interval(
                energy.minimum_temperature_K, energy.maximum_temperature_K
            )
        self.fit_temperature_domains: dict[str, tuple[float, float]] = {}
        self.rhs_temperature_domains: dict[str, tuple[float, float]] = {}
        if isinstance(spatial.transport, GasTransportPolynomialSpec):
            if spatial.transport.thermal_diffusion != "off":
                raise ValueError("thermal source-fit spatial gas requires thermal_diffusion='off'")
            thermo_low = max(s.min_temperature_K for s in self.thermo.values())
            thermo_high = min(s.max_temperature_K for s in self.thermo.values())
            for name, energy in self.energy.items():
                low, high = thermo_low, thermo_high
                if name in spatial.cell_vessel_ids:
                    fit_low, fit_high = spatial.transport.fit_temperature_range_K
                    low, high = max(low, fit_low), min(high, fit_high)
                    common = (
                        max(low, energy.minimum_temperature_K),
                        min(high, energy.maximum_temperature_K),
                    )
                    if common[0] >= common[1]:
                        raise ValueError(f"no common temperature domain for source-fit cell {name}")
                    self.fit_temperature_domains[name] = common
                self.rhs_temperature_domains[name] = (low, high)
        self.reaction = reaction_network_process(network, species_thermo=self.thermo)

    def views(self, state: CoupledStateView) -> dict[str, CoupledStateView]:
        return {name: gas_node_view(state, node, self.network) for name, node in self.nodes.items()}

    def extend_source_boundary_trial(self, state: CoupledStateView) -> CoupledStateView:
        """Policy B: extend only the numerical RHS/Mach view, never a snapshot.

        Raw event callbacks retain their signed distances. The shared ODE
        wrapper counts RHS extensions; this is not a claim that its raw trial
        vectors stay inside the fit or that all trials are archived.
        """
        anchor_t = state.temperature_K
        coordinates = dict(state.coordinates)
        changed = False
        for name, (low, high) in self.rhs_temperature_domains.items():
            value = (
                anchor_t
                if name == self.spec.anchor_vessel_id
                else coordinates[gas_temperature_key(name)]
            )
            bounded = max(low, min(high, value))
            if bounded != value:
                changed = True
                if name == self.spec.anchor_vessel_id:
                    anchor_t = bounded
                else:
                    coordinates[gas_temperature_key(name)] = bounded
        return replace(state, temperature_K=anchor_t, coordinates=coordinates) if changed else state

    def validate_fit_temperature(self, state: CoupledStateView) -> None:
        if not self.fit_temperature_domains:
            return
        low, high = self.spatial.transport.fit_temperature_range_K
        for name in self.fit_temperature_domains:
            temperature = (
                state.temperature_K
                if name == self.spec.anchor_vessel_id
                else state.coordinates[gas_temperature_key(name)]
            )
            if not low <= temperature <= high:
                raise ValueError(f"source-fit temperature outside declared domain: {name}")

    def fit_domain_receipt(self) -> dict[str, Any]:
        return {
            "policy": "B:constant-boundary-trial-view",
            "fit_temperature_range_K": list(self.spatial.transport.fit_temperature_range_K),
            "cell_temperature_domains_K": {
                name: list(bounds) for name, bounds in self.fit_temperature_domains.items()
            },
            "rhs_trial_temperature_domains_K": {
                name: list(bounds) for name, bounds in self.rhs_temperature_domains.items()
            },
            "coefficient_policy": "strict fit-domain evaluation in the numerical view",
            "accepted_states_projected": False,
            "raw_rhs_trials_guaranteed_in_fit_domain": False,
            "complete_raw_trial_trace": False,
            "t0_boundary_policy": "zero-duration stop without an inward-direction test",
            "concurrent_event_limit": (
                "another selected event may not provide a fit-feasible rounded root; "
                "strict accepted-state validation fails closed"
            ),
            "rhs_extension": None,
        }

    def heat(
        self, views: dict[str, CoupledStateView]
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        heater = {
            name: self.device.delivered_power_w(view.temperature_K)
            if self.device is not None and name == self.spec.anchor_vessel_id
            else 0.0
            for name, view in views.items()
        }
        loss = {
            name: self.energy[name].loss_conductance_W_K
            * (view.temperature_K - self.energy[name].environment_temperature_K)
            for name, view in views.items()
        }
        links = {
            edge.connection_id: edge.conductance_W_K
            * (
                views[edge.source_vessel_id].temperature_K
                - views[edge.target_vessel_id].temperature_K
            )
            for edge in self.spec.heat_links
        }
        return heater, loss, links

    def evaluate(
        self, state: CoupledStateView
    ) -> tuple[
        dict[str, ProcessContribution],
        dict[str, dict[str, float]],
        dict[str, Any],
        dict[str, float],
        dict[str, float],
        dict[str, float],
    ]:
        self.validate_fit_temperature(state)
        ids, cells, mesh = self.network.species_ids, self.spatial.cell_vessel_ids, self.spatial.mesh
        views = self.views(state)
        total = {name: fsum(v.amounts_mol.values()) for name, v in views.items()}
        if min(total.values()) <= 0 or any(v.temperature_K <= 0 for v in views.values()):
            raise ValueError("thermal spatial gas requires positive node inventory and temperature")
        x = {
            name: {s: view.amounts_mol[s] / total[name] for s in ids}
            for name, view in views.items()
        }
        temperatures = {name: v.temperature_K for name, v in views.items()}
        # This reduces to the common ideal-gas pressure on the accepted manifold.
        pressure = (
            R
            * fsum(total[name] for name in cells)
            / fsum(self.nodes[name].volume_L / (1000 * temperatures[name]) for name in cells)
        )
        chemistry = {
            name: self.reaction.evaluate(v)
            if self.nodes[name].chemistry_enabled
            else ProcessContribution()
            for name, v in views.items()
        }
        dn = {
            name: {s: c.amount_rates_mol_s.get(s, 0.0) for s in ids}
            for name, c in chemistry.items()
        }
        heater, loss, links = self.heat(views)
        energy = {name: heater[name] - loss[name] for name in views}
        for link in self.spec.heat_links:
            energy[link.source_vessel_id] -= links[link.connection_id]
            energy[link.target_vessel_id] += links[link.connection_id]
        u = {
            name: {s: self.models[name].species_energy_j_mol(s, v.temperature_K) for s in ids}
            for name, v in views.items()
        }
        capacity = {name: self.models[name].heat_capacity_j_k(v) for name, v in views.items()}
        if min(capacity.values()) <= 0:
            raise ValueError("thermal gas/support heat capacity must be positive")
        faces: dict[str, Any] = {}
        responses = {}
        for j, (face, left, right) in enumerate(self.spatial.topology, 1):
            distance = float(mesh.centers_m[j] - mesh.centers_m[j - 1])
            weight = float((mesh.faces_m[j] - mesh.centers_m[j - 1]) / distance)
            t = (1 - weight) * temperatures[left] + weight * temperatures[right]
            if self.fit_temperature_domains:
                # Equal boundary temperatures remain exactly at that boundary;
                # the original LJ path retains its original arithmetic.
                t = temperatures[left] + weight * (temperatures[right] - temperatures[left])
            composition = {s: (1 - weight) * x[left][s] + weight * x[right][s] for s in ids}
            response = gas_transport_response(
                self.spatial.transport,
                self.network,
                temperature_K=t,
                pressure_Pa=min(pressure, self.spatial.transport.maximum_pressure_Pa),
                mole_fractions=composition,
            )
            if pressure > self.spatial.transport.maximum_pressure_Pa:
                # Same exact dilute 1/P trial extension as the old spatial model;
                # every accepted state must remain at/below the pressure event.
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
                mole_fraction_gradients_per_m={
                    s: (x[right][s] - x[left][s]) / distance for s in ids
                },
                temperature_gradient_K_m=(temperatures[right] - temperatures[left]) / distance,
                model=self.spatial.flux_model,
                frame="molar",
                thermal_diffusion="off",
            )
            area = float(mesh.face_areas_m2[j])
            diff = {s: area * flux.molar_fluxes_mol_m2_s[s] for s in ids}
            hdiff = area * flux.diffusion_enthalpy_flux_W_m2
            fourier = area * flux.fourier_heat_flux_W_m2
            faces[face] = {
                "source": left,
                "target": right,
                "diffusion_mol_s": diff,
                "diffusion_enthalpy_W": hdiff,
                "fourier_heat_W": fourier,
                "temperature_K": t,
                "pressure_Pa": pressure,
                "mole_fractions": composition,
                "diffusive_molar_residual_mol_s": fsum(diff.values()),
            }
            responses[face] = response
            for name, sign in ((left, -1), (right, 1)):
                for s in ids:
                    dn[name][s] += sign * diff[s]
                energy[name] += sign * (hdiff + fourier)
        packet: dict[str, Any] = {"faces": faces, "pressure_Pa": pressure, "maximum_mach": 0.0}
        if self.spatial.boundary_mode == "finite-flow":
            boundary_packet: dict[str, Any] = {"faces": {}, "pressure_Pa": pressure}
            add_spatial_boundary_faces(self.spec, self.network, views, boundary_packet)
            faces.update(boundary_packet.pop("faces"))
            packet.update(boundary_packet)
            for edge in self.spec.connections:
                row = faces[edge.connection_id]
                for name, sign in ((row["source"], -1), (row["target"], 1)):
                    for s in ids:
                        dn[name][s] += sign * row["amount_rates_mol_s"][s]
                    energy[name] += sign * row["enthalpy_W"]
        m = len(cells)
        indexes = {name: j for j, name in enumerate(cells)}
        signs = tuple(True for _ in self.spatial.topology)
        seen = set()
        for _iteration in range(max(8, 4 * m)):
            if signs in seen:
                raise RuntimeError("thermal spatial upwind closure did not resolve its flow signs")
            seen.add(signs)
            matrix = np.zeros((2 * m, 2 * m))
            rhs = np.zeros(2 * m)
            for j, name in enumerate(cells):
                matrix[j, m - 1 + j] = capacity[name]
                rhs[j] = energy[name] - fsum(u[name][s] * dn[name][s] for s in ids)
                matrix[m + j, m - 1 + j] = -total[name] / temperatures[name]
                matrix[m + j, -1] = total[name]
                rhs[m + j] = fsum(dn[name].values())
            for k, (_, left, right) in enumerate(self.spatial.topology):
                donor = left if signs[k] else right
                h = {s: self.thermo[s].evaluate(temperatures[donor]).enthalpy_J_mol for s in ids}
                for name, sign in ((left, -1), (right, 1)):
                    j = indexes[name]
                    matrix[j, k] = -sign * fsum(x[donor][s] * (h[s] - u[name][s]) for s in ids)
                    matrix[m + j, k] = -sign
            scales = np.max(np.abs(matrix), axis=1)
            solution = linear_solve(matrix / scales[:, None], rhs / scales)
            if not np.all(np.isfinite(solution)):
                raise RuntimeError("nonfinite thermal spatial continuity/energy solution")
            # Homogeneous exact-zero bulk flow has no preferred donor. Retain
            # its orientation within linear-solve roundoff only; never clip
            # the signed physical flow or project any original inventory.
            flow_scale = max(
                *(abs(v) for v in rhs[m:]),
                *(
                    abs(total[name] * solution[m - 1 + j] / temperatures[name])
                    for j, name in enumerate(cells)
                ),
                *(abs(v) for v in solution[: m - 1]),
            )
            roundoff = 128 * np.finfo(float).eps * flow_scale
            next_signs = tuple(
                signs[k] if abs(value) <= roundoff else value >= 0
                for k, value in enumerate(solution[: m - 1])
            )
            if next_signs == signs:
                break
            signs = next_signs
        else:
            raise RuntimeError("thermal spatial upwind iteration budget exhausted")
        residual = matrix @ solution - rhs
        if np.max(np.abs(residual) / (1 + np.abs(rhs))) > 1e-9:
            raise RuntimeError("thermal spatial continuity/energy closure residual")
        for j, (face, left, right) in enumerate(self.spatial.topology):
            row, response = faces[face], responses[face]
            rate = float(solution[j])
            donor = left if rate >= 0 else right
            conv = {s: rate * x[donor][s] for s in ids}
            hconv = fsum(
                self.thermo[s].evaluate(temperatures[donor]).enthalpy_J_mol * conv[s] for s in ids
            )
            amounts = {s: conv[s] + row["diffusion_mol_s"][s] for s in ids}
            for name, sign in ((left, -1), (right, 1)):
                for s in ids:
                    dn[name][s] += sign * conv[s]
                energy[name] += sign * hconv
            area = float(mesh.face_areas_m2[j + 1])
            mass_rate = fsum(response.molecular_weights_kg_mol[s] * amounts[s] for s in ids)
            velocity = mass_rate / (area * response.density_kg_m3)
            cp = fsum(
                row["mole_fractions"][s] * self.thermo[s].evaluate(row["temperature_K"]).cp_J_mol_K
                for s in ids
            )
            mass = response.density_kg_m3 / response.molar_density_mol_m3
            sound = sqrt(cp / (cp - R) * R * row["temperature_K"] / mass)
            row.update(
                bulk_mol_s=rate,
                convection_mol_s=conv,
                amount_rates_mol_s=amounts,
                convection_enthalpy_W=hconv,
                enthalpy_W=hconv + row["diffusion_enthalpy_W"],
                molar_velocity_m_s=rate / (area * response.molar_density_mol_m3),
                mass_velocity_m_s=velocity,
                mach=abs(velocity) / sound,
            )
        # Original node material ports own local open-system enthalpy semantics.
        # Advective and diffusive ports stay distinct: their molar rates can
        # cancel while their enthalpies, evaluated at different T, do not.
        ports: dict[str, list[MaterialPort]] = {name: [] for name in views}
        port_rates: dict[str, dict[str, float]] = {name: {} for name in views}
        port_enthalpies: dict[str, dict[str, float]] = {name: {} for name in views}
        heat = {name: heater[name] for name in views}
        for link in self.spec.heat_links:
            heat[link.source_vessel_id] -= links[link.connection_id]
            heat[link.target_vessel_id] += links[link.connection_id]
        for face, row in faces.items():
            left, right = row["source"], row["target"]
            heat[left] -= row["fourier_heat_W"]
            heat[right] += row["fourier_heat_W"]
            for kind in ("convection", "diffusion"):
                for species, rate in row[f"{kind}_mol_s"].items():
                    source, target = (left, right) if rate >= 0 else (right, left)
                    temperature = (
                        row["temperature_K"] if kind == "diffusion" else temperatures[source]
                    )
                    h = self.thermo[species].evaluate(temperature).enthalpy_J_mol
                    for name, direction in ((source, "out"), (target, "in")):
                        key = _key("local-spatial", face, kind, species, direction)
                        ports[name].append(MaterialPort(key, species, source, target, direction))
                        port_rates[name][key] = abs(rate)
                        port_enthalpies[name][key] = h
        tdot = {}
        for name, view in views.items():
            contribution = ProcessContribution(
                chemistry[name].amount_rates_mol_s,
                port_rates_mol_s=port_rates[name],
                port_enthalpies_J_mol=port_enthalpies[name],
            )

            def contribution_at(
                _: CoupledStateView, c: ProcessContribution = contribution
            ) -> ProcessContribution:
                return c

            def heat_at(
                _: CoupledStateView, q: float = heat[name], lost: float = loss[name]
            ) -> HeatFlux:
                return HeatFlux(q, lost)

            local = assemble_coupled_rates(
                view,
                processes=(CoupledProcess("spatial-node-balance", contribution_at),),
                thermodynamics=self.models[name],
                ports=tuple(ports[name]),
                heat_flux=heat_at,
            )
            tdot[name] = local.temperature_rate_K_s
            expected = (energy[name] - fsum(u[name][s] * dn[name][s] for s in ids)) / capacity[name]
            if not isclose(tdot[name], expected, abs_tol=1e-9, rel_tol=1e-10):
                raise RuntimeError(
                    "thermal spatial material ports differ from joint energy closure"
                )
        packet.update(
            pressure_rate_Pa_s=pressure * float(solution[-1]),
            maximum_mach=max((row["mach"] for row in faces.values()), default=0.0),
            temperature_rates_K_s=tdot,
            cell_total_rates_mol_s={name: fsum(dn[name].values()) for name in cells},
            closure_iterations=_iteration + 1,
            energy_closure_residual_W=residual[:m].tolist(),
            continuity_residual_mol_s=residual[m:].tolist(),
            common_pressure_residual_Pa={
                name: gas_node_pressure(self.nodes[name], total[name], temperatures[name])
                - pressure
                for name in cells
            },
        )
        return chemistry, dn, packet, heater, loss, links

    def process(self, state: CoupledStateView) -> ProcessContribution:
        chemistry, _, packet, heater, loss, links = self.evaluate(state)
        counters, transfers = {}, []
        for face, row in packet["faces"].items():
            left, right = row["source"], row["target"]
            for s, rate in row["amount_rates_mol_s"].items():
                source, target = (left, right) if rate >= 0 else (right, left)
                transfers.append(
                    InternalTransfer(
                        _key("spatial", face, s),
                        gas_inventory_key(source, s),
                        gas_inventory_key(target, s),
                        abs(rate),
                    )
                )
                counters[gas_transfer_key(face, s)] = rate
                counters[spatial_diffusion_key(face, s)] = row["diffusion_mol_s"][s]
            counters[gas_enthalpy_key(face)] = row["enthalpy_W"]
            counters[spatial_diffusion_heat_key(face)] = row["diffusion_enthalpy_W"]
            counters[spatial_fourier_key(face)] = row["fourier_heat_W"]
            counters[spatial_bulk_key(face)] = row["bulk_mol_s"]
        views = self.views(state)
        for name in views:
            if name != self.spec.anchor_vessel_id:
                counters[gas_temperature_key(name)] = packet["temperature_rates_K_s"][name]
            counters[gas_bath_key(name)] = heater[name]
            counters[gas_loss_key(name)] = loss[name]
            counters[gas_reaction_energy_key(name)] = self.models[name].chemical_energy_rate_w(
                views[name], chemistry[name].amount_rates_mol_s, (chemistry[name],)
            )
        counters.update({gas_link_heat_key(key): value for key, value in links.items()})
        return ProcessContribution(
            {
                gas_inventory_key(name, s): v
                for name, c in chemistry.items()
                for s, v in c.amount_rates_mol_s.items()
            },
            {
                _key("reaction", name, r): v
                for name, c in chemistry.items()
                for r, v in c.reaction_progress_mol_s.items()
            },
            tuple(transfers),
            coordinate_rates=counters,
        )

    def validate(self, state: CoupledStateView) -> None:
        self.validate_fit_temperature(state)
        if not isclose(state.volume_L, sum(n.volume_L for n in self.spec.nodes), rel_tol=1e-12):
            raise ValueError("thermal spatial mesh and tank volumes must remain fixed")
        if set(state.coordinates) != {
            c.name for c in gas_network_coordinates(self.spec, self.network)
        }:
            raise ValueError("thermal spatial accounting layout differs")
        for name, view in self.views(state).items():
            node, energy = self.nodes[name], self.energy[name]
            if fsum(view.amounts_mol.values()) < node.minimum_amount_mol - 1e-13:
                raise ValueError("thermal spatial node below positive inventory floor")
            if (
                not energy.minimum_temperature_K - 1e-7
                <= view.temperature_K
                <= energy.maximum_temperature_K + 1e-7
            ):
                raise ValueError("thermal spatial node outside declared temperature limits")
            self.models[name].validate_state(view)
            if gas_node_pressure(
                node, fsum(view.amounts_mol.values()), view.temperature_K
            ) > node.maximum_pressure_Pa * (1 + 1e-9):
                raise ValueError("thermal spatial node exceeds pressure ceiling")
        packet = self.evaluate(state)[2]
        if (
            max(abs(v) for v in packet["common_pressure_residual_Pa"].values())
            > 1e-5 + 2e-7 * packet["pressure_Pa"]
        ):
            raise ValueError("thermal spatial cells do not satisfy their common-pressure manifold")
        if packet["pressure_Pa"] > self.spatial.transport.maximum_pressure_Pa * (1 + 1e-9):
            raise ValueError("thermal spatial transport pressure ceiling exceeded")
        if packet["maximum_mach"] > self.spatial.maximum_mach * (1 + 1e-8):
            raise ValueError("thermal spatial low-Mach ceiling exceeded")

    def external_flux(self, state: CoupledStateView) -> HeatFlux:
        heater, loss, _ = self.heat(self.views(state))
        return HeatFlux(fsum(heater.values()), fsum(loss.values()))

    def events(self) -> tuple[CoupledEvent, ...]:
        def thermal_event(name: str, maximum: bool) -> CoupledEvent:
            def thermal_limit(state: CoupledStateView) -> float:
                t = gas_node_view(state, self.nodes[name], self.network).temperature_K
                e = self.energy[name]
                return e.maximum_temperature_K - t if maximum else t - e.minimum_temperature_K

            return CoupledEvent(
                f"gas:{name}:maximum_temperature" if maximum else f"gas:{name}:minimum_temperature",
                thermal_limit,
            )

        def pressure_limit(state: CoupledStateView) -> float:
            views = self.views(state)
            return min(
                (
                    min(n.maximum_pressure_Pa, self.spatial.transport.maximum_pressure_Pa)
                    if n.vessel_id in self.spatial.cell_vessel_ids
                    else n.maximum_pressure_Pa
                )
                - gas_node_pressure(
                    n,
                    fsum(views[n.vessel_id].amounts_mol.values()),
                    views[n.vessel_id].temperature_K,
                )
                for n in self.spec.nodes
            )

        def mach_limit(state: CoupledStateView) -> float:
            # Match the original RHS boundary extension only for event trials.
            # Accepted inventories are checked, never projected or replenished.
            trial = replace(
                state, amounts_mol={s: max(n, 0.0) for s, n in state.amounts_mol.items()}
            )
            if self.fit_temperature_domains:
                trial = self.extend_source_boundary_trial(trial)
            return self.spatial.maximum_mach - self.evaluate(trial)[2]["maximum_mach"]

        fit_events = ()
        if self.fit_temperature_domains:

            def fit_limit(state: CoupledStateView) -> float:
                low, high = self.spatial.transport.fit_temperature_range_K
                views = self.views(state)
                return min(
                    min(views[name].temperature_K - low, high - views[name].temperature_K)
                    for name in self.fit_temperature_domains
                )

            fit_events = (
                CoupledEvent("spatial:transport_fit_temperature", fit_limit, feasible_side=True),
            )
        return (
            *fit_events,
            CoupledEvent("spatial:maximum_pressure", pressure_limit),
            CoupledEvent("spatial:maximum_mach", mach_limit),
            CoupledEvent(
                "spatial:minimum_amount",
                lambda s: min(
                    fsum(v.amounts_mol.values()) - self.nodes[name].minimum_amount_mol
                    for name, v in self.views(s).items()
                ),
            ),
            *(thermal_event(name, maximum) for name in self.nodes for maximum in (False, True)),
        )


def gas_spatial_energy_system(
    spec: GasReactorNetworkSpec,
    network: ReactionNetworkSpec,
    *,
    thermal_device: ThermalDeviceSpec | None = None,
) -> CoupledSystem:
    model = _SpatialEnergyClosure(spec, network, thermal_device)
    return CoupledSystem(
        model.thermal,
        (CoupledProcess("thermal-spatial-gas", model.process, validate_state=model.validate),),
        heat_flux=model.external_flux,
        coordinates=gas_network_coordinates(spec, network),
        events=model.events(),
        policy=replace(DEFAULT_COUPLED_ODE_POLICY, rtol=1e-10),
        conservation_laws={
            e: {
                gas_inventory_key(n.vessel_id, s.species_id): s.composition.get(e, 0)
                for n in spec.nodes
                for s in network.species
            }
            for e in {e for s in network.species for e in s.composition}
        },
        **(
            {"trial_state_extension": model.extend_source_boundary_trial}
            if model.fit_temperature_domains
            else {}
        ),
    )


def spatial_energy_rates(
    spec: GasReactorNetworkSpec,
    network: ReactionNetworkSpec,
    state: CoupledStateView,
    *,
    thermal_device: ThermalDeviceSpec | None = None,
) -> dict[str, Any]:
    """Read the current joint closure without committing stock, time or resources."""
    model = _SpatialEnergyClosure(spec, network, thermal_device)
    packet = model.evaluate(state)[2]
    if model.fit_temperature_domains:
        packet["source_fit_temperature_domain"] = model.fit_domain_receipt()
    return packet


def advance_gas_spatial_energy(
    spec: GasReactorNetworkSpec,
    network: ReactionNetworkSpec,
    state: CoupledState,
    duration_s: float,
    *,
    thermal_device: ThermalDeviceSpec | None = None,
) -> tuple[CoupledAdvanceResult, dict[str, Any]]:
    result = gas_spatial_energy_system(spec, network, thermal_device=thermal_device).advance(
        state, duration_s
    )
    final = result.final_state
    delta = {k: final.coordinates[k] - state.coordinates[k] for k in state.coordinates}
    faces: dict[str, Any] = {
        face: {
            "source": left,
            "target": right,
            "amounts_mol": {s: delta[gas_transfer_key(face, s)] for s in network.species_ids},
            "enthalpy_J": delta[gas_enthalpy_key(face)],
            "diffusion_enthalpy_J": delta[spatial_diffusion_heat_key(face)],
            "fourier_heat_J": delta[spatial_fourier_key(face)],
            "bulk_mol": delta[spatial_bulk_key(face)],
            "diffusion_mol": {
                s: delta[spatial_diffusion_key(face, s)] for s in network.species_ids
            },
        }
        for face, left, right in gas_material_topology(spec)
    }
    links: dict[str, Any] = {
        link.connection_id: {
            "source": link.source_vessel_id,
            "target": link.target_vessel_id,
            "heat_J": delta[gas_link_heat_key(link.connection_id)],
        }
        for link in spec.heat_links
    }
    thermal = _NetworkCalorics(spec, network)
    details = {}
    for node in spec.nodes:
        name = node.vessel_id
        before, after = gas_node_view(state, node, network), gas_node_view(final, node, network)
        du = thermal.local_models[name].internal_energy_j(after) - thermal.local_models[
            name
        ].internal_energy_j(before)
        incoming = fsum(row["enthalpy_J"] for row in faces.values() if row["target"] == name)
        outgoing = fsum(row["enthalpy_J"] for row in faces.values() if row["source"] == name)
        fourier = fsum(
            ((1 if row["target"] == name else 0) - (1 if row["source"] == name else 0))
            * row["fourier_heat_J"]
            for row in faces.values()
        )
        conducted = fsum(
            ((1 if row["target"] == name else 0) - (1 if row["source"] == name else 0))
            * row["heat_J"]
            for row in links.values()
        )
        heater, loss = delta[gas_bath_key(name)], delta[gas_loss_key(name)]
        residual = du - heater + loss - incoming + outgoing - fourier - conducted
        if abs(residual) > 2e-6 + 5e-6 * sum(
            abs(x) for x in (du, heater, loss, incoming, outgoing, fourier, conducted)
        ):
            raise RuntimeError(f"thermal spatial local first law failed: {name}: {residual} J")
        details[name] = {
            "amounts_mol": dict(after.amounts_mol),
            "temperature_K": after.temperature_K,
            "pressure_Pa": gas_node_pressure(
                node, fsum(after.amounts_mol.values()), after.temperature_K
            ),
            "volume_L": after.volume_L,
            "bath_heat_J": heater,
            "heat_loss_J": loss,
            "enthalpy_in_J": incoming,
            "enthalpy_out_J": outgoing,
            "fourier_heat_J": fourier,
            "conducted_heat_J": conducted,
            "internal_energy_change_J": du,
            "chemical_energy_change_J": delta[gas_reaction_energy_key(name)],
            "energy_residual_J": residual,
        }
    heater, loss = (
        fsum(r["bath_heat_J"] for r in details.values()),
        fsum(r["heat_loss_J"] for r in details.values()),
    )
    if not isclose(
        heater, final.boundary_heat_J - state.boundary_heat_J, abs_tol=2e-7, rel_tol=2e-7
    ) or not isclose(loss, final.heat_loss_J - state.heat_loss_J, abs_tol=2e-7, rel_tol=2e-7):
        raise RuntimeError("thermal spatial receipts differ from original common energy ledger")
    spatial = spatial_energy_rates(spec, network, final, thermal_device=thermal_device)
    if "source_fit_temperature_domain" in spatial:
        receipt = spatial["source_fit_temperature_domain"]
        extension = result.solver_diagnostic.get("rhs_state_extension", {})
        coordinate_offsets = dict(extension.get("maximum_coordinate_offsets", {}))
        anchor_offset = extension.get("maximum_temperature_offset_K", 0.0)
        receipt["rhs_extension"] = {
            "count_scope": "shared ODE RHS; excludes Mach event callback evaluations",
            "extended_evaluations": extension.get("extended_evaluations", 0),
            "maximum_anchor_temperature_offset_K": anchor_offset,
            "maximum_coordinate_offsets": coordinate_offsets,
            "maximum_node_temperature_offset_K": max(
                anchor_offset, *coordinate_offsets.values(), 0.0
            ),
        }
        result = replace(
            result,
            solver_diagnostic={
                **result.solver_diagnostic,
                "source_fit_temperature_domain": receipt,
            },
        )
    return result, {
        "nodes": details,
        "connections": faces,
        "heat_links": links,
        "spatial": spatial,
        **(
            {"source_fit_temperature_domain": spatial["source_fit_temperature_domain"]}
            if "source_fit_temperature_domain" in spatial
            else {}
        ),
        "bath_heat_J": heater,
        "heat_loss_J": loss,
        "actual_duration_s": result.actual_duration_s,
        "stopping_event": result.stopping_event,
        "energy_residual_J": result.energy_balance_residual_J,
        "element_residuals_mol": dict(result.conservation_residuals),
        "energy_scope": (
            "finite fixed-volume gas/support U; joint low-Mach continuity/energy; "
            "upwind H, diffusion H and Fourier once"
        ),
    }
