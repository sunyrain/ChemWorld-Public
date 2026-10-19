"""Real finite-reservoir continuous flow through declared component ports."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any

import numpy as np

from chemworld.foundation import (
    WorldState,
    equipment_settings,
    process_with_metrics,
    upsert_equipment_record,
)
from chemworld.foundation.solvents import HEAT_CAPACITY_RATIOS, SolventInventory
from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.physchem.coupled_flow import FLOW_MODEL, integrate_flow
from chemworld.physchem.reaction_network import evaluate_rate_law
from chemworld.runtime.material_routing import (
    _active,
    _contents,
    _install,
    _merge,
    _occupied,
    _portion,
    _reference_amounts,
)
from chemworld.world.reaction_kernel import _reaction_effect_tables

STREAM_ID = "continuous_streams"


def settings(state: WorldState) -> dict[str, Any]:
    return equipment_settings(state.equipment, STREAM_ID)


def _edges(state: WorldState, program: dict[str, Any]) -> list[tuple[dict[str, Any], float]]:
    network = state.metadata.get("component_network")
    if not network:
        raise ValueError("Continuous streams require declared vessel connections")
    result = []
    for index, rate in sorted(program.get("rates", {}).items(), key=lambda item: int(item[0])):
        edge = network["connections"][int(index)]
        if state.vessel_id not in (edge["source_vessel"], edge["target_vessel"]):
            raise ValueError("Stream must connect the selected reactor")
        if edge["source"]["port"] != "out":
            raise ValueError("Continuous streams require the whole single-liquid outlet")
        if rate > 0:
            result.append((edge, rate / 60000))
    return result


def _liquid(state: WorldState) -> None:
    if state.phases is None or set(state.phases.phases) != {"reactor_liquid"} or state.volume_L < 0:
        raise ValueError("Continuous streams support one homogeneous liquid phase")


def stream_error(state: WorldState, operation: str, action: dict[str, Any]) -> str | None:
    try:
        if state.terminated:
            raise ValueError("Continuous streams require an open episode")
        program = settings(state)
        if operation == "set_flow_stream":
            index = action.get("connection")
            edges = state.metadata.get("component_network", {}).get("connections", ())
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(edges):
                raise ValueError("Unknown stream connection")
            rate = action.get("flow_rate_mL_min")
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not isfinite(rate)
                or not 0 <= rate <= 20
            ):
                raise ValueError("Stream rate outside [0, 20] mL/min; zero closes the valve")
            _edges(state, {"rates": {**program.get("rates", {}), str(index): rate}})
            return None
        for key, low, high in (
            ("duration_s", 1, 14400),
            ("target_temperature_K", 250, 430),
            ("current_mA", 0, 500),
        ):
            value = action.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"Invalid stream field: {key}")
        if action["current_mA"] > 0 and "electrolyze" not in state.metadata.get(
            "component_network", {}
        ).get("operations", {}).get(state.vessel_id, ()):
            raise ValueError(
                "Current requires an electrochemistry component in the selected vessel"
            )
        _liquid(state)
        bounds = state.metadata["component_network"].get("bounds", {}).get(state.vessel_id, {})
        low, high = bounds.get("heat:target_temperature_K", (250, 430))
        if not low <= action["target_temperature_K"] <= high:
            raise ValueError("Flow jacket target outside the authored thermal domain")
        if state.quenched and action["current_mA"] > 0:
            raise ValueError("Quenched material cannot restart an electrochemical reaction")
        if state.volume_L <= 1e-6:
            raise ValueError("Fill the tank before continuous advancement")
        edges = _edges(state, program)
        inputs = [(e, q) for e, q in edges if e["target_vessel"] == state.vessel_id]
        outputs = [(e, q) for e, q in edges if e["source_vessel"] == state.vessel_id]
        if len(inputs) > 4 or len(outputs) > 1:
            raise ValueError("At most four inlet streams and one collected outlet")
        sources: dict[str, float] = {}
        for edge, rate in inputs:
            name = edge["source_vessel"]
            sources[name] = sources.get(name, 0) + rate
        mixed = _active(state)
        for name, rate in sources.items():
            source = state.inactive_vessels[name]
            _liquid(source)
            if source.quenched != state.quenched:
                raise ValueError("Flow mixing requires matching reaction stop state")
            if rate * action["duration_s"] > source.volume_L + 1e-12:
                raise ValueError("Finite feed reservoir exhausted; shorten the requested interval")
            mixed = _merge(
                mixed,
                _portion(
                    _active(source),
                    min(1, rate * action["duration_s"] / source.volume_L),
                    state.vessel_id,
                ),
            )
        if outputs:
            edge, rate = outputs[0]
            name = edge["target_vessel"]
            if name in sources:
                raise ValueError(
                    "Recirculation through the same held reservoir is outside this model"
                )
            target = state.inactive_vessels[name]
            _liquid(target)
            assert target.vessels is not None
            if (
                target.volume_L + rate * action["duration_s"]
                > target.vessels.vessels[name].max_volume_L + 1e-12
            ):
                raise ValueError("Collector capacity exceeded")
            if _occupied(_active(target)):
                if target.quenched != state.quenched:
                    raise ValueError("Collected flow requires matching reaction stop state")
                _merge(_active(target), _active(state))
        final_volume = (
            state.volume_L
            + (sum(q for _, q in inputs) - sum(q for _, q in outputs)) * action["duration_s"]
        )
        assert state.vessels is not None
        if not 1e-6 < final_volume <= state.vessels.vessels[state.vessel_id].max_volume_L + 1e-12:
            raise ValueError("Tank volume would leave the declared capacity domain")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        return str(exc)
    return None


def _install_sample(state: WorldState, sample: Any) -> WorldState:
    contents = _contents(sample)
    return _install(state, sample).replace(
        samples=replace(
            state.samples,
            active_lineage=sample.lineage,
            active_reference_sources=contents.reference_sources,
            active_reference_shares=contents.reference_shares,
        )
    )


class ContinuousStreamServices:
    def __init__(self, world: Any, species_view: Any) -> None:
        self.world, self.species_view = world, species_view

    def apply(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        error = stream_error(state, action["operation"], action)
        if error:
            raise ValueError(error)
        program = settings(state)
        if action["operation"] == "set_flow_stream":
            return state.replace(
                equipment=upsert_equipment_record(
                    state.equipment,
                    equipment_id=STREAM_ID,
                    equipment_type="finite_reservoir_flow",
                    attached_vessel_id=state.vessel_id,
                    status="configured",
                    settings={
                        **program,
                        "rates": {
                            **program.get("rates", {}),
                            str(action["connection"]): action["flow_rate_mL_min"],
                        },
                    },
                )
            )
        duration = action["duration_s"]
        edges = _edges(state, program)
        inputs = [(e, q) for e, q in edges if e["target_vessel"] == state.vessel_id]
        outputs = [(e, q) for e, q in edges if e["source_vessel"] == state.vessel_id]
        original = _active(state)
        feeds = [(e, q, _active(state.inactive_vessels[e["source_vessel"]])) for e, q in inputs]
        sources = dict(_contents(original).reference_sources)
        for _, _, feed in feeds:
            sources.update(_contents(feed).reference_sources)
        refs = sorted(sources)
        network = self.species_view.mechanism.network
        ids = network.species_ids
        count = len(ids)

        def vector(sample: Any) -> np.ndarray:
            content = _contents(sample)
            solvents = content.phases.phases["reactor_liquid"].solvents
            return np.asarray(
                [
                    *(sample.species_amounts_mol.get(s, 0) for s in ids),
                    *solvents.volumes_L,
                    *(content.reference_shares.get(k, 0) for k in refs),
                ]
            )

        inventory = vector(original)
        rates = np.zeros_like(inventory)
        inlet_heat = 0.0
        cp = np.asarray(HEAT_CAPACITY_RATIOS) * self.world.rho_cp_J_per_L_K
        for _, rate, feed in feeds:
            flow = vector(feed) * rate / feed.volume_L
            rates += flow
            inlet_heat += float(np.dot(flow[count : count + 4], cp)) * (feed.temperature_K - 298.15)
        catalyst_effects, solvent_effects = _reaction_effect_tables(self.world, state)
        reactor = equipment_settings(state.equipment, "batch_reactor")
        catalyst_index = int(reactor.get("catalyst", 0))
        # Mixing already checks catalyst identity. Carry the inlet identity into the tank.
        for _, _, feed in feeds:
            feed_settings = equipment_settings(_contents(feed).equipment, "batch_reactor")
            if feed_settings.get("catalyst_amount_mol", 0) > 0:
                catalyst_index = int(feed_settings["catalyst"])
        assert state.species is not None
        catalyst_ids = [
            i for i, s in enumerate(ids) if "catalyst" in state.species.species_roles.get(s, ())
        ]
        stirring = 0.70 + 0.30 * (1 - np.exp(-600 / 420))

        def reaction(
            amounts: np.ndarray, solvents: np.ndarray, temperature: float
        ) -> tuple[np.ndarray, float]:
            derivative, heat = np.zeros(count), 0.0
            if state.quenched:
                return derivative, heat
            volume = float(solvents.sum())
            concentrations = dict(zip(ids, amounts / volume, strict=True))
            for index, item in enumerate(network.reactions):
                if item.rate_law.equation_id == "runtime_owned":
                    continue
                column = min(index, solvent_effects.shape[-1] - 1)
                solvent_factor = float(
                    np.exp(np.dot(solvents / volume, np.log(solvent_effects[:, column])))
                )
                catalyst_factor = (
                    catalyst_effects[catalyst_index, min(index, catalyst_effects.shape[-1] - 1)]
                    if any(amounts[i] > 0 for i in catalyst_ids)
                    else 1.0
                )
                rate = (
                    evaluate_rate_law(
                        item, concentrations_mol_L=concentrations, temperature_K=temperature
                    )
                    * volume
                    * solvent_factor
                    * catalyst_factor
                    * stirring
                )
                for species, coefficient in item.stoichiometry.items():
                    derivative[ids.index(species)] += rate * coefficient
                heat += rate * item.delta_h_J_per_mol
            return derivative, heat

        electrode = (
            self.species_view.reactant_species(state),
            self.species_view.primary_target_species,
            self.species_view.primary_impurity_species,
        )
        if action["current_mA"] > 0:
            compositions = [network.species[ids.index(s)].composition for s in electrode]
            if any(c != compositions[0] for c in compositions):
                raise ValueError(
                    "The finite electrode channel requires isoelemental lumped species"
                )
        result = integrate_flow(
            inventory=inventory,
            inlet_rates=rates,
            species_count=count,
            inlet_heat_W=inlet_heat,
            temperature_K=state.temperature_K,
            duration_s=duration,
            outlet_rate_L_s=sum(q for _, q in outputs),
            inlet_pipes=tuple(
                (q, tuple(vector(feed)[count : count + 4] / feed.volume_L)) for _, q, feed in feeds
            ),
            rho_cp_J_L_K=self.world.rho_cp_J_per_L_K,
            reaction=reaction,
            target_temperature_K=action["target_temperature_K"],
            ua_W_K=self.world.ua_W_per_K,
            environment_temperature_K=self.world.environment_temperature_K,
            current_mA=action["current_mA"],
            electrode_indices=tuple(ids.index(s) for s in electrode),
        )
        lineage = tuple(
            dict.fromkeys(
                (*original.lineage, *(name for _, _, feed in feeds for name in feed.lineage))
            )
        )

        def material(values: np.ndarray, temperature: float) -> Any:
            solvents = SolventInventory(tuple(values[count : count + 4]))
            amounts = dict(zip(ids, map(float, values[:count]), strict=True))
            shares = dict(zip(refs, map(float, values[count + 4 :]), strict=True))
            content = _contents(original)
            phase = replace(
                content.phases.phases["reactor_liquid"],
                volume_L=solvents.volume_L,
                species_amounts_mol=amounts,
                solvents=solvents,
            )
            contents = replace(
                content,
                phases=PhaseLedger({"reactor_liquid": phase}),
                species=replace(
                    content.species, initial_amounts_mol=_reference_amounts(sources, shares)
                ),
                reference_sources=sources,
                reference_shares=shares,
                metrics={},
            )
            equipment = upsert_equipment_record(
                contents.equipment,
                equipment_id="batch_reactor",
                equipment_type="batch_reactor",
                attached_vessel_id=state.vessel_id,
                settings={
                    **reactor,
                    "catalyst": catalyst_index,
                    "catalyst_amount_mol": sum(values[i] for i in catalyst_ids),
                    "solvent_volume_L": solvents.volume_L,
                },
            )
            return replace(
                original,
                species_amounts_mol=amounts,
                volume_L=solvents.volume_L,
                temperature_K=temperature,
                lineage=lineage,
                contents=replace(contents, equipment=equipment),
            )

        after = _install_sample(state, material(result.tank, result.temperature_K))
        inactive = dict(state.inactive_vessels)
        for edge, rate, _feed in feeds:
            name = edge["source_vessel"]
            fraction = rate * duration / inactive[name].volume_L
            inactive[name] = _install_sample(
                inactive[name], _portion(_active(inactive[name]), max(0, 1 - fraction), name)
            )
        if outputs:
            name = outputs[0][0]["target_vessel"]
            collected = _merge(
                _active(inactive[name]), material(result.outlet, result.outlet_temperature_K)
            )
            inactive[name] = _install_sample(inactive[name], collected)

        def measured_process(local: WorldState) -> WorldState:
            assert local.species is not None
            reactant = self.species_view.reactant_species(local)
            reference = local.species.initial_amounts_mol.get(reactant, 0)
            conversion = (
                float(np.clip(1 - local.species_amounts.get(reactant, 0) / reference, 0, 1))
                if reference > 0
                else 0.0
            )
            metrics = {"flow_conversion": conversion}
            if action["current_mA"] > 0:
                product, impurity = electrode[1:]
                produced = max(
                    0,
                    local.species_amounts.get(product, 0)
                    - local.species.initial_amounts_mol.get(product, 0),
                )
                byproduct = max(
                    0,
                    local.species_amounts.get(impurity, 0)
                    - local.species.initial_amounts_mol.get(impurity, 0),
                )
                metrics.update(
                    electrochemical_conversion=conversion,
                    selective_product_yield=produced / reference if reference > 0 else 0,
                    electrochemical_selectivity=produced / (produced + byproduct)
                    if produced + byproduct > 0
                    else 0,
                )
            return local.replace(process=process_with_metrics(local.process, **metrics))

        after = measured_process(after)
        if outputs:
            name = outputs[0][0]["target_vessel"]
            inactive[name] = measured_process(inactive[name])
        difference = (
            inventory[:count]
            + rates[:count] * duration
            - result.tank[:count]
            - result.outlet[:count]
        )
        elements = sorted(
            {element for species in network.species for element in species.composition}
        )
        balance = {
            element: float(
                sum(
                    difference[i] * species.composition.get(element, 0)
                    for i, species in enumerate(network.species)
                )
            )
            for element in elements
        }
        if any(abs(value) > 1e-8 for value in balance.values()):
            raise ValueError("Flow element balance failed")
        ledger = result.ledger
        step = {
            "duration_s": duration,
            "inlet_volume_L": sum(q for _, q in inputs) * duration,
            "outlet_volume_L": sum(q for _, q in outputs) * duration,
            "residence_time_s": state.volume_L / sum(q for _, q in inputs) if inputs else None,
            "residence_definition": (
                "initial tank volume / total inlet flow; constant only for equal inlet/outlet"
            ),
            "ledger": ledger,
            "diagnostics": result.diagnostics,
            "element_residuals_mol": balance,
        }
        totals = {k: program.get("totals", {}).get(k, 0) + v for k, v in ledger.items()}
        after = after.replace(
            inactive_vessels=inactive,
            equipment=upsert_equipment_record(
                after.equipment,
                equipment_id=STREAM_ID,
                equipment_type="finite_reservoir_flow",
                attached_vessel_id=state.vessel_id,
                status="configured",
                settings={**program, "last_step": step, "totals": totals},
            ),
            ledger=state.ledger.with_updates(
                time_s=state.ledger.time_s + duration,
                cost=state.ledger.cost
                + duration * 0.03 / 3600
                + (ledger["electrical_work_J"] + ledger["pump_work_J"]) / 250000,
                energy_jacket_J=state.ledger.energy_jacket_J
                + ledger["jacket_J"]
                + ledger["cell_heat_J"],
                heat_reaction_J=state.ledger.heat_reaction_J + ledger["reaction_J"],
                heat_loss_J=state.ledger.heat_loss_J + ledger["loss_J"],
                flow_pump_work_J=state.ledger.flow_pump_work_J + ledger["pump_work_J"],
                flow_electrical_work_J=state.ledger.flow_electrical_work_J
                + ledger["electrical_work_J"],
            ),
        )
        return after


def public_streams(state: WorldState) -> dict[str, Any] | None:
    program = settings(state)
    if not program:
        return None
    return {
        "model": FLOW_MODEL,
        "rates_mL_min": program["rates"],
        "last_step": program.get("last_step"),
        "totals": program.get("totals", {}),
    }
