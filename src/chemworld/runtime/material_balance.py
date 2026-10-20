"""Translate existing service executions into external boundary receipts.

This is the trusted adapter for the current native services, installed when the
runtime is constructed. A raw transaction has a closed boundary. New kernels
do not acquire permission to create material merely by reusing an action name.
"""

from __future__ import annotations

from math import fsum, isclose
from typing import TYPE_CHECKING, Any

from chemworld.foundation import WorldState, equipment_settings
from chemworld.foundation.balances import MaterialExchange, TransitionBalance
from chemworld.foundation.solvents import SOLVENT_IDS
from chemworld.runtime.full_process_contract import sample_domain
from chemworld.runtime.phase_ledger_services import action_float
from chemworld.runtime.solvent_transport import solvent_input, solvent_outputs
from chemworld.world.operations import MATERIAL_STOCK_FIELDS, instrument_name

if TYPE_CHECKING:
    from chemworld.runtime.domain_services import ChemWorldDomainServices


class ServiceMaterialBoundary:
    def __init__(self, services: ChemWorldDomainServices) -> None:
        self.services = services

    def resolve(
        self, before: WorldState, after: WorldState, action: dict[str, Any]
    ) -> TransitionBalance:
        operation = str(action["operation"])
        if before.lab_clock is not None and operation == "wait":
            return self.services.lab_schedule.boundary(before, after)
        exchanges: list[MaterialExchange] = []
        stocks: dict[str, float] = {}
        unmodeled: list[str] = []
        vessel = f"vessel:{before.vessel_id}"

        def exchange(
            source: str, destination: str, direction: str, amounts: dict[str, float], receipt: str
        ) -> None:
            exchanges.append(
                MaterialExchange(
                    receipt, source, destination, direction, tuple(sorted(amounts.items()))
                )
            )

        if operation in {"add_reagent", "add_component"}:
            amounts = self.services.primitive.feed_additions(before, action)
            router = self.services.material_routing
            if router.uses_native_packets(before):
                source_id = router.physics.feed_sources[action["component"]]
                old_source = before.inactive_vessels[source_id].species_amounts
                new_source = after.inactive_vessels[source_id].species_amounts
                for key in old_source.keys() | new_source.keys() | amounts.keys():
                    if not isclose(
                        old_source.get(key, 0) - new_source.get(key, 0),
                        amounts.get(key, 0),
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    ):
                        raise ValueError("finite source debit disagrees with executed feed")
                # Both vessels are inside the existing elemental boundary.
                unmodeled.append("fixed_temperature_feed_source_utility_heat_and_work")
            else:
                exchange("supply:reagent", vessel, "in", amounts, "feed")
            stocks["reagent_mol"] = fsum(amounts.values())
            # The charge counter is a consumer of the same actual additions,
            # not a source from which to infer an arbitrary balancing receipt.
            old = {} if before.species is None else before.species.initial_amounts_mol
            new = {} if after.species is None else after.species.initial_amounts_mol
            for key in old.keys() | new.keys() | amounts.keys():
                if not isclose(
                    new.get(key, 0) - old.get(key, 0),
                    amounts.get(key, 0),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    raise ValueError("reagent charge counter disagrees with executed feed")
        elif operation == "add_catalyst":
            amount = self.services.primitive.catalyst_charge(action)
            species = self.services.species_view.active_catalyst_species(before)
            if species is None:
                unmodeled.append("catalyst_element_inventory")
            else:
                exchange("supply:catalyst", vessel, "in", {species: amount}, "catalyst")
            stocks["catalyst_mol"] = amount
            self._counter(before, after, "batch_reactor", "catalyst_amount_mol", amount)
        elif operation == "seed_crystals":
            mass, amount = self.services.crystallization.seed_charge(action, before)
            exchange(
                "supply:seed",
                vessel,
                "in",
                {self.services.crystallization.target_species(before): amount},
                "seed",
            )
            stocks["seed_g"] = mass
            self._counter(before, after, "crystallizer", "crystal_seed_mass_g", mass)

        # These carriers have an explicit volume stock, not a registered
        # molecular composition. Keep their units instead of inventing moles.
        stock_contract = MATERIAL_STOCK_FIELDS.get(operation)
        added = solvent_input(operation, action, state=before, after=after)
        if operation == "add_solvent":
            from chemworld.runtime.condensed_material_routing import validate_carrier_input

            receipt = validate_carrier_input(
                before, after, action, self.services.material_routing.physics
            )
            if receipt is not None:
                stocks["solvent_L"] = receipt["volume_L"]
        liquid = self.services.reaction_thermal.liquid
        if liquid.supports(before):
            unmodeled.append("strict_isothermal_liquid_mixing_heat_and_PV_work")
            if operation == "wash":
                model = liquid.model(before)
                source_id = liquid.composition.physics.feed_sources[model.wash_feed_id]
                source, new_source = (
                    before.inactive_vessels[source_id],
                    after.inactive_vessels[source_id],
                )
                receipt = after.metadata["last_liquid_wash"]
                volume = receipt["wash_volume_L"]
                if (
                    receipt["source"] != source_id
                    or not isclose(volume, float(action["wash_volume_L"]), abs_tol=1e-12)
                    or not isclose(source.volume_L - new_source.volume_L, volume, abs_tol=1e-12)
                ):
                    raise ValueError("finite wash source debit differs from its actual volume")
                for key, amount in source.species_amounts.items():
                    expected = amount * volume / source.volume_L
                    if not isclose(
                        amount - new_source.species_amounts[key],
                        expected,
                        abs_tol=1e-12,
                        rel_tol=1e-10,
                    ) or not isclose(
                        receipt["amounts_mol"][key], expected, abs_tol=1e-12, rel_tol=1e-10
                    ):
                        raise ValueError("finite wash molecular debit disagrees with its receipt")
                stocks["wash_solvent_L"] = volume
        if stock_contract is not None and stock_contract[0] not in stocks:
            stocks[stock_contract[0]] = added.volume_L

        if operation in {"sample", "measure"}:
            if operation == "sample":
                volume = min(max(action_float(action, "sample_volume_L", 0.0001), 0), 0.002)
                destination = "sink:sample"
            else:
                instrument = instrument_name(
                    action.get("instrument", "hplc"), modules=self.services.modules
                )
                volume = self.services.constitution.instruments[instrument].sample_volume_L
                destination = f"sink:instrument:{instrument}"
            if not isclose(
                after.ledger.sample_consumed_L - before.ledger.sample_consumed_L,
                volume,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError("sample debit disagrees with executed withdrawal")
            if before.phases is None:
                raise ValueError("sampling requires phase-resolved source inventory")
            domain = sample_domain(before)
            available = sum(before.phases.phases[key].volume_L for key in domain)
            if volume < 0 or volume > available:
                raise ValueError("withdrawal exceeds source inventory")
            amounts: dict[str, float] = {}
            if volume:
                for key in domain:
                    for species, amount in before.phases.phases[key].species_amounts_mol.items():
                        amounts[species] = amounts.get(species, 0) + amount * volume / available
            sample_receipts = None
            if operation == "measure" and self.services.modules is not None:
                spec = self.services.modules.instrument_specs().get(instrument)
                if spec is not None and spec.sample_outlet is not None:
                    # The declared physical sampling law resolves the outlet
                    # from the pre-state and actual sample, never from a deficit.
                    sample_receipts = spec.sample_outlet(before, action, amounts)
            if sample_receipts is None:
                exchange(vessel, destination, "out", amounts, "sample")
            else:
                exchanges.extend(sample_receipts)

        old_history = before.metadata.get("removed_phase_inventory_history", [])
        new_history = after.metadata.get("removed_phase_inventory_history", [])
        # Selecting a vessel switches its local history, without crossing the
        # global material boundary. Routing itself must conserve globally.
        if before.vessel_id == after.vessel_id:
            if new_history[: len(old_history)] != old_history:
                raise ValueError("previous outlet history was rewritten")
            fresh = new_history[len(old_history) :]
            if fresh:
                if len(fresh) != 1 or fresh[0].get("operation") != operation:
                    raise ValueError("duplicate or mismatched outlet execution")
                native_concentration = self.services.reaction_thermal.concentration
                if native_concentration.supports(before):
                    from chemworld.runtime.concentration import validate_outlet_execution

                    validate_outlet_execution(
                        before, after, native_concentration.model(before), fresh[0]
                    )
                elif operation not in {"separate_phase", "wash", "dry", "concentrate", "transfer"}:
                    raise ValueError("no outlet producer is installed for this service")
                if operation == "dry" and "media_packet" in fresh[0]:
                    from chemworld.runtime.disposable_sorbent import (
                        MEDIA_QUANTITY,
                        validate_media_packet,
                    )

                    packet = fresh[0]["media_packet"]
                    mass = validate_media_packet(before, after, packet)
                    unmodeled.append(
                        "legacy_disposable_frame_element_inventory_and_adsorption_heat"
                    )
                    exchanges.append(
                        MaterialExchange(
                            "outlet:spent_sorbent:media",
                            f"equipment:{packet['source_equipment']}",
                            "sink:spent_sorbent",
                            "out",
                            ((MEDIA_QUANTITY, mass),),
                            "kg",
                            "entity",
                        )
                    )
                inventories = fresh[0].get("inventories")
                if not isinstance(inventories, dict):
                    raise ValueError("outlet execution requires explicit inventories")
                for key, item in inventories.items():
                    if not isinstance(item, dict) or not isinstance(
                        item.get("species_amounts_mol"), dict
                    ):
                        raise ValueError("outlet requires a registered-species inventory")
                    exchanges.append(
                        MaterialExchange(
                            f"outlet:{key}",
                            vessel,
                            f"sink:{key}",
                            "out",
                            tuple(sorted(item["species_amounts_mol"].items())),
                            item.get("unit"),
                        )
                    )
        if added.volume_L:
            exchanges.append(
                MaterialExchange(
                    "carrier:input",
                    "supply:carrier",
                    vessel,
                    "in",
                    tuple(zip(SOLVENT_IDS, added.volumes_L, strict=True)),
                    "L",
                    "carrier",
                )
            )
        for key, inventory in solvent_outputs(before, after, action).items():
            exchanges.append(
                MaterialExchange(
                    f"carrier:{key}",
                    vessel,
                    f"sink:{key}",
                    "out",
                    tuple(zip(SOLVENT_IDS, inventory.volumes_L, strict=True)),
                    "L",
                    "carrier",
                )
            )
        return TransitionBalance(tuple(exchanges), tuple(sorted(stocks.items())), tuple(unmodeled))

    @staticmethod
    def _counter(
        before: WorldState, after: WorldState, equipment: str, key: str, expected: float
    ) -> None:
        old = float(equipment_settings(before.equipment, equipment).get(key, 0))
        new = float(equipment_settings(after.equipment, equipment).get(key, 0))
        if not isclose(new - old, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("stock debit disagrees with executed material input")
