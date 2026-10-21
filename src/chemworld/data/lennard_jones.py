"""Both complete external LJ tables using the original source/identity/cache contract."""

from dataclasses import dataclass, fields

from chemworld.data.constant_catalog import ConstantField, ConstantSource
from chemworld.data.constants import ConstantDatabase, ConstantRecord
from chemworld.physchem.lennard_jones import LennardJonesSpec

LENNARD_JONES_SOURCES = (
    ConstantSource("MAGALHAES_LJ", "Viscosity/MagalhaesLJ.tsv",
                   citation="Magalhaes et al., LJ potential parameters; chemicals 1.5.2"),
    ConstantSource("POLING_LJ", "Viscosity/PolingLJ.tsv",
                   citation="Poling et al., The Properties of Gases and Liquids; chemicals 1.5.2"),
)
_SOURCES = {source.source_id: source for source in LENNARD_JONES_SOURCES}


@dataclass(frozen=True)
class LennardJonesRecord(ConstantRecord):
    @property
    def source(self):
        return _SOURCES[self.source_id]

    def descriptor(self, field_id):
        self.raw(field_id)
        return ConstantField(
            field_id, {"molecular_diameter": "angstrom", "Stockmayer": "K"}.get(field_id),
            "Lennard-Jones sigma in angstrom; Stockmayer means epsilon/k in K, not energy in J",
            "22E1",
        )


class LennardJonesDatabase(ConstantDatabase):
    @property
    def catalog(self):
        return LENNARD_JONES_SOURCES

    def source(self, source_id):
        return _SOURCES[source_id] if source_id in _SOURCES else super().source(source_id)

    def _read_records(self, source, path):
        records = super()._read_records(source, path)
        if source.source_id not in _SOURCES:
            return records
        return tuple(
            LennardJonesRecord(**{f.name: getattr(r, f.name) for f in fields(ConstantRecord)})
            for r in records
        )

    def parameters(self, identity, component, *, energy_method, diameter_method):
        """Select each parameter's source explicitly; both original records are retained."""
        energy = self.value(identity, "Stockmayer", method=energy_method)
        diameter = self.value(identity, "molecular_diameter", method=diameter_method)
        for value in (energy, diameter):
            original_identity = value.record.cas_number or value.record.source_key
            if original_identity not in {component.cas_number, component.identifier}:
                raise ValueError("Lennard-Jones source identity does not match ComponentSpec")
        return LennardJonesSpec(
            component.identifier, energy.quantity("K").value, diameter.quantity("m").value,
            "Explicit source-selected LJ parameters; source uncertainty not quantified",
            {"epsilon_k": energy.to_dict(), "sigma": diameter.to_dict()},
        )

    def viscosity_input(self, identity, component, *, energy_method, diameter_method,
                        collision_method, temperature_range_K):
        """Source records -> existing input card -> existing gas property package."""
        from chemworld.data.scientific_inputs import InputSource
        from chemworld.physchem.property_inputs import PropertyInputCard
        from chemworld.physchem.property_packages import ComponentPropertyPackage
        from chemworld.physchem.specs import ComponentUncertainty

        parameters = self.parameters(identity, component, energy_method=energy_method,
                                     diameter_method=diameter_method)
        curve = parameters.viscosity_correlation(
            component, collision_method=collision_method, temperature_range_K=temperature_range_K,
        )
        sources = tuple(InputSource(self.source(method).provenance, "literature")
                        for method in dict.fromkeys((energy_method, diameter_method)))
        return PropertyInputCard(
            ComponentPropertyPackage(component, (curve,)), sources,
            {curve.correlation_id: ComponentUncertainty(
                curve.correlation_id, unit="Pa*s", source_id=sources[0].provenance.source_id,
                coverage="unspecified", note="LJ and dilute gas model uncertainties unknown",
            )}, None, "Gas viscosity only; no caloric reference is inferred",
        )
