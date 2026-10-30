"""Native MQMQA phase contributions on the common Gibbs/material interface."""

from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType

from chemworld.data.chem_sage import ChemSageDatabase
from chemworld.physchem.gibbs_phase import GibbsPhaseModel
from chemworld.physchem.mqmqa_energy import energies
from chemworld.physchem.mqmqa_topology import compile_topology
from chemworld.physchem.tdb_expressions import scalar_value


@dataclass(frozen=True)
class _Curve:
    source: object


@dataclass(frozen=True)
class MQMQAPhaseModel(GibbsPhaseModel):
    database: ChemSageDatabase
    phase_name: str
    components: tuple | None = None
    extrapolation: str = "reject"
    activate_suppressed: bool = False
    _phase: object = field(init=False, repr=False)
    _parameters: tuple = field(init=False, repr=False)
    _source: object = field(init=False, repr=False)
    _topology: object = field(init=False, repr=False)
    _diagnostics: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.database, ChemSageDatabase):
            raise ValueError("MQMQA requires original validated ChemSage input")
        if (not isinstance(self.phase_name, str)
                or self.phase_name.upper() not in self.database.phases):
            raise ValueError("unknown MQMQA phase")
        if self.extrapolation not in {"reject", "extend"}:
            raise ValueError("MQMQA extrapolation must be reject or extend")
        if type(self.activate_suppressed) is not bool:
            raise ValueError("MQMQA source activation must be an explicit boolean")
        name = self.phase_name.upper()
        phase = self.database.phases[name]
        if phase.model_hints.get("unsupported_physics"):
            raise ValueError("unconsumed MQMQA physical declarations: "+"; ".join(
                phase.model_hints["unsupported_physics"]))
        if phase.model_hints.get("chemical_group_overrides"):
            raise ValueError("MQMQA group overrides require their explicitly supported model")
        if not self.activate_suppressed and (phase.model_hints.get("initial_status") != "entered"
                                            or phase.model_hints.get("suppressed_endmembers")):
            raise ValueError("MQMQA source suppression requires explicit activation")
        eligible = set(self.database.species)
        if self.components is not None:
            if (not isinstance(self.components, (list, tuple))
                    or any(not isinstance(c, str) for c in self.components)):
                raise ValueError("MQMQA components must name declared elements")
            components = tuple(c.upper() for c in self.components)
            if not components or set(components) - self.database.species.keys():
                raise ValueError("MQMQA components must be declared species or elements")
            elements = {e for c in components for e, _ in self.database.species[c].constituents}
            eligible = {s.name for s in self.database.species.values()
                        if {e for e, _ in s.constituents} <= elements}
            object.__setattr__(self, "components", components)
        topology = compile_topology(self.database, name).restrict(eligible)
        source = self.database.phase_records[name]
        non_atom_indices = [i for i, e in enumerate(self.database.declarations.header.components)
                            if e not in self.database.elements and e != "VA"]
        if any(m.stoichiometry[i] for m in source.endmembers for i in non_atom_indices):
            raise ValueError("MQMQA nonzero source auxiliary balances need an explicit consumer")
        parameters = tuple(_Curve(p) for p in self.database.parameters if p.phase == name)
        if any(p.source.kind not in {"MQMG", "MQMX"} for p in parameters):
            raise ValueError("MQMQA has an unconsumed contribution type")
        nc, na = len(source.mqmqa.cations), len(source.mqmqa.anions)
        for item in source.interactions:
            if item.mixing_type not in {3, 4}:
                raise ValueError("MQMQA source interaction type must be 3 or 4")
            if (len(item.indices) != 4 or len(item.exponents) != 4
                    or any(not 1 <= i <= nc for i in item.indices[:2])
                    or any(not nc < i <= nc+na for i in item.indices[2:])):
                raise ValueError("MQMQA interaction needs four valid source ion indices")
            a, b, x, y = item.indices
            if any(p < 0 for p in item.exponents):
                raise ValueError("MQMQA exponents must be nonnegative")
            if a != b and x != y:
                if (item.mixing_code != "R" or not any(item.exponents)
                        or not 1 <= item.additional_constituent <= nc
                        or not nc < item.additional_anion_constituent <= nc+na):
                    raise ValueError("MQMQA reciprocal R term needs its cation/anion and exponent")
            elif item.mixing_code not in {"G", "Q"}:
                raise ValueError("MQMQA binary/ternary mixing code must be G or Q")
            if (a != b and x == y and (item.additional_anion_constituent
                    or (not item.additional_constituent and item.exponents[2])
                    or item.exponents[3])):
                raise ValueError("MQMQA cation mixing has an unconsumed anion/additional exponent")
            if a == b and x != y and (item.additional_constituent or item.exponents[2]
                    or (not item.additional_anion_constituent and item.exponents[3])):
                raise ValueError("MQMQA anion mixing has an unconsumed cation/additional exponent")
            if (a != b and x == y and item.additional_constituent
                    and (not 1 <= item.additional_constituent <= nc or item.exponents[2] < 1)):
                raise ValueError("MQMQA ternary cation needs its index and positive exponent")
            if (a == b and x != y and item.additional_anion_constituent
                    and (not nc < item.additional_anion_constituent <= nc+na
                         or item.exponents[3] < 1)):
                raise ValueError("MQMQA ternary anion needs its index and positive exponent")
        active = set(topology.ions)

        def is_active(parameter):
            if parameter.source.kind == "MQMG":
                c, a = source.mqmqa.pair_indices[parameter.source.source_index]
                return {c-1, nc+a-1} <= active
            item = source.interactions[parameter.source.source_index]
            indices = (*item.indices, item.additional_constituent,
                       item.additional_anion_constituent)
            return {i-1 for i in indices if i} <= active

        parameters = tuple(p for p in parameters if is_active(p))
        diagnostics = (
            "quadruplet fractions represent one mole of quadruplets; G is per real atom",
            "legacy twelve-field annotations retained; source parser defines no energy effect",
        )
        object.__setattr__(self, "phase_name", name)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_topology", topology)
        object.__setattr__(self, "_parameters", parameters)
        object.__setattr__(self, "_phase", replace(phase, site_ratios=(1.,),
                                                 constituents=(topology.labels,)))
        object.__setattr__(self, "_diagnostics", diagnostics)

    @property
    def constituents(self):
        return self._phase.constituents

    @property
    def constraint_labels(self):
        return "site_sum:0", "charge"

    def _composition(self, fractions):
        amounts = {}
        for i, n in self._topology.ion_amounts(fractions[0]).items():
            for e, count in self._topology.ions[i].constituents:
                if e not in {"VA", "/-"}:
                    amounts[e] = amounts.get(e, 0.)+n*count
        total = sum(amounts.values())
        if scalar_value(total) <= 0 or not isfinite(scalar_value(total)):
            raise ValueError("MQMQA real atom normalization must be positive and finite")
        return total, {e: value/total for e, value in amounts.items()}

    def _constraints(self, fractions):
        charge = sum(self._topology.ions[i].charge*n
                     for i, n in self._topology.ion_amounts(fractions[0]).items())
        return sum(fractions[0].values())-1, charge

    def _energies(self, fractions, temperature, pressure):
        return energies(self, fractions, temperature, pressure)

    def to_dict(self):
        return {"model_type": "calphad_mqmqa_v1", "database": self.database.to_dict(),
                "phase_name": self.phase_name, "components": self.components,
                "extrapolation": self.extrapolation,
                "activate_suppressed": self.activate_suppressed}

    @classmethod
    def from_dict(cls, payload):
        from chemworld.data.scientific_inputs import _keys

        _keys(payload, ("model_type", "database", "phase_name", "components", "extrapolation",
                        "activate_suppressed"))
        if payload["model_type"] != "calphad_mqmqa_v1":
            raise ValueError("unknown MQMQA model schema")
        return cls(ChemSageDatabase.from_dict(payload["database"]), payload["phase_name"],
                   payload["components"], payload["extrapolation"], payload["activate_suppressed"])

    def ion_amounts_per_quadruplet(self, fractions):
        fractions = self._fractions(fractions)
        return MappingProxyType({self._topology.ions[i].name: scalar_value(n)
            for i, n in self._topology.ion_amounts(fractions[0]).items()})
