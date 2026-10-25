"""Compile reaction SDE callbacks from one immutable chemical network and noise model."""

import json
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite, sqrt
from types import MappingProxyType

import numpy as np

from chemworld.foundation.dimensions import DimensionVector
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem.mathml_dimensions import mathml_dimension
from chemworld.physchem.reaction_rate_laws import mass_action_rate
from chemworld.physchem.stochastic_ri5 import MAX_RI5_NOISE_CHANNELS
from chemworld.physchem.stochastic_sde_controls import resolve_sde_controls
from chemworld.physchem.stochastic_sde_specs import ReactionNoise, StochasticSDESpec, finite_real
from chemworld.physchem.stoichiometric_rates import stoichiometric_amount_rates

_ONE = DimensionVector()
_T = DimensionVector(time=1)
_N = DimensionVector(amount=1)
_V = DimensionVector(length=3)


@dataclass(frozen=True)
class ReactionSDE:
    """Configuration and callbacks only. Evolving stock is always supplied by the caller."""

    network: object
    spec: StochasticSDESpec
    control_values: object = None
    volume_L: float | None = None
    noises: tuple = field(init=False)
    dimensions: object = field(init=False, repr=False, compare=False)
    signature: str = field(init=False)
    _stoichiometry: object = field(init=False, repr=False, compare=False)
    control_metadata: object = field(init=False, repr=False, compare=False)
    _rate_scales: object = field(init=False, repr=False, compare=False)
    _flux_parameters: object = field(init=False, repr=False, compare=False)
    _noise_parameters: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        from chemworld.physchem.reaction_network import ReactionNetworkSpec

        if not isinstance(self.network, ReactionNetworkSpec) or not isinstance(
            self.spec, StochasticSDESpec
        ):
            raise ValueError(
                "reaction SDE requires the original network and its explicit physical spec"
            )
        if self.network.stochastic is not None or not self.network.reactions:
            raise ValueError("SDE cannot compete with an integer SSA network or an empty model")
        volume = (
            self.spec.volume_L
            if self.volume_L is None
            else finite_real(self.volume_L, "SDE stage volume", positive=True)
        )
        minimum = (
            self.spec.volume_L if self.spec.minimum_volume_L is None else self.spec.minimum_volume_L
        )
        if not minimum <= volume <= self.spec.volume_L:
            raise ValueError("SDE geometry differs from its declared sampling interval")
        object.__setattr__(self, "volume_L", volume)
        if not isfinite(AVOGADRO * volume) or volume / 1000.0 == 0:
            raise ValueError("SDE geometry cannot be represented in native SI/population units")
        fluxes = {f.reaction_id: f for f in self.spec.fluxes}
        if fluxes.keys() - set(self.network.reaction_ids):
            raise ValueError("explicit SDE flux must bind an existing reaction channel")
        for reaction in self.network.reactions:
            if reaction.reaction_id in fluxes:
                continue
            orders = reaction.kinetic_forward_orders
            if (
                reaction.reversible
                or reaction.rate_law.equation_id != "mass_action"
                or reaction.rate_law.uses_activities
                or set(reaction.rate_law.parameters) != {"k"}
                or any(not float(v).is_integer() or v <= 0 for v in orders.values())
                or sum(orders.values()) > 3
                or any(not float(v).is_integer() for v in reaction.stoichiometry.values())
                or any(orders.get(key) != n for key, n in reaction.reactants.items())
                or any(
                    key in reaction.stoichiometry
                    for key in orders.keys() - reaction.reactants.keys()
                )
            ):
                raise ValueError("SDE requires elementary channels or explicit typed fluxes")
            coefficient = reaction.rate_law.parameters["k"]
            if isinstance(coefficient, bool) or not isfinite(coefficient) or coefficient < 0:
                raise ValueError("SDE rate constants must be finite and nonnegative")
        noises = (
            tuple(ReactionNoise(r.reaction_id) for r in self.network.reactions)
            if self.spec.noises is None
            else self.spec.noises
        )
        if len(noises) > MAX_RI5_NOISE_CHANNELS:
            raise ValueError("SDE exceeds the RI5 noise dimension budget")
        if any(n.reaction_id not in self.network.reaction_ids for n in noises):
            raise ValueError("SDE noise must own an existing stoichiometric reaction channel")
        current, metadata, rate_scales, flux_parameters, noise_parameters = resolve_sde_controls(
            self.network, self.spec, noises, self.control_values
        )
        object.__setattr__(self, "control_values", MappingProxyType(current))
        object.__setattr__(self, "control_metadata", MappingProxyType(metadata))
        object.__setattr__(self, "_rate_scales", rate_scales)
        object.__setattr__(self, "_flux_parameters", flux_parameters)
        object.__setattr__(self, "_noise_parameters", noise_parameters)
        dimensions = {
            "$time": _T,
            "time_s": _T,
            "volume_m3": _V,
            "avogadro": _ONE / _N,
        }
        for i in range(len(self.network.species)):
            dimensions.update({f"n{i}": _ONE, f"a{i}": _N, f"c{i}": _N / _V})
        noise_dimensions = {
            **dimensions,
            "flux": _ONE / _T,
            **{f"f{i}": _ONE / _T for i in range(len(self.network.reactions))},
        }
        for declaration in (*self.spec.fluxes, *noises):
            if {p.name for p in declaration.parameters} & noise_dimensions.keys():
                raise ValueError("SDE parameter shadows a physical quantity symbol")
            if declaration.expression is not None:
                is_flux = declaration in self.spec.fluxes
                parameters = {p.name: p.value for p in declaration.parameters}
                parameters.update(
                    (flux_parameters if is_flux else noise_parameters).get(
                        declaration.reaction_id, {}
                    )
                )
                shape = mathml_dimension(
                    declaration.expression,
                    {
                        **(dimensions if is_flux else noise_dimensions),
                        **{p.name: p.dimension for p in declaration.parameters},
                    },
                    constants=parameters,
                )
                expected = _T ** (-1 if is_flux else -0.5)
                if shape != expected:
                    raise ValueError(
                        "SDE flux/noise expression has incompatible physical dimensions"
                    )
        matrix = np.array(
            [
                [r.stoichiometry.get(key, 0.0) for r in self.network.reactions]
                for key in self.network.species_ids
            ],
            dtype=float,
        )
        matrix.flags.writeable = False
        object.__setattr__(self, "_stoichiometry", matrix)
        object.__setattr__(self, "noises", noises)
        object.__setattr__(self, "dimensions", MappingProxyType(noise_dimensions))
        object.__setattr__(
            self,
            "signature",
            sha256(
                json.dumps(
                    [self.network.to_dict(), self.spec.to_dict(), volume, current],
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        )

    @property
    def noise_dimension(self):
        return len(self.noises)

    def _view(self, values, time_s):
        values = np.asarray(values)
        if (
            values.dtype.kind not in "iuf"
            or values.shape != (len(self.network.species),)
            or not np.isfinite(values).all()
            or isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or not isfinite(time_s)
            or time_s < 0
        ):
            raise ValueError("SDE population view must be a finite vector of the declared species")
        values = values.astype(float)
        concentrations = {
            key: float(v) / (AVOGADRO * self.volume_L)
            for key, v in zip(self.network.species_ids, values, strict=True)
        }
        scope = {
            "$time": time_s,
            "time_s": time_s,
            "volume_m3": self.volume_L / 1000.0,
            "avogadro": AVOGADRO,
        }
        for i, value in enumerate(values):
            scope.update(
                {
                    f"n{i}": float(value),
                    f"a{i}": float(value) / AVOGADRO,
                    f"c{i}": float(value) / AVOGADRO / scope["volume_m3"],
                }
            )
        fluxes = {f.reaction_id: f for f in self.spec.fluxes}
        rates = {}
        for reaction in self.network.reactions:
            declaration = fluxes.get(reaction.reaction_id)
            rates[reaction.reaction_id] = (
                declaration.expression.evaluate(
                    {
                        **scope,
                        **{p.name: p.value for p in declaration.parameters},
                        **self._flux_parameters.get(reaction.reaction_id, {}),
                    }
                )
                if declaration is not None
                else (
                    mass_action_rate(
                        reaction.kinetic_forward_orders,
                        concentrations,
                        float(reaction.rate_law.parameters["k"]),
                        allow_signed_trial=True,
                    )
                    * AVOGADRO
                    * self.volume_L
                )
            ) * self._rate_scales.get(reaction.reaction_id, 1.0)
        if any(isinstance(value, bool) or not isfinite(value) for value in rates.values()):
            raise ArithmeticError("SDE particle flux overflow")
        scope.update({f"f{i}": rates[r.reaction_id] for i, r in enumerate(self.network.reactions)})
        return rates, scope

    def drift(self, time_s, values):
        rates, _ = self._view(values, time_s)
        result = stoichiometric_amount_rates(
            self.network.species_ids,
            tuple((r.reaction_id, r.stoichiometry) for r in self.network.reactions),
            rates,
        )
        return np.array([result[key] for key in self.network.species_ids])

    def diffusion(self, time_s, values):
        rates, scope = self._view(values, time_s)
        result = np.empty((len(self.network.species), self.noise_dimension))
        indices = {key: i for i, key in enumerate(self.network.reaction_ids)}
        for column, noise in enumerate(self.noises):
            amplitude = (
                sqrt(abs(rates[noise.reaction_id]))
                if noise.expression is None
                else noise.expression.evaluate(
                    {
                        **scope,
                        "flux": rates[noise.reaction_id],
                        **{p.name: p.value for p in noise.parameters},
                        **self._noise_parameters.get(noise.reaction_id, {}),
                    }
                )
            )
            if isinstance(amplitude, bool) or not isfinite(amplitude):
                raise ValueError("SDE noise amplitude must be a finite real scalar")
            result[:, column] = self._stoichiometry[:, indices[noise.reaction_id]] * amplitude
        return result
