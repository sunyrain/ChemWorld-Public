"""Construct existing property declarations for explicitly selected volume fits."""

from math import isfinite

from chemworld.physchem.correlation_families import CORRELATION_FAMILIES
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.volume_kernels import VOLUME_METHODS, volume_value


def volume_correlation(
    method,
    *,
    parameters,
    temperature_range_K,
    source_note,
    output="molar_volume",
    correlation_id=None,
    pressure_range_Pa=None,
):
    """No inferred validity, MW, phase or parameter source.

    Density is kg/m³, molar volume m³/mol. Use PropertyInputCard to attach
    InputSource and uncertainty metadata. Saturation slopes describe a path
    along saturation, not an isobaric response or a compressed-liquid EOS.
    """
    if method not in VOLUME_METHODS or output not in {"molar_volume", "density"}:
        raise ValueError("unknown pure volume method or output basis")
    if not isinstance(source_note, str) or not source_note.strip():
        raise ValueError("volume correlation requires an explicit source note")
    spec = VOLUME_METHODS[method]
    ranges = {"temperature": tuple(temperature_range_K)}
    if spec.uses_pressure:
        if pressure_range_Pa is None:
            raise ValueError("pressure-dependent volume requires an explicit pressure range")
        ranges["pressure"] = tuple(pressure_range_Pa)
    elif pressure_range_Pa is not None:
        raise ValueError("this pure correlation does not model independent pressure")
    for bounds in ranges.values():
        if (
            len(bounds) != 2
            or any(not isfinite(v) or v <= 0 for v in bounds)
            or bounds[0] >= bounds[1]
        ):
            raise ValueError("application ranges require ordered positive finite bounds")
    equation = "volume_" + method.lower() + "_" + output
    family = CORRELATION_FAMILIES[equation]
    parameters = dict(parameters)
    for selector in ("has_hydroxyl", "hydroxyl_polar_term", "estimate_saturation"):
        if isinstance(parameters.get(selector), bool):
            parameters[selector] = int(parameters[selector])
    if method == "Campbell_Thodos" and parameters.get("has_hydroxyl", 0):
        parameters.setdefault("hydroxyl_polar_term", 1)
    curve = PropertyCorrelation(
        correlation_id=correlation_id or equation,
        property_id=spec.phase + "_" + output,
        equation_id=equation,
        coefficients=parameters,
        input_units=family.input_units,
        output_unit=family.fixed_output_unit,
        validity_ranges=ranges,
        source_note=source_note,
        metadata={
            "phase": spec.phase,
            "physical_relation": spec.relation,
            "reference_method": method,
            "temperature_derivative_path": ("fixed_pressure_and_supplied_parameters"
                if spec.uses_pressure else spec.relation),
        },
    )
    # Endpoint value validation never requests a divergent critical derivative.
    for temperature in ranges["temperature"]:
        values = dict(parameters)
        for pressure in ranges.get("pressure", (None,)):
            if spec.uses_pressure:
                values["P"] = pressure
            volume_value(method, temperature, output=output, **values)
    return curve
