"""Acceptance index for the complete four-module chemicals 1.5.2 public surface.

This is test evidence, not a second production registry. Native dispatch and
source catalogs remain authoritative; the index checks every reference export.
"""

import importlib
import inspect
import json
from pathlib import Path

from chemworld.physchem.correlation_families import CORRELATION_FAMILIES
from chemworld.physchem.viscosity_kernels import VISCOSITY_METHODS


def entry(native, blocks, **details):
    return {"native": native, "qualification_blocks": blocks, **details}


def method_catalog():
    root = Path(__file__).resolve().parents[2]
    p = "chemworld.physchem."
    lj = {
        "Stockmayer": entry(
            "chemworld.data.lennard_jones.LennardJonesDatabase.value",
            ["E1/sources"],
            field="Stockmayer",
            selection="explicit source",
        ),
        "Stockmayer_methods": entry(
            "chemworld.data.lennard_jones.LennardJonesDatabase.methods",
            ["E1/sources"],
            field="Stockmayer",
        ),
        "molecular_diameter": entry(
            "chemworld.data.lennard_jones.LennardJonesDatabase.value",
            ["E1/sources"],
            field="molecular_diameter",
        ),
        "molecular_diameter_methods": entry(
            "chemworld.data.lennard_jones.LennardJonesDatabase.methods",
            ["E1/sources"],
            field="molecular_diameter",
        ),
        "T_star": entry(p + "lennard_jones.lennard_jones_reduced_temperature", ["E1/methods"]),
    }
    for family, methods in (
        (
            "epsilon",
            (
                "Bird_Stewart_Lightfoot_boiling",
                "Bird_Stewart_Lightfoot_critical",
                "Bird_Stewart_Lightfoot_melting",
                "Flynn",
                "Stiel_Thodos",
                "Tee_Gotoh_Steward_1",
                "Tee_Gotoh_Steward_2",
            ),
        ),
        (
            "sigma",
            (
                "Bird_Stewart_Lightfoot_boiling",
                "Bird_Stewart_Lightfoot_critical_1",
                "Bird_Stewart_Lightfoot_critical_2",
                "Bird_Stewart_Lightfoot_melting",
                "Flynn",
                "Silva_Liu_Macedo",
                "Stiel_Thodos",
                "Tee_Gotoh_Steward_1",
                "Tee_Gotoh_Steward_2",
            ),
        ),
    ):
        for method in methods:
            lj[family + "_" + method] = entry(
                p
                + "lennard_jones.lennard_jones_"
                + ("energy" if family == "epsilon" else "diameter"),
                ["E1/methods"],
                method=method.replace("Bird_Stewart_Lightfoot", "BSL"),
            )
    for name in ("Kim_Monroe", "Neufeld_Janzen_Aziz"):
        lj["collision_integral_" + name] = entry(
            p + "lennard_jones.lennard_jones_collision_integral",
            ["E1/methods"],
            method=name,
            order_pairs=16,
        )
    mu = {
        name: entry("correlation:viscosity_" + name.lower(), ["E2/methods"])
        for name in VISCOSITY_METHODS
        if name not in {"mu_IAPWS"}
    }
    mu.update(
        {
            "Brokaw": entry(p + "viscosity_mixtures.brokaw_viscosity", ["E2/models"]),
            "Herning_Zipperer": entry(
                p + "viscosity_mixtures.herning_zipperer_viscosity", ["E2/methods"]
            ),
            "Lorentz_Bray_Clarke": entry(
                p + "viscosity_mixtures.lorentz_bray_clarke_viscosity", ["E2/models"]
            ),
            "Lucas": entry(
                p + "dense_viscosity.lucas_liquid_viscosity", ["E2/models", "E2/branches"]
            ),
            "Twu_1985": entry(p + "petroleum_viscosity.twu_1985_viscosity", ["E2/models"]),
            "dPPDS9_dT": entry(
                p + "viscosity_kernels.viscosity_derivatives", ["E2/methods"], method="PPDS9"
            ),
            "dmu_Yaws_dT": entry(
                p + "viscosity_kernels.viscosity_derivatives", ["E2/methods"], method="mu_Yaws"
            ),
            "mu_IAPWS": entry(p + "water_transport.water_viscosity_iapws", ["E2/water"]),
            "mu_Yaws_fitting_jacobian": entry(
                p + "viscosity_kernels.viscosity_yaws_parameter_jacobian", ["E2/methods"]
            ),
            "viscosity_converter": entry(
                p + "viscosity_scales.viscosity_converter",
                ["E2/scales", "E2/scale-monotone"],
                scale_count=47,
            ),
            "viscosity_index": entry(
                p + "petroleum_viscosity.viscosity_index", ["E2/index-standard"]
            ),
        }
    )
    mu["Przedziecki_Sridhar"]["qualification_blocks"] = ["E2/models", "E2/branches"]
    for name in ("Wilke", "Wilke_large", "Wilke_prefactored"):
        mu[name] = entry(
            p + "viscosity_mixtures.wilke_viscosity",
            ["E2/methods"],
            note="Same kernel; optional explicit factors, O(N) scratch storage",
        )
    mu["Wilke_prefactors"] = entry(p + "viscosity_mixtures.wilke_prefactors", ["E2/methods"])
    conductivity = {}
    evidence = json.loads(
        (root / "workstreams/world_foundation/reports/convergence/22/"
         "NATIVE_CONV22E3_RESULTS.json").read_text()
    )
    for row in evidence["methods"]:
        conductivity[row["reference"]] = entry(
            [
                "correlation:" + name if name.startswith("conductivity_") else name
                for name in row["native"]
            ],
            ["E3/" + row["qualification_block"]],
        )
    interface_names = [
        "API10A32",
        "PPDS14",
        "Aleem",
        "Brock_Bird",
        "Hakim_Steinberg_Stiel",
        "ISTExpansion",
        "Jasper",
        "Mersmann_Kind_sigma",
        "Miqueu",
        "Pitzer_sigma",
        "REFPROP_sigma",
        "Sastri_Rao",
        "Somayajulu",
        "Watson_sigma",
        "Zuo_Stenby",
        "sigma_Gharagheizi_1",
        "sigma_Gharagheizi_2",
        "sigma_IAPWS",
    ]
    interface = {
        name: entry("correlation:interface_" + name.lower(), ["E4/methods"])
        for name in interface_names
    }
    for ref, native, block in (
        ("Diguilio_Teja", "diguilio_teja", "mixtures"),
        ("Weinaug_Katz", "weinaug_katz", "mixtures"),
        ("Winterfeld_Scriven_Davis", "winterfeld_scriven_davis", "mixtures"),
        ("Meybodi_Daryasafar_Karimi", "water_oil_interfacial_tension", "water-oil"),
    ):
        interface[ref] = entry(p + "interfaces." + native, ["E4/" + block])
    return {
        "lennard_jones": lj,
        "viscosity": mu,
        "thermal_conductivity": conductivity,
        "interface": interface,
    }


def verify_catalog():
    catalog = method_catalog()
    for module, mapping in catalog.items():
        reference = importlib.import_module("chemicals." + module)
        names = {name for name in reference.__all__ if inspect.isfunction(getattr(reference, name))}
        assert names == set(mapping), {
            "module": module,
            "missing": sorted(names - mapping.keys()),
            "extra": sorted(mapping.keys() - names),
        }
        for item in mapping.values():
            targets = item["native"] if isinstance(item["native"], list) else [item["native"]]
            for target in targets:
                if target.startswith("correlation:"):
                    assert target.removeprefix("correlation:") in CORRELATION_FAMILIES, target
                else:
                    parts = target.split(".")
                    for end in range(len(parts) - 1, 1, -1):
                        try:
                            resolved = importlib.import_module(".".join(parts[:end]))
                        except ModuleNotFoundError:
                            continue
                        for part in parts[end:]:
                            resolved = getattr(resolved, part)
                        assert callable(resolved), target
                        break
                    else:
                        raise AssertionError(target)
    assert [len(v) for v in catalog.values()] == [23, 29, 30, 22]
    return catalog


if __name__ == "__main__":
    print(json.dumps({module: len(v) for module, v in verify_catalog().items()}))
