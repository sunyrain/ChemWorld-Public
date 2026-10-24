"""Fixed27B: exhaustive selected-source subset grid and zero-component faces.

Run as a module from the repository root in the teqp0.23.2 reference overlay.
"""

import argparse
import itertools
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import teqp
from tests.reference.run_multifluid_core import R, reference_values

from chemworld.data.multifluid_catalog import MultifluidSourceCatalog
from chemworld.physchem.molecular_eos import molecular_scaled_derivative

NAMES = (
    "Methane",
    "Ethane",
    "n-Propane",
    "n-Butane",
    "IsoButane",
    "Nitrogen",
    "CarbonDioxide",
    "Hydrogen",
)


def source_catalog():
    root = Path(teqp.get_datapath()) / "dev"
    return MultifluidSourceCatalog(
        catalog_id="teqp-fluiddata-selected-natural-gas-eight",
        revision="teqp-0.23.2-bundled",
        source="teqp0.23.2 bundled fluiddata, explicit caller import",
        fluids=[json.loads((root / "fluids" / (name + ".json")).read_text()) for name in NAMES],
        binary_pairs=json.loads((root / "mixtures/mixture_binary_pairs.json").read_text()),
        departure_functions=json.loads(
            (root / "mixtures/mixture_departure_functions.json").read_text()
        ),
    )


def select(catalog, names):
    cases = {c["name"]: c["cas"] for c in catalog.components}
    return catalog.select(
        [cases[n] for n in names],
        gas_constant_J_mol_K=R,
        gas_constant_convention="common_constant_override",
    )


class Progress:
    def __init__(self, total):
        self.total, self.done, self.current = total, 0, "initializing"
        self.start = time.monotonic()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def _heartbeat(self):
        while not self.stop.wait(20):
            self.report()

    def report(self):
        elapsed = time.monotonic() - self.start
        rate = self.done / max(elapsed, 1e-9)
        eta = (self.total - self.done) / rate if rate else None
        print(
            f"27B {self.done}/{self.total}; current={self.current}; "
            f"{elapsed:.1f}s; rate={rate:.3f}/s; ETA={eta}",
            flush=True,
        )

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite a previous attempt")
    catalog = source_catalog()
    namesets = [c for n in (1, 2, 3, 4) for c in itertools.combinations(NAMES, n)] + [NAMES]
    all_model = select(catalog, NAMES)
    records, errors, selections = [], [], []

    def compare(block, case, quantity, native, reference, atol=2e-8, rtol=2e-7):
        a, b = float(native), float(reference)
        records.append(
            {
                "block": block,
                "case": case,
                "quantity": quantity,
                "native": a,
                "reference": b,
                "atol": atol,
                "rtol": rtol,
                "passed": math.isfinite(a)
                and math.isfinite(b)
                and abs(a - b) <= atol + rtol * abs(b),
            }
        )

    with Progress(163) as progress:
        for names in namesets:
            progress.current = "/".join(names)
            try:
                model = select(catalog, names)
                source = json.loads(model.source.split("; selection=", 1)[1])
                selections.append(source)
                ref = teqp.build_multifluid_model(
                    list(names), teqp.get_datapath(), flags={"Rmodel": "CODATA"}
                )
            except Exception as exc:
                errors.append(
                    {"block": "selection", "case": names, "error": repr(exc), "planned_states": 4}
                )
                progress.done += 1
                continue
            xs = (1 / len(names),) * len(names)
            for t, rho in itertools.product((450.0, 500.0), (50.0, 1000.0)):
                case = [names, t, rho]
                try:
                    derivatives, state, mu, lnphi = reference_values(ref, t, rho, xs)
                    actual = model.state(t, rho, xs, require_stable=False)
                    for (i, j), target in derivatives.items():
                        if i + j <= 2:
                            compare(
                                "independent",
                                case,
                                f"Ar{i}{j}",
                                molecular_scaled_derivative(
                                    model, t, rho, xs, inverse_temperature_order=i, density_order=j
                                ),
                                target,
                            )
                    for field, target in state.items():
                        compare("independent", case, field, getattr(actual, field), target)
                    for i, value in enumerate(model.residual_chemical_potentials(t, rho, xs)):
                        compare("independent", case, f"mu{i}", value, mu[i], atol=2e-5)
                        compare(
                            "independent",
                            case,
                            f"lnphi{i}",
                            actual.log_fugacity_coefficients[i],
                            lnphi[i],
                        )
                    tr, rhor = model.reducing(xs)
                    compare("independent", case, "Tr", tr, ref.get_Tr(np.array(xs)))
                    compare("independent", case, "rhor", rhor, ref.get_rhor(np.array(xs)))
                except Exception as exc:
                    errors.append({"block": "independent", "case": case, "error": repr(exc)})
            try:
                restored = MultifluidSourceCatalog.from_dict(
                    json.loads(json.dumps(catalog.to_dict()))
                )
                roundtrip = select(restored, names)
                compare(
                    "invariants",
                    names,
                    "exact_catalog_selection_roundtrip",
                    roundtrip.to_dict() == model.to_dict(),
                    True,
                    atol=0,
                    rtol=0,
                )
                reverse = select(catalog, names[::-1])
                base = model.state(450.0, 500.0, xs)
                compare(
                    "invariants",
                    names,
                    "component_permutation_pressure",
                    reverse.state(450.0, 500.0, xs).pressure_Pa,
                    base.pressure_Pa,
                )
                explicit = catalog.select(
                    [c["cas"] for c in source["components"]],
                    gas_constant_J_mol_K=R,
                    gas_constant_convention="common_constant_override",
                    eos_indexes={c["cas"]: c["eos_index"] for c in source["components"]},
                    pair_indices=[p["index"] for p in source["binary_records"]],
                    selection_reason="Reconstruct exactly the recorded source selection",
                )
                compare(
                    "invariants",
                    names,
                    "explicit_source_pressure",
                    explicit.state(450.0, 500.0, xs).pressure_Pa,
                    base.pressure_Pa,
                )
                if len(names) < 8:
                    full_x = tuple(1 / len(names) if n in names else 0.0 for n in NAMES)
                    full = all_model.state(450.0, 500.0, full_x)
                    compare(
                        "faces",
                        names,
                        "alphar",
                        all_model.alphar(450.0, 500.0, full_x),
                        model.alphar(450.0, 500.0, xs),
                    )
                    for field in (
                        "pressure_Pa",
                        "residual_enthalpy_J_mol",
                        "residual_entropy_J_mol_K",
                    ):
                        compare("faces", names, field, getattr(full, field), getattr(base, field))
                    mus = all_model.residual_chemical_potentials(450.0, 500.0, full_x)
                    for name, value in zip(
                        names, model.residual_chemical_potentials(450.0, 500.0, xs), strict=True
                    ):
                        compare(
                            "faces",
                            names,
                            "active_mu_" + name,
                            mus[NAMES.index(name)],
                            value,
                            atol=2e-5,
                        )
            except Exception as exc:
                errors.append({"block": "invariants", "case": names, "error": repr(exc)})
            progress.done += 1
            if progress.done % 20 == 0:
                progress.report()
    counts = {
        b: {
            "executed": sum(r["block"] == b for r in records),
            "passed": sum(r["block"] == b and r["passed"] for r in records),
        }
        for b in ("independent", "invariants", "faces")
    }
    result = {
        "mode": "development",
        "reference": "teqp " + teqp.__version__,
        "fixed_expected": {
            "models": 163,
            "states": 652,
            "independent_comparisons": 15896,
            "invariants": 489,
            "faces": 1160,
        },
        "catalog": {
            "catalog_id": catalog.catalog_id,
            "revision": catalog.revision,
            "sha256": catalog.declaration_sha256,
            "components": catalog.components,
            "binary_table_records": len(catalog.to_dict()["binary_pairs"]),
            "departure_table_records": len(catalog.to_dict()["departure_functions"]),
        },
        "selected_models": len(selections),
        "counts": counts,
        "errors": errors,
        "selections": selections,
        "comparisons": records,
        "elapsed_s": time.monotonic() - progress.start,
    }
    result["complete_and_passed"] = (
        len(selections) == 163
        and counts["independent"]["executed"] == 15896
        and counts["invariants"]["executed"] == 489
        and counts["faces"]["executed"] == 1160
        and not errors
        and all(r["passed"] for r in records)
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("comparisons", "selections", "catalog")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
