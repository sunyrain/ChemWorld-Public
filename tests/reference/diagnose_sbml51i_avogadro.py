"""Actual import/conversion policy observation on the two saved S01 sources."""

import argparse
import json
import traceback
from pathlib import Path


def main():
    import COPASI

    api = COPASI

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for encoding in ("item_amount", "mol_concentration"):
        job = json.loads(
            (args.baseline / ("S01-" + encoding) / "direct/jobs.json").read_text(encoding="utf-8")
        )[0]
        dm = api.CRootContainer.addDatamodel()
        try:
            assert dm.importSBMLFromString(job["sbml"])
            model = dm.getModel()

            def capture(stage, model=model, encoding=encoding):
                assert model.compileIfNecessary()
                model.applyInitialValues()
                math = model.getMathContainer()
                math.updateSimulatedValues(False)
                math.pushAllTransientValues()
                return {
                    "encoding": encoding,
                    "stage": stage,
                    "avogadro": model.getAvogadro(),
                    "quantity_to_number": model.getQuantity2NumberFactor(),
                    "particles": {
                        str(m.getObjectName()): m.getValue() for m in model.getMetabolites()
                    },
                    "initial_particles": {
                        str(m.getObjectName()): m.getInitialValue() for m in model.getMetabolites()
                    },
                    "initial_concentrations": {
                        str(m.getObjectName()): m.getInitialConcentration()
                        for m in model.getMetabolites()
                    },
                    "particle_fluxes": {
                        str(r.getObjectName()): r.getParticleFlux() for r in model.getReactions()
                    },
                }

            records.append(capture("imported"))
            result = model.setAvogadro(job["expected_avogadro"], api.CCore.Framework_Concentration)
            records.append({"encoding": encoding, "setAvogadro_result": result})
            records.append(capture("configured"))
            model.updateInitialValues(api.CCore.Framework_Concentration)
            records.append(capture("synchronized"))
        except Exception as exc:
            records.append(
                {"encoding": encoding, "error": repr(exc), "traceback": traceback.format_exc()}
            )
        finally:
            api.CRootContainer.removeDatamodel(dm)
            (args.output / "summary.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
    (args.output / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
