"""One unchanged saved source CPS: retain exact time-series and entity metadata."""

import argparse
import json
from pathlib import Path


def main():
    import COPASI

    api = COPASI

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dm = api.CRootContainer.addDatamodel()
    if not dm.loadModel(str(args.cps)):
        raise RuntimeError(api.CCopasiMessage.getAllMessageText())
    model = dm.getModel()
    model.applyInitialValues()
    container = model.getMathContainer()
    container.updateSimulatedValues(False)
    container.pushAllTransientValues()
    initial = {
        str(m.getObjectName()): {
            "value": float(m.getValue()),
            "initial": float(m.getInitialValue()),
            "status": int(m.getStatus()),
        }
        for m in model.getMetabolites()
    }
    task = dm.getTask("Time-Course")
    processed = task.process(True)
    series = task.getTimeSeries()
    titles = [str(series.getTitle(i)) for i in range(series.getNumVariables())]
    result = {
        "cps": str(args.cps),
        "version": api.CVersion.VERSION.getVersion(),
        "process_returned": bool(processed),
        "titles": titles,
        "raw_table": [
            [float(series.getData(i, j)) for j in range(series.getNumVariables())]
            for i in range(series.getRecordedSteps())
        ],
        "initial": initial,
        "final": {
            str(m.getObjectName()): {"value": float(m.getValue()), "status": int(m.getStatus())}
            for m in model.getMetabolites()
        },
        "messages": str(api.CCopasiMessage.getAllMessageText()),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "raw_table"}), flush=True)


if __name__ == "__main__":
    main()
