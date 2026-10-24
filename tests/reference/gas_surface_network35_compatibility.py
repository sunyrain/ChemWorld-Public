"""Direct old/new source comparison under one locked Python environment."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROBE = r"""
import json,sys
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
sys.path.insert(0,sys.argv[2])
import chemworld.physchem.gas_reactor_network as gas
from chemworld.examples.gas_network_sensitivity import example_inputs
from chemworld.world.gas_reactor_network import gas_reactor_network_world
from gas_network35_energy_cases import case,native
rows=[]
for name in ('isothermal','thermal'):
    if name=='isothermal':
        spec,network,amounts=example_inputs()
    else:
        card=case('loop')
        spec,network,_=native(card)
        amounts=card['amounts']
    state=gas.pack_gas_reactor_network_state(spec,network,amounts)
    result,diagnostic=gas.advance_gas_reactor_network(spec,network,state,.2)
    final={field.name:(dict(value) if isinstance(value,Mapping) else value)
           for field in fields(result.final_state)
           for value in [getattr(result.final_state,field.name)]}
    rows.append({'scenario':name,'spec_codec':spec.to_dict(),
                 'world_codec':gas_reactor_network_world(spec,network,amounts).to_dict(),
                 'physical_endpoint':final,'physical_diagnostics':diagnostic})
Path(sys.argv[1]).write_text(json.dumps({'module_path':gas.__file__,'rows':rows},indent=2,allow_nan=False))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve previous compatibility observations")
    root = Path(__file__).resolve().parents[2]
    raw = []
    with tempfile.TemporaryDirectory(prefix="chemworld-35D-compat-") as temporary:
        for name, source in (("old", args.baseline_src), ("current", root / "src")):
            output = Path(temporary) / (name + ".json")
            environment = {**os.environ, "PYTHONPATH": str(source.resolve())}
            subprocess.run(
                [sys.executable, "-c", PROBE, str(output), str(Path(__file__).parent.resolve())],
                cwd=temporary,
                env=environment,
                check=True,
            )
            report = json.loads(output.read_text())
            if not Path(report["module_path"]).is_relative_to(source.resolve()):
                raise ValueError("probe did not load the requested source tree")
            raw.append(report)
    checks = [
        {"scenario": old["scenario"], "label": label, "passed": old[label] == new[label]}
        for old, new in zip(raw[0]["rows"], raw[1]["rows"], strict=True)
        for label in ("spec_codec", "world_codec", "physical_endpoint", "physical_diagnostics")
    ]
    payload = {
        "baseline_commit": "722bc8f4",
        "planned_comparisons": 8,
        "passed_comparisons": sum(c["passed"] for c in checks),
        "checks": checks,
        "raw": raw,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "raw"}), flush=True)
    if payload["passed_comparisons"] != 8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
