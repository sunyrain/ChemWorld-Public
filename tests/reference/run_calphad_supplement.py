"""The fixed 24 supplementary CEF states use the same reference-block driver."""

import importlib.util
import sys
from pathlib import Path

BASE = """
ELEMENT FE BCC_A2 55.845 0 0 !
ELEMENT NI FCC_A1 58.69 0 0 !
TYPE_DEF % SEQ * !
PHASE ALPHA % 1 1 !
CONSTITUENT ALPHA :FE,NI: !
PARAMETER G(ALPHA,FE;0) 100 -1000+10*T-20*T*LN(T); 2000 N !
PARAMETER G(ALPHA,NI;0) 100 -2000+5*T-25*T*LN(T); 2000 N !
"""
CASES = {
    "synthetic-two-state.tdb": BASE + """
    PARAMETER GD(ALPHA,FE;0) 100 1200-2*T; 2000 N !
    PARAMETER GD(ALPHA,NI;0) 100 800-T; 2000 N !
    """,
    "synthetic-einstein.tdb": BASE + """
    PARAMETER THETA(ALPHA,FE;0) 100 5.991464547107982+1E-4*T; 2000 N !
    PARAMETER THETA(ALPHA,NI;0) 100 6.396929655216146+2E-4*T; 2000 N !
    """,
    "synthetic-volume.tdb": BASE + """
    PARAMETER V0(ALPHA,FE;0) 100 1E-5; 2000 N !
    PARAMETER V0(ALPHA,NI;0) 100 1.5E-5; 2000 N !
    PARAMETER V0(ALPHA,FE,NI;0) 100 2E-6; 2000 N !
    PARAMETER VA(ALPHA,*;0) 100 2E-4*T; 2000 N !
    """,
    "synthetic-xiong.tdb": BASE.replace("PHASE ALPHA %", "PHASE ALPHA %&") + """
    TYPE_DEF & GES A_P_D ALPHA MAGNETIC 0 .4 !
    PARAMETER TC(ALPHA,FE;0) 100 1200; 2000 N !
    PARAMETER TC(ALPHA,NI;0) 100 900; 2000 N !
    PARAMETER NT(ALPHA,FE;0) 100 400; 2000 N !
    PARAMETER NT(ALPHA,NI;0) 100 500; 2000 N !
    PARAMETER BMAGN(ALPHA,FE;0) 100 2; 2000 N !
    PARAMETER BMAGN(ALPHA,NI;0) 100 1.5; 2000 N !
    """,
}


def main():
    driver_path = Path(__file__).with_name("run_calphad_block.py")
    spec = importlib.util.spec_from_file_location("calphad_reference_driver", driver_path)
    block = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(block)
    block.DATABASES = tuple(CASES)
    block.TEMPERATURES = (350.0, 650.0, 1250.0)
    block.PRESSURE = 201325.0
    block.REFERENCE_SCRIPT = Path(__file__).resolve()
    if "--reference-only" not in sys.argv:
        if "--source-directory" not in sys.argv:
            raise ValueError("--source-directory must name a new external directory")
        path = Path(sys.argv[sys.argv.index("--source-directory") + 1])
        if path.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
            raise ValueError("supplementary inputs must be outside the checkout")
        if path.exists() and "--second-order" in sys.argv:
            if any((path/name).read_text(encoding="utf-8") != text for name, text in CASES.items()):
                raise ValueError(
                    "existing supplementary inputs differ from the fixed original cases",
                )
        else:
            path.mkdir(parents=True, exist_ok=False)
            for name, text in CASES.items():
                (path / name).write_text(text, encoding="utf-8")
    return block.main()


if __name__ == "__main__":
    raise SystemExit(main())
