# Finite mechanism fixtures

`pressure_log.yaml` and `chebyshev.yaml` are synthetic balanced isomerizations,
authored for numerical validation, not measured chemical parameters. P-log's
`temperature-range` is the finite ChemWorld import contract; Cantera itself does
not require that field and permits extrapolation outside its pressure table.
ChemWorld rejects extrapolation.

`cantera-3.1.0/h2o2.yaml` is copied from the Cantera 3.1.0 distribution
(`cantera/data/h2o2.yaml`). Its upstream BSD license is retained alongside it.
Only trailing whitespace on a blank description line is normalized; parameters
and declarations are unchanged.
Source: https://github.com/Cantera/cantera/blob/v3.1.0/data/h2o2.yaml .
It describes the hydrogen/oxygen subset of GRI-Mech 3.0, including N2. Import
explicitly selects `ohmech`, not the Redlich–Kwong phase. The selected phase's
10 species and 29 reactions are all retained. Transport/EOS/state metadata is
preserved but not implemented by the kinetics importer.
