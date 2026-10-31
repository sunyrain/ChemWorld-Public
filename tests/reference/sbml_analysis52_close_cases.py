"""Fixed new analysis cases and independent equations; no production derivatives."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from tests.reference.sbml_finite_time52_cases import model as finite_model

from chemworld.data.sbml import SBMLModel
from chemworld.examples.sbml_calibration import source


def model(kind, *, scaled=False):
    if kind in {"birth", "closed", "nonnormal", "growth", "nonlinear"}:
        return finite_model(kind, scaled=scaled)
    if kind == "neutral":
        base = finite_model("birth")
        return SBMLModel(
            base.to_xml().replace("<ci>j</ci>", "<apply><times/><ci>a</ci><ci>X</ci></apply>"),
            source(),
        )
    if kind in {"separated", "defective"}:
        base = finite_model("nonnormal")
        return SBMLModel(
            base.to_xml().replace(
                'id="d" value="1.2"',
                'id="d" value="20.0"' if kind == "separated" else 'id="d" value="0.5"',
            ),
            source(),
        )
    if kind == "cycle":
        reactions = "".join(
            f'<reaction id="r{i}" reversible="false"><listOfReactants>'
            f'<speciesReference species="{a}" stoichiometry="1" constant="true"/>'
            "</listOfReactants><listOfProducts>"
            f'<speciesReference species="{b}" stoichiometry="1" constant="true"/>'
            '</listOfProducts><kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">'
            f"<apply><times/><ci>k</ci><ci>{a}</ci></apply></math></kineticLaw></reaction>"
            for i, (a, b) in enumerate((("X", "Y"), ("Y", "Z"), ("Z", "X")))
        )
        species = "".join(
            f'<species id="{s}" compartment="V" initialAmount="{n}" '
            'hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>'
            for s, n in zip("XYZ", (0.2, 0.3, 0.5), strict=True)
        )
        xml = (
            '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
            '<model id="cycle" substanceUnits="mole" extentUnits="mole" timeUnits="second" '
            'volumeUnits="litre"><listOfUnitDefinitions><unitDefinition id="per_s">'
            '<listOfUnits><unit kind="second" exponent="-1" scale="0" multiplier="1"/>'
            "</listOfUnits></unitDefinition></listOfUnitDefinitions><listOfCompartments>"
            '<compartment id="V" spatialDimensions="3" size="1" units="litre" constant="true"/>'
            "</listOfCompartments><listOfSpecies>"
            + species
            + '</listOfSpecies><listOfParameters><parameter id="k" value="1" units="per_s" '
            'constant="true"/></listOfParameters><listOfReactions>'
            + reactions
            + "</listOfReactions></model></sbml>"
        )
        return SBMLModel(xml, source())
    raise ValueError(kind)


def initial_and_matrix(kind):
    if kind == "birth":
        return np.array([1.0]), np.array([[-0.5]]), np.array([0.5])
    if kind == "growth":
        return np.array([0.5]), np.array([[0.4]]), np.array([0.0])
    if kind == "neutral":
        return np.array([1.0]), np.array([[0.0]]), np.array([0.0])
    if kind == "closed":
        return np.array([0.8, 0.2]), np.array([[-0.3, 0.2], [0.3, -0.2]]), np.zeros(2)
    if kind in {"nonnormal", "separated", "defective"}:
        rate = {"nonnormal": 1.2, "separated": 20.0, "defective": 0.5}[kind]
        return np.array([0.6, 4.0]), np.array([[-0.5, 0.0], [8.0, -rate]]), np.array([0.3, 0])
    if kind == "cycle":
        return (
            np.array([0.2, 0.3, 0.5]),
            np.array([[-1, 0, 1], [1, -1, 0], [0, 1, -1]]),
            np.zeros(3),
        )
    raise ValueError(kind)


def reference(kind, time_s):
    if kind == "nonlinear":
        a, b, total, initial = 0.7, 0.3, 1.0, 0.8
        delta = np.sqrt(b * b + 8 * a * b * total)
        positive, negative = (-b + delta) / (4 * a), (-b - delta) / (4 * a)
        r0 = (initial - positive) / (initial - negative)
        ratio = r0 * np.exp(-delta * time_s)
        amount = (positive - negative * ratio) / (1 - ratio)
        log_tangent = -delta * time_s + 2 * np.log((1 - r0) / (1 - ratio))
        return {
            "amounts": [amount, (total - amount) / 2],
            "eigenvalues": [-4 * a * amount - b],
            "log_tangent": log_tangent,
        }
    initial, matrix, incoming = initial_and_matrix(kind)
    augmented = np.zeros((len(initial) + 1, len(initial) + 1))
    augmented[:-1, :-1], augmented[:-1, -1] = matrix, incoming
    state = (expm(augmented * time_s) @ np.r_[initial, 1])[:-1]
    spectrum = np.linalg.eigvals(matrix)
    if kind in {"closed", "cycle"}:
        spectrum = [x for x in spectrum if abs(x) > 1e-10]
    return {"amounts": state, "matrix": matrix, "eigenvalues": spectrum}


def qr_reference(kind, basis, metric, schedule):
    basis, metric = np.asarray(basis), np.asarray(metric)
    q = np.eye(basis.shape[1])
    logs = np.zeros(basis.shape[1])
    previous = elapsed = 0.0
    for row in schedule:
        t = row["time_s"]
        if kind == "nonlinear":
            growth = np.array(
                [
                    [
                        np.exp(
                            reference(kind, t)["log_tangent"]
                            - reference(kind, previous)["log_tangent"]
                        )
                    ]
                ]
            )
        else:
            matrix = initial_and_matrix(kind)[1]
            reduced = basis.T @ ((matrix * metric[None, :]) / metric[:, None]) @ basis
            growth = expm(reduced * (t - previous))
        q, r = np.linalg.qr(growth @ q)
        signs = np.where(np.diag(r) < 0, -1.0, 1.0)
        q, r = q * signs, signs[:, None] * r
        if row["phase"] == "measurement":
            logs += np.log(np.diag(r))
            elapsed += t - previous
        previous = t
    return logs / elapsed
