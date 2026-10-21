"""Independent pinned symbolic model, constrained minimization and caloric curves.

This module is loaded only by the isolated reference interpreter. It imports
no ChemWorld code and receives no native constitution or energy values.
"""

from collections import Counter
from copy import deepcopy

import numpy as np
import pycalphad
import pycalphad.variables as v
import symengine as se
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import LinearConstraint, least_squares, minimize

FILES = ("2026-Dixon-Na-K-Cl-I.dat", "Kaye_Pd-Ru-Tc-Mo.dat", "KF-NIF2_switched.dat",
         "MQMQA-tern-tests.dat", "Ocadiz-Flores.dat", "Shishin_Fe-Sb-O-S_slag.dat", "Viitala.dat")
PRESSURE = 101325.


def phases(directory):
    from pycalphad.io.cs_dat import parse_cs_dat

    if pycalphad.__version__ != "0.11.2":
        raise ValueError("independent reference must be pycalphad 0.11.2")
    for name in FILES:
        path = directory/name
        header, solutions, pure = parse_cs_dat(path.read_text(encoding="utf-8-sig").upper())
        database = pycalphad.Database(str(path))
        originals = (*solutions, *pure)
        counts, seen = Counter(p.phase_name for p in originals), Counter()
        for raw in originals:
            seen[raw.phase_name] += 1
            if raw.phase_type not in {"SUBG", "SUBQ"}:
                continue
            phase = (raw.phase_name if counts[raw.phase_name] == 1 else
                     f"{raw.phase_name}#OCCURRENCE{seen[raw.phase_name]}")
            selected = database
            if phase != raw.phase_name:
                selected = deepcopy(database)
                occurrence = deepcopy(raw)
                occurrence.phase_name = phase
                occurrence.insert(selected, header.pure_elements,
                                  header.gibbs_coefficient_idxs, header.excess_coefficient_idxs)
                selected.process_parameter_queue()
            yield name, phase, raw, header, selected


def source_issues(raw, header):
    """Independent source identity/electroneutrality checks, with original values."""
    issues = []
    nc = len(raw.subl_1_const)
    ion_names = [*raw.subl_1_const, *raw.subl_2_const]
    charges = [*raw.subl_1_charges, *(-q for q in raw.subl_2_charges)]
    # The seven fixed files use monatomic ions or VA. Element supports alone
    # already expose the two mismatched source pair mappings, including the
    # supplied switched fixture; no native formula interpretation is reused.
    for (c, a), member in zip(raw.subl_const_idx_pairs, raw.endmembers, strict=True):
        support = {e for i in (c-1, nc+a-1) for e in v.Species(ion_names[i]).constituents
                   if e not in {"VA", "/-"}}
        authored = {e for e, amount in zip(header.pure_elements,
            member.stoichiometry_pure_elements, strict=True) if amount and e not in {"VA", "/-"}}
        if support != authored:
            issues.append({"kind": "pair_element_identity", "member": member.species_name,
                           "ion_names": [ion_names[c-1], ion_names[nc+a-1]],
                           "source_elements": sorted(authored), "ion_elements": sorted(support)})
    for record in raw.quadruplets:
        charge = sum(charges[i-1]/z for i, z in zip(record.quadruplet_idxs,
                     record.quadruplet_coordinations, strict=True))
        if abs(charge) > 1e-8:
            issues.append({"kind": "quadruplet_charge", "indices": record.quadruplet_idxs,
                           "coordinations": record.quadruplet_coordinations,
                           "residual": charge})
    return issues


class ReferencePhase:
    def __init__(self, database, phase, header):
        from pycalphad.models.model_mqmqa import ModelMQMQA

        components = sorted(database.elements)
        if any("VA" in s.constituents for s in database.species) and "VA" not in components:
            components.append("VA")
        model = ModelMQMQA(database, components, phase)
        variables = [v.Y(phase, 0, species) for species in model.constituents[0]]
        self.elements = sorted({e for ion in (*model.cations, *model.anions)
                                for e in ion.constituents
                                if e in header.pure_elements and e not in {"VA", "/-"}})
        self.n, self.ne = len(variables), len(self.elements)
        self.initial = np.full(self.n, 1/self.n)
        atoms = [model.moles(e, per_formula_unit=True) for e in self.elements]
        coefficients = np.array([[float(a.diff(q)) for q in variables] for a in atoms])
        initial_atoms = coefficients@self.initial
        self.x = initial_atoms/initial_atoms.sum()
        equalities = np.vstack((np.ones(self.n), coefficients
                                -self.x[:, None]*coefficients.sum(axis=0)))
        _, singular, vectors = np.linalg.svd(equalities, full_matrices=True)
        rank = int(np.sum(singular > 1e-10*max(singular[0], 1.)))
        self.matrix, self.tangent = vectors[:rank], vectors[rank:].T
        self.target = self.matrix@self.initial
        g = model.GM
        gradient = [g.diff(q) for q in variables]
        x = [model.moles(e) for e in self.elements]
        expressions = [g, model.HM, *gradient,
                       *(d.diff(q) for d in gradient for q in variables), *x,
                       *(amount.diff(q) for amount in x for q in variables)]
        self.evaluator = se.Lambdify([v.T, v.P, *variables], expressions, cse=True, backend="llvm")

    def at(self, temperature):
        n, ne, matrix = self.n, self.ne, self.matrix
        energy_scale = 8.3145*temperature

        def evaluate(y):
            values = np.asarray(self.evaluator([temperature, PRESSURE, *y]), dtype=float).ravel()
            offset = 2+n+n*n
            return (values[0], values[1], values[2:2+n], values[2+n:offset].reshape(n, n),
                    values[offset:offset+ne], values[offset+ne:].reshape(ne, n))

        answer = minimize(lambda y: evaluate(y)[0]/energy_scale, self.initial,
            jac=lambda y: evaluate(y)[2]/energy_scale, method="SLSQP",
            bounds=[(1e-12, 1-1e-12)]*n,
            constraints=LinearConstraint(matrix, self.target, self.target),
            options={"maxiter": 1000, "ftol": 1e-13})
        multiplier = np.linalg.lstsq(matrix.T, evaluate(answer.x)[2]/energy_scale, rcond=None)[0]

        def residual(z):
            return np.r_[evaluate(z[:n])[2]/energy_scale-matrix.T@z[n:],
                         matrix@z[:n]-self.target]

        def jacobian(z):
            return np.block([[evaluate(z[:n])[3]/energy_scale, -matrix.T],
                             [matrix, np.zeros((len(matrix), len(matrix)))]])

        refined = least_squares(residual, np.r_[answer.x, multiplier], jac=jacobian,
            bounds=(np.r_[np.full(n, 1e-12), np.full(len(matrix), -np.inf)],
                    np.r_[np.full(n, 1-1e-12), np.full(len(matrix), np.inf)]),
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
        norm = float(np.max(np.abs(residual(refined.x))))
        if not refined.success or norm > 1e-9:
            raise ValueError(f"independent KKT refinement failed: {norm} {refined.message}")
        q = refined.x[:n]
        g, h, gradient, hessian, x, dx = evaluate(q)
        curvature = None
        if self.tangent.shape[1]:
            curvature = float(np.linalg.eigvalsh(self.tangent.T@hessian@self.tangent).min())
            if curvature < -1e-6*max(1., float(np.max(np.abs(hessian)))):
                raise ValueError("independent reference is an internal saddle")
        mu_matrix = np.block([[dx.T, np.ones((n, 1))], [x[None, :], np.zeros((1, 1))]])
        mu_rhs = np.r_[gradient, g]
        singular = np.linalg.svd(mu_matrix, compute_uv=False)
        rank = int(np.sum(singular > 1e-10*singular[0]))
        mu = None
        if rank == ne+1:
            solution = np.linalg.lstsq(mu_matrix, mu_rhs, rcond=1e-10)[0]
            mu = dict(zip(self.elements, map(float, solution[:ne]), strict=True))
        return {"T": float(temperature), "G": float(g), "H": float(h),
                "x": dict(zip(self.elements, map(float, x), strict=True)),
                "sites": list(map(float, q)), "mu": mu,
                "mu_status": "determined" if mu else "underdetermined",
                "diagnostics": {"slsqp_success": bool(answer.success),
                    "slsqp_message": str(answer.message),
                    "refinement_success": bool(refined.success),
                    "residual": norm, "minimum_tangent_eigenvalue": curvature,
                    "mu_matrix_rank": rank}}


def trajectory(points, stride):
    nodes = points[::stride]
    h = [.001*p["H"]+.2*(p["T"]-1100) for p in nodes]
    if min(np.diff(h)) <= 0:
        raise ValueError("independent caloric branch is not increasing")
    curve = PchipInterpolator([p["T"] for p in nodes], h).derivative()
    current, endpoints = 1100., []
    for target in (1130., 1080.):
        def rhs(t, y, target=target):
            if not 1050 <= y[0] <= 1150 or float(curve(y[0])) <= 0:
                raise ValueError("independent trajectory left its positive caloric branch")
            return [float(np.clip(.05*(target-y[0]), -2, 2))/float(curve(y[0]))]

        answer = solve_ivp(rhs, (0, 3), [current], method="DOP853", rtol=1e-10,
                           atol=1e-11, max_step=.05)
        if not answer.success:
            raise ValueError(answer.message)
        current = float(answer.y[0, -1])
        endpoints.append(current)
    return endpoints
