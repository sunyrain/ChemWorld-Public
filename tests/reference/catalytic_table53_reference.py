"""Independent physical-coordinate WLS; no native energy/fitting evaluators."""

import numpy as np


def weighted_reference(calibration, fixed):
    table = calibration.table
    columns = {c.quantity_id: c for c in table.columns}
    bep = calibration.energy_kind == "activation_free_energy"
    names = (
        ("intercept_J_mol", "alpha")
        if bep
        else ("intercept_J_mol", "coefficient:binding", "coefficient:strain")
    )
    train, heldout = [], []
    for row in table.rows:

        def si(key, row=row):
            value = row.values[key]
            scale = {"J/mol": 1.0, "kJ/mol": 1000.0}[value.unit]
            return value.value * scale

        x = [1.0, si("delta_g")] if bep else [1.0, si("binding"), si("strain")]
        sigma = (
            row.values["energy"].uncertainty
            * {"J/mol": 1.0, "kJ/mol": 1000.0}[columns["energy"].unit]
        )
        item = (x, si("energy"), sigma)
        (train if row.split == "train" else heldout).append(item)
    full_x = np.asarray([r[0] for r in train])
    response = np.asarray([r[1] for r in train])
    sigmas = np.asarray([r[2] for r in train])
    free = [i for i, name in enumerate(names) if name not in fixed]
    for i, name in enumerate(names):
        if name in fixed:
            response = response - full_x[:, i] * fixed[name]
    design = full_x[:, free] / sigmas[:, None]
    target = response / sigmas
    solution = np.linalg.lstsq(design, target, rcond=None)[0]
    covariance = np.linalg.inv(design.T @ design)
    estimate = dict(fixed)
    estimate.update({names[i]: float(value) for i, value in zip(free, solution, strict=True)})
    coefficients = np.asarray([estimate[name] for name in names])
    objective = float(
        np.sum(((full_x @ coefficients - np.asarray([r[1] for r in train])) / sigmas) ** 2)
    )
    return {
        "estimate": estimate,
        "covariance": covariance,
        "standard_errors": np.sqrt(np.diag(covariance)),
        "objective": objective,
        "validation_predictions_J_mol": np.asarray([r[0] for r in heldout]) @ coefficients,
        "physical_weighted_jacobian": design,
        "free_names": [names[i] for i in free],
    }
