"""Independent phase split, finite washing and same-reference ideal/Davies calorics."""

from .aqueous_precipitation40_reference import MW, close, properties, transfer


def initial_stocks():
    blank = dict.fromkeys(("H2O", "Na+", "Cl-", "K+", "Br-", "NaCl(s)"), 0.0)
    return {
        "feed-a": {**blank, "H2O": 0.5 / MW, "Na+": 0.1, "Br-": 0.1},
        "feed-b": {**blank, "H2O": 0.5 / MW, "K+": 0.1, "Cl-": 0.1},
        "reactor": {**blank, "H2O": 0.05 / MW},
        "diluent": {**blank, "H2O": 8 / MW},
        **{k: dict(blank) for k in ("cake", "filtrate", "washed-cake", "wash-filtrate")},
    }


def split(stocks, source, cake, filtrate, eta, retention, family):
    if any(stocks[cake].values()) or any(stocks[filtrate].values()):
        raise ValueError("reference filtration requires actually empty receivers")
    result = {k: dict(v) for k, v in stocks.items()}
    selected = {k: n * (eta if k == "NaCl(s)" else retention) for k, n in stocks[source].items()}
    passed = {k: n - selected[k] for k, n in stocks[source].items()}
    result[cake], result[filtrate] = close(selected, family), close(passed, family)
    result[source] = dict.fromkeys(stocks[source], 0.0)
    return result


def totals(stocks, family):
    values = [properties(n, family) for n in stocks.values() if any(n.values())]
    return {key: sum(v[key] for v in values) for key in ("H", "U", "V")}


def reference_history(family, eta, retention):
    initial = initial_stocks()
    first = transfer(initial, "feed-a", "reactor", 0.5, family)
    precipitated = transfer(first, "feed-b", "reactor", 0.5, family)
    separated = split(precipitated, "reactor", "cake", "filtrate", eta, retention, family)
    washed = transfer(separated, "diluent", "cake", 0.025, family)
    recovered = split(washed, "cake", "washed-cake", "wash-filtrate", eta, retention, family)
    return (initial, first, precipitated, separated, washed, recovered)
