"""Explicit IAPWS parameter cards on the existing pure Helmholtz engine.

IAPWS R6-95(2018), Tables 1--2; IAPWS G8-10, Tables 1--3.
Scientific coefficients reproduced with attribution to IAPWS, as permitted
by the releases. No reference package, downloaded database or new EOS engine.
"""

from functools import lru_cache

from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm

WATER_MOLAR_MASS_KG_MOL = .018015268
DRY_AIR_MOLAR_MASS_KG_MOL = .02896546
_WATER_SOURCE = "https://iapws.org/technical-guidance/release/IAPWS-95"
_AIR_SOURCE = "https://iapws.org/technical-guidance/release/SeaAir"

# n, d, t, l; the first seven powers have no exponential density damping.
_WATER_POWER = (
    (.12533547935523e-1, 1, -.5, 0), (.78957634722828e1, 1, .875, 0),
    (-.87803203303561e1, 1, 1, 0), (.31802509345418, 2, .5, 0),
    (-.26145533859358, 2, .75, 0), (-.78199751687981e-2, 3, .375, 0),
    (.88089493102134e-2, 4, 1, 0), (-.66856572307965, 1, 4, 1),
    (.20433810950965, 1, 6, 1), (-.66212605039687e-4, 1, 12, 1),
    (-.19232721156002, 2, 1, 1), (-.25709043003438, 2, 5, 1),
    (.16074868486251, 3, 4, 1), (-.40092828925807e-1, 4, 2, 1),
    (.39343422603254e-6, 4, 13, 1), (-.75941377088144e-5, 5, 9, 1),
    (.56250979351888e-3, 7, 3, 1), (-.15608652257135e-4, 9, 4, 1),
    (.11537996422951e-8, 10, 11, 1), (.36582165144204e-6, 11, 4, 1),
    (-.13251180074668e-11, 13, 13, 1), (-.62639586912454e-9, 15, 1, 1),
    (-.10793600908932, 1, 7, 2), (.17611491008752e-1, 2, 1, 2),
    (.22132295167546, 2, 9, 2), (-.40247669763528, 2, 10, 2),
    (.58083399985759, 3, 10, 2), (.49969146990806e-2, 4, 3, 2),
    (-.31358700712549e-1, 4, 7, 2), (-.74315929710341, 4, 10, 2),
    (.47807329915480, 5, 10, 2), (.20527940895948e-1, 6, 6, 2),
    (-.13636435110343, 6, 10, 2), (.14180634400617e-1, 7, 10, 2),
    (.83326504880713e-2, 9, 1, 2), (-.29052336009585e-1, 9, 2, 2),
    (.38615085574206e-1, 9, 3, 2), (-.20393486513704e-1, 9, 4, 2),
    (-.16554050063734e-2, 9, 8, 2), (.19955571979541e-2, 10, 6, 2),
    (.15870308324157e-3, 10, 9, 2), (-.16388568342530e-4, 12, 8, 2),
    (.43613615723811e-1, 3, 16, 3), (.34994005463765e-1, 4, 22, 3),
    (-.76788197844621e-1, 4, 23, 3), (.22446277332006e-1, 5, 23, 3),
    (-.62689710414685e-4, 14, 10, 4), (-.55711118565645e-9, 3, 50, 6),
    (-.19905718354408, 6, 44, 6), (.31777497330738, 6, 46, 6),
    (-.11841182425981, 6, 50, 6),
)
_AIR_POWER = (
    (.118160747229, 1, 0, 0), (.713116392079, 1, .33, 0),
    (-.161824192067e1, 1, 1.01, 0), (.714140178971e-1, 2, 0, 0),
    (-.865421396646e-1, 3, 0, 0), (.134211176704, 3, .15, 0),
    (.112626704218e-1, 4, 0, 0), (-.420533228842e-1, 4, .2, 0),
    (.349008431982e-1, 4, .35, 0), (.164957183186e-3, 6, 1.35, 0),
    (-.101365037912, 1, 1.6, 1), (-.173813690970, 3, .8, 1),
    (-.472103183731e-1, 5, .95, 1), (-.122523554253e-1, 6, 1.25, 1),
    (-.146629609713, 1, 3.6, 2), (-.316055879821e-1, 3, 6, 2),
    (.233594806142e-3, 11, 3.25, 2), (.148287891978e-1, 1, 3.5, 3),
    (-.938782884667e-2, 3, 15, 3),
)


def _power(rows, source):
    return HelmholtzTerm("ResidualHelmholtzPower",
                         {key: tuple(row[i] for row in rows)
                          for i, key in enumerate(("n", "d", "t", "l"))}, source)


@lru_cache(maxsize=1)
def iapws95_water():
    """Explicit, immutable official card; homogeneous 273.16--1273 K domain.

    This factory does not assert a globally stable phase at arbitrary rho.
    G8 humid-air vapor below 273.16 K evaluates its documented alpha extension.
    """
    ideal = (
        HelmholtzTerm("IdealGasHelmholtzLead", {"a1": -8.3204464837497,
                                               "a2": 6.6832105275932}, _WATER_SOURCE),
        HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 3.00632}, _WATER_SOURCE),
        HelmholtzTerm("IdealGasHelmholtzPlanckEinstein", {
            "n": (.012436, .97315, 1.27950, .96956, .24873),
            "t": (1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105)}, _WATER_SOURCE),
    )
    residual = (
        _power(_WATER_POWER, _WATER_SOURCE),
        HelmholtzTerm("ResidualHelmholtzGaussian", {
            "n": (-.31306260323435e2, .31546140237781e2, -.25213154341695e4),
            "d": (3, 3, 3), "t": (0, 1, 4), "eta": (20, 20, 20),
            "epsilon": (1, 1, 1), "beta": (150, 150, 250), "gamma": (1.21, 1.21, 1.25),
        }, _WATER_SOURCE),
        HelmholtzTerm("ResidualHelmholtzNonAnalytic", {
            "n": (-.14874640856724, .31806110878444), "a": (3.5, 3.5),
            "b": (.85, .95), "beta": (.3, .3), "A": (.32, .32), "B": (.2, .2),
            "C": (28, 32), "D": (700, 800),
        }, _WATER_SOURCE),
    )
    return HelmholtzFluid(
        "IAPWS95-Water", "7732-18-5", WATER_MOLAR_MASS_KG_MOL,
        461.51805*WATER_MOLAR_MASS_KG_MOL, 647.096, 322/WATER_MOLAR_MASS_KG_MOL,
        273.16, 1273, 1e9, ideal, residual, _WATER_SOURCE, citation="IAPWS R6-95(2018)",
    )


@lru_cache(maxsize=1)
def iapws_dry_air():
    """G8 dry-air card, with its explicit zero-h/zero-s reference and domain."""
    ideal = (
        HelmholtzTerm("IdealGasHelmholtzLead", {
            "a1": .974502517439480e1, "a2": .100986147428912e2}, _AIR_SOURCE),
        HelmholtzTerm("IdealGasHelmholtzPower", {
            "n": (.605719400e-7, -.210274769e-4, -.158860716e-3, -.195363420e-3),
            "t": (-3, -2, -1, 1.5)}, _AIR_SOURCE),
        HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": .2490888032e1}, _AIR_SOURCE),
        HelmholtzTerm("IdealGasHelmholtzPlanckEinstein", {
            "n": (.791309509, .212236768), "t": (.2536365e2, .1690741e2)}, _AIR_SOURCE),
        HelmholtzTerm("IdealGasHelmholtzPlanckEinsteinGeneralized", {
            "n": (-.197938904,), "t": (.8731279e2,), "c": (2/3,), "d": (1,)}, _AIR_SOURCE),
    )
    return HelmholtzFluid(
        "IAPWS-G8-DryAir", "", DRY_AIR_MOLAR_MASS_KG_MOL, 8.31451, 132.6312, 10447.7,
        193, 473, 5e6, ideal, (_power(_AIR_POWER, _AIR_SOURCE),), _AIR_SOURCE,
        pseudo_pure=True, citation="IAPWS G8-10, Tables 1--3; Lemmon et al. (2000)",
    )
