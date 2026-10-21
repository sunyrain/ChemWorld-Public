from __future__ import annotations

import pytest

from chemworld.physchem import (
    FragmentIonSpec,
    MassSpectrumAnalyteSpec,
    MaturityLevel,
    isotope_envelope,
    simulate_mass_spectrum,
    spectroscopy_model_cards,
    validate_model_card,
)


def _peak_by_shift(formula: str, shift: int):
    return next(peak for peak in isotope_envelope(formula) if peak.nominal_mass_shift == shift)


def test_carbon_m_plus_one_envelope_matches_binomial_abundance_ratio() -> None:
    envelope = isotope_envelope("C2H6")
    m = next(peak for peak in envelope if peak.nominal_mass_shift == 0)
    m_plus_one = next(peak for peak in envelope if peak.nominal_mass_shift == 1)
    expected_ratio = (
        2.0 * 0.0107 / 0.9893
        + 6.0 * 0.000115 / 0.999885
    )

    assert m.relative_intensity == pytest.approx(100.0)
    assert m_plus_one.relative_intensity / m.relative_intensity == pytest.approx(
        expected_ratio,
        rel=0.02,
    )
    assert sum(peak.abundance for peak in envelope) == pytest.approx(1.0)


def test_chlorine_and_bromine_show_diagnostic_m_plus_two_patterns() -> None:
    chlorine_ratio = (
        _peak_by_shift("C2H5Cl", 2).relative_intensity
        / _peak_by_shift("C2H5Cl", 0).relative_intensity
    )
    bromine_ratio = (
        _peak_by_shift("C2H5Br", 2).relative_intensity
        / _peak_by_shift("C2H5Br", 0).relative_intensity
    )

    assert chlorine_ratio == pytest.approx(0.2422 / 0.7578, rel=0.02)
    assert bromine_ratio == pytest.approx(0.4931 / 0.5069, rel=0.02)


def test_mass_spectrum_reports_fragments_and_detector_uncertainty() -> None:
    analyte = MassSpectrumAnalyteSpec(
        analyte_id="ethanol",
        formula="C2H6O",
        molecular_ion_charge=1,
        ionization_method="EI_70eV",
        fragments=(
            FragmentIonSpec(
                fragment_id="ch2oh_plus",
                formula="CH3O",
                charge=1,
                relative_intensity=100.0,
                assignment="alpha-cleavage oxygenated fragment",
                neutral_loss="CH3",
            ),
        ),
        detector_response_factor=2.0e9,
        detector_relative_standard_deviation=0.05,
        provenance_id="synthetic-ei-fragment-card",
    )
    report = simulate_mass_spectrum(analyte, analyte_amount_mol=2.0e-9)

    assert report.detector_response_mean == pytest.approx(4.0)
    assert report.detector_response_standard_deviation == pytest.approx(0.2)
    assert report.fragments[0].mass_to_charge < report.isotope_envelope[0].mass_to_charge
    assert report.fragments[0].neutral_loss == "CH3"
    assert report.warnings == ()


def test_high_uncertainty_and_missing_fragments_are_explicit_warnings() -> None:
    analyte = MassSpectrumAnalyteSpec(
        analyte_id="chloromethane",
        formula="CH3Cl",
        molecular_ion_charge=1,
        ionization_method="EI",
        fragments=(),
        detector_response_factor=1.0,
        detector_relative_standard_deviation=0.25,
        provenance_id="screening-ms-card",
    )
    report = simulate_mass_spectrum(analyte, analyte_amount_mol=1.0)

    assert "no_fragmentation_metadata" in report.warnings
    assert "high_detector_response_uncertainty" in report.warnings


def test_isotope_envelope_rejects_unsupported_elements_and_decimal_formula() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        isotope_envelope("Fe")
    with pytest.raises(ValueError, match="integer atom counts"):
        isotope_envelope("C1.5H3")


def test_fragment_heavier_than_parent_is_rejected() -> None:
    analyte = MassSpectrumAnalyteSpec(
        analyte_id="bad_fragment",
        formula="CH4",
        molecular_ion_charge=1,
        ionization_method="EI",
        fragments=(
            FragmentIonSpec(
                fragment_id="impossible",
                formula="C2H6",
                charge=1,
                relative_intensity=10.0,
                assignment="invalid fragment",
            ),
        ),
        detector_response_factor=1.0,
        detector_relative_standard_deviation=0.1,
        provenance_id="invalid-fragment-test",
    )

    with pytest.raises(ValueError, match="cannot exceed"):
        simulate_mass_spectrum(analyte, analyte_amount_mol=1.0)


def test_mass_spectrometry_model_card_is_reference_validated() -> None:
    card = {
        item.model_id: item for item in spectroscopy_model_cards()
    }["small_formula_isotope_fragment_ms_v1"]

    assert card.maturity is MaturityLevel.REFERENCE_VALIDATED
    assert validate_model_card(card) == []
