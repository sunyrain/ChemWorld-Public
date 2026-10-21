"""Closed resource and XPath semantics before experiment execution is enabled."""

from xml.etree import ElementTree as ET

import pytest

from chemworld.data.sedml_xml import closed_xpath, experiment_xml, local_archive_path


def test_namespace_bindings_follow_the_actual_node_scope_and_are_immutable():
    parsed = experiment_xml('<a xmlns="urn:outer" xmlns:s="urn:first"><b xmlns:s="urn:second" '
        'target="/s:root"/><c target="/s:root"/></a>')
    assert parsed.children[0].namespaces["s"] == "urn:second"
    assert parsed.children[1].namespaces["s"] == "urn:first"
    with pytest.raises(TypeError):
        parsed.children[0].namespaces["s"] = "urn:forged"
    assert parsed.element().tag == "{urn:outer}a"
    with pytest.raises(ValueError, match="DTD"):
        experiment_xml('<!DOCTYPE a [<!ENTITY x "hidden">]><a>&x;</a>')


@pytest.mark.parametrize("uri", ["../model.xml", "/root/model.xml", "C:/model.xml",
    "https://example.org/m.xml", "//example.org/m.xml", "a\\m.xml", "a/%2e%2e/../m.xml",
    "m.xml#fragment", "m.xml?query=1", "%252e%252e/m.xml", "%00.xml", "con.xml", "a/m.xml."])
def test_unsafe_resource_paths_are_rejected_without_any_io(uri):
    with pytest.raises(ValueError):
        local_archive_path(uri)


def test_internal_relative_resources_resolve_within_the_archive():
    assert local_archive_path("../models/m.xml", base="experiments") == "models/m.xml"
    assert local_archive_path("./models/./m.xml") == "models/m.xml"
    assert local_archive_path("models/with%20space.xml") == "models/with space.xml"
    assert local_archive_path(".", archive_self=True) == "."
    with pytest.raises(ValueError):
        local_archive_path("../../models/m.xml", base="experiments")


def model_tree():
    return ET.fromstring('<sbml xmlns="urn:sbml"><model><listOfParameters>'
        '<parameter id="k" value="2"/><parameter id="p" value="3"/>'
        '</listOfParameters></model></sbml>')


def test_xpath_uses_namespace_uris_and_selects_exactly_one_source_attribute():
    path = '/s:sbml/s:model/s:listOfParameters/s:parameter[@id="k"]/@value'
    element, attribute = closed_xpath(model_tree(), path, {"s": "urn:sbml"},
                                      require_attribute=True)
    assert element.get("id") == "k" and attribute == "value"
    with pytest.raises(ValueError):
        closed_xpath(model_tree(), path, {"s": "urn:wrong"})


@pytest.mark.parametrize("tail", ["s:parameter", "s:parameter/@absent", "s:*",
    "s:parameter[1]", "s:parameter[@id='unknown']", "s:parameter[@value='2']",
    "s:parameter/parent::s:model", "/s:parameter", "s:parameter[@id='k']/text()"])
def test_xpath_does_not_expand_the_supported_grammar_or_accept_ambiguous_targets(tail):
    with pytest.raises(ValueError):
        closed_xpath(model_tree(), "/s:sbml/s:model/s:listOfParameters/"+tail, {"s": "urn:sbml"})
