from clumping_factor.methods.registry import METHOD_REGISTRY, expand_preset


def test_registry_covers_legacy_clumping_presets():
    assert expand_preset("sphere")["identifier"] == "clumping.sphere"
    assert expand_preset("raw-volume")["weighting"] == "volume"
    assert expand_preset("voronoi-transmission")["identifier"] == "transmission.voronoi"
    raw_transmission = expand_preset("raw-transmission")
    assert raw_transmission["field_representation"] == "native-cells-plus-density-grid"
    assert "grid-size" in raw_transmission["grid_requirements"]


def test_registry_exposes_all_domains():
    domains = {identifier.split(".", 1)[0] for identifier in METHOD_REGISTRY.identifiers}
    assert {"clumping", "transmission", "power-spectrum", "forest", "thermodynamics", "diagnostics", "campaign", "operations"} <= domains


def test_registry_catalog_is_sorted_and_serializable():
    catalog = METHOD_REGISTRY.catalog()
    assert [item["identifier"] for item in catalog] == sorted(item["identifier"] for item in catalog)
    required = {
        "field_representation", "mask_semantics", "execution_modes", "field_builder",
        "estimator", "selection", "producer", "command_kind", "command_variant",
    }
    assert all(required <= item.keys() for item in catalog)


def test_execution_modes_are_method_specific():
    assert METHOD_REGISTRY.get("sphere").execution_modes == ("full", "chunked", "threaded", "pbs")
    assert METHOD_REGISTRY.get("arepo-comparison").execution_modes == ("local",)
    assert METHOD_REGISTRY.get("raw-volume").execution_modes == ("full", "chunked", "pbs")


def test_registry_owns_campaign_command_capabilities():
    specs = [METHOD_REGISTRY.get(identifier) for identifier in METHOD_REGISTRY.identifiers]
    compute_methods = [spec for spec in specs if spec.command_kind]
    assert compute_methods
    assert all(spec.command_kind for spec in compute_methods)
    assert all(
        spec.command_kind in {
            "clumping-compute", "power-spectrum-compute", "alternative-compute", "ionized-sweep",
            "forest-spectra", "forest-ionizing", "forest-snapshot", "temperature", "diagnostics",
        }
        for spec in compute_methods
    )
    workflow_methods = [spec for spec in specs if spec.domain in {"forest", "thermodynamics", "diagnostics", "campaign", "operations"}]
    assert all(spec.command_kind is None for spec in workflow_methods)
