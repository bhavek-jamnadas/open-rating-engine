"""Tests for source-column accessors and override coexistence.

Every source column enters the pipeline as a ``SourceColumn`` layer. The pipeline
snapshots each one's raw fed-in value as ``<column>{}`` before any wave runs, so:

* the raw stays reachable via ``df['<col>{}']`` even after an override
  ``ColumnCalculation`` reclaims the plain name (the source layer is **not**
  deleted — it coexists as the accessor), and
* accessors are internal-only — stripped from the scored output.

When a column is not overridden, ``df['<col>{}']`` simply equals ``df['<col>']``.
"""

from open_rating_engine.rating_engine.pricing_layers import ColumnCalculation, SourceColumn


def test_unoverridden_accessor_equals_plain_column(pipeline, sample_data):
    # policy.base_premium arrives as [100, 200, 300, 400] and is NOT overridden.
    pipeline.add_parameter(SourceColumn("policy.base_premium{}", "policy", "base_premium"))
    # A layer reads the accessor — with no override it equals the plain column.
    pipeline.add_parameter(
        ColumnCalculation("raw_copy", "policy", "raw_copy", "df['base_premium{}']")
    )

    out = pipeline.predict(sample_data)

    assert list(out["policy"]["raw_copy"]) == [100.0, 200.0, 300.0, 400.0]
    assert list(out["policy"]["base_premium"]) == [100.0, 200.0, 300.0, 400.0]
    # The accessor is internal-only — never exposed in the output.
    assert "base_premium{}" not in out["policy"].columns


def test_override_coexists_with_source_and_raw_stays_reachable(pipeline, sample_data):
    # The source column survives as the accessor…
    pipeline.add_parameter(SourceColumn("policy.base_premium{}", "policy", "base_premium"))
    # …while a separate ColumnCalculation owns the plain name and overrides it,
    # reading its own raw via the accessor (no cycle — different layer).
    pipeline.add_parameter(
        ColumnCalculation(
            "policy.base_premium", "policy", "base_premium", "df['base_premium{}'] * 10",
        )
    )
    # Another layer reads the raw accessor — sees the pre-override value.
    pipeline.add_parameter(
        ColumnCalculation("raw_copy", "policy", "raw_copy", "df['base_premium{}']")
    )

    out = pipeline.predict(sample_data)

    # Plain name now holds the overridden (computed) value…
    assert list(out["policy"]["base_premium"]) == [1000.0, 2000.0, 3000.0, 4000.0]
    # …the raw fed-in value is still readable via the accessor…
    assert list(out["policy"]["raw_copy"]) == [100.0, 200.0, 300.0, 400.0]
    # …and the accessor itself is stripped from the output.
    assert "base_premium{}" not in out["policy"].columns


def test_accessors_stripped_from_every_frame(pipeline, sample_data):
    pipeline.add_parameter(SourceColumn("policy.base_premium{}", "policy", "base_premium"))
    pipeline.add_parameter(SourceColumn("risk.driver_youngest_age{}", "risk", "driver_youngest_age"))

    out = pipeline.predict(sample_data)

    for table, frame in out.items():
        assert not any(str(c).endswith("{}") for c in frame.columns), table


def test_no_accessor_without_source_column(pipeline, sample_data):
    # With no SourceColumn layer, no accessor is materialised.
    pipeline.add_parameter(
        ColumnCalculation("base_premium", "policy", "base_premium", "df['base_premium'] * 2")
    )
    out = pipeline.predict(sample_data)
    assert "base_premium{}" not in out["policy"].columns
    assert list(out["policy"]["base_premium"]) == [200.0, 400.0, 600.0, 800.0]


def test_source_column_accessor_visible_in_profile_predict(pipeline, sample_data, mock_context):
    pipeline.add_parameter(SourceColumn("policy.base_premium{}", "policy", "base_premium"))
    pipeline.add_parameter(
        ColumnCalculation("raw_copy", "policy", "raw_copy", "df['base_premium{}']")
    )

    out, _timing = pipeline.profile_predict(mock_context, sample_data)

    assert list(out["policy"]["raw_copy"]) == [100.0, 200.0, 300.0, 400.0]
    # Stripped here too.
    assert "base_premium{}" not in out["policy"].columns
