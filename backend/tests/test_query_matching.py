from app.services.query_matching import normalize_query, query_variants, structured_query_parts


def test_normalize_query_strips_punctuation_and_case():
    assert normalize_query("R.E.M.") == "r e m"
    assert normalize_query("  Hey, Soul Sister!  ") == "hey soul sister"


def test_structured_query_parts_splits_on_dash():
    assert structured_query_parts("Bohemian Rhapsody - Queen") == ("bohemian rhapsody", "queen")


def test_structured_query_parts_splits_on_by():
    assert structured_query_parts("Bohemian Rhapsody by Queen") == ("bohemian rhapsody", "queen")


def test_structured_query_parts_rejects_short_fragment_from_title_containing_by():
    """Regression test: 'Ben E. King Stand By Me' used to split on the bare
    word 'by' (from the song title itself, not an artist-title separator)
    into first='ben e king stand', second='me' -- and that 2-character 'me'
    fragment then substring-matched 'american' inside an unrelated song's
    normalized title+artist, silently misattaching real imported tap data
    onto the wrong song (scripts/import_qbt_extended.py, song #36 -> #2)."""
    assert structured_query_parts("Ben E. King Stand By Me") is None


def test_structured_query_parts_rejects_other_short_parts():
    assert structured_query_parts("Yesterday - U2") is None  # "U2" normalizes to 2 chars
    assert structured_query_parts("Hey - Ho") is None


def test_query_variants_never_yields_a_needle_shorter_than_three_chars():
    for query in [
        "Ben E. King Stand By Me",
        "Yesterday - U2",
        "Bohemian Rhapsody by Queen",
        "Stand By Me",
    ]:
        for variant in query_variants(query):
            assert len(variant) >= 3, f"query_variants({query!r}) yielded too-short needle {variant!r}"
