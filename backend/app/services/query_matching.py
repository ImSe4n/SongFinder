import re

_QUERY_SPLIT_RE = re.compile(
    r"^\s*(?P<first>.+?)\s*(?:-|–|—|/|\bby\b)\s*(?P<second>.+?)\s*$",
    re.IGNORECASE,
)
_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)

# Below this length a "part" is too short to be a meaningful artist/title
# fragment and is more likely a mis-split — e.g. "Ben E. King Stand By Me"
# splits on the bare word "by" (meant for "<title> by <artist>" queries)
# into first="ben e king stand", second="me". "me" then substring-matches
# almost anything ("american" contains "me"), silently misattaching one
# song's data onto an unrelated one. Rejecting short parts here is the
# first of two guards — see MIN_SUBSTRING_NEEDLE_LENGTH in catalog.py for
# the second, independent of how a short needle was produced.
MIN_QUERY_PART_LENGTH = 3


def normalize_query(value: str) -> str:
    return " ".join(_NON_WORD_RE.sub(" ", value.lower()).split())


def structured_query_parts(query: str) -> tuple[str, str] | None:
    match = _QUERY_SPLIT_RE.match(query)
    if not match:
        return None

    first = normalize_query(match.group("first"))
    second = normalize_query(match.group("second"))
    if len(first) < MIN_QUERY_PART_LENGTH or len(second) < MIN_QUERY_PART_LENGTH:
        return None

    return first, second


def query_variants(query: str) -> list[str]:
    variants: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = normalize_query(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            variants.append(normalized)

    add(query)

    parts = structured_query_parts(query)
    if parts is not None:
        first, second = parts
        add(first)
        add(second)
        add(f"{first} {second}")
        add(f"{second} {first}")

    return variants