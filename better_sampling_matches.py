"""
FRC Match Sampler
=================
Sampling design:
  1. Stratified random sampling — strata defined by (year x week)
     Years : 2022–2026
     Weeks : 1–6, 7 (week 0 in TBA = preseason; week 8 = champs in some years)
             TBA uses event_type: 0=Regional,1=District,2=District Champ,
             3=Champ Division,4=Einstein/Champ Finals,5=OFFSEASON,6=Preseason
             Week field is 0-indexed (week 1 => week=0, etc.)
     Extra strata: 'dcmp' (district championships, between w7 and cmp)
                   'cmp'  (world championship divisions + Einstein)
  2. Cluster sampling — within each stratum, randomly select N_EVENTS events
  3. Simple random sampling — within each selected event, randomly select
     N_MATCHES matches

District championship finals events (same event_type=2 as divisions, but
typically only 2 matches) are detected by having fewer than DCMP_MIN_MATCHES
matches.  Their matches are folded into a randomly chosen division event's
pool before cluster+SRS sampling proceeds, so finals footage still has a
chance of being drawn without distorting the sample.

Stratum order (for reference):
  w1 → w2 → w3 → w4 → w5 → w6 → [w7] → dcmp → cmp

Output format (one record per match):
  { "match_key": "2022txdri_qm4", "videos": ["https://youtu.be/...", ...] }

Usage:
    export TBA_API_KEY="your_key_here"
    python frc_sampler.py
"""

import os
import json
import random
import time
from collections import defaultdict
from typing import Optional

import dotenv
import requests

# ── Constants (not user-tunable) ─────────────────────────────────────────────

dotenv.load_dotenv()
TBA_BASE    = "https://www.thebluealliance.com/api/v3"
API_KEY     = os.environ["TBA_API_KEY"]   # required — raises KeyError if absent
COMP_WEEKS  = list(range(7))              # TBA week 0-6 → comp weeks 1-7
CHAMP_TYPES = {3, 4}                      # Champ Division / Einstein Finals

# District champ events with fewer than this many matches are treated as
# finals events and folded into a division pool rather than sampled directly.
DCMP_MIN_MATCHES = 5

REQUEST_DELAY = 0.2   # seconds between TBA requests (be a good API citizen)

# Human-readable labels for comp_level codes
COMP_LEVEL_LABELS = {
    "qm": "Qualification",
    "sf": "Semifinal",
    "f":  "Final",
}

# ── Video URL builder ─────────────────────────────────────────────────────────

_VIDEO_PREFIXES = {
    "youtube":    "https://www.youtube.com/watch?v=",
    "tba":        "https://www.thebluealliance.com/gameday/",
    "livestream": "https://livestream.com/",
    "twitch":     "https://www.twitch.tv/videos/",
}

def video_url(v: dict) -> str:
    """Convert a TBA video object to a full URL string."""
    prefix = _VIDEO_PREFIXES.get(v.get("type", ""), "")
    return f"{prefix}{v.get('key', '')}" if prefix else v.get("key", "")

# ── TBA API helpers ───────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({"X-TBA-Auth-Key": API_KEY})

_etag_cache: dict[str, tuple[str, any]] = {}   # url -> (etag, data)


def tba_get(path: str, retries: int = 3) -> Optional[any]:
    """GET from TBA with ETag caching and simple retry logic."""
    url = f"{TBA_BASE}{path}"
    headers = {}
    if url in _etag_cache:
        headers["If-None-Match"] = _etag_cache[url][0]

    for attempt in range(retries):
        try:
            resp = SESSION.get(url, headers=headers, timeout=15)
        except requests.RequestException as exc:
            print(f"  [warn] network error ({exc}), retry {attempt+1}/{retries}")
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 304:                   # Not Modified
            return _etag_cache[url][1]
        if resp.status_code == 200:
            data = resp.json()
            etag = resp.headers.get("ETag", "")
            if etag:
                _etag_cache[url] = (etag, data)
            time.sleep(REQUEST_DELAY)
            return data
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            print(f"  [warn] rate-limited, sleeping {wait}s")
            time.sleep(wait)
            continue

        print(f"  [warn] HTTP {resp.status_code} for {path}")
        return None

    return None


# ── Stratum helpers ───────────────────────────────────────────────────────────

def stratum_key(year: int, week_label: str) -> str:
    return f"{year}_w{week_label}" if week_label.isdigit() else f"{year}_{week_label}"


def week_label(event: dict) -> Optional[str]:
    """
    Return a canonical week label for an event, or None to exclude it.
    Labels: '1'..'7' for regular-season weeks,
            'dcmp'   for district championships (type 2),
            'cmp'    for world championship divisions / Einstein (types 3, 4).
    """
    etype = event.get("event_type")
    week  = event.get("week")          # 0-indexed; None for champs

    if etype in CHAMP_TYPES:
        return "cmp"

    if etype == 2:
        return "dcmp"

    # Exclude offseason (5) and preseason (6)
    if etype in {5, 6}:
        return None

    if week is None:
        return None

    if week in COMP_WEEKS:             # 0-6 → labels 1-7
        return str(week + 1)

    return None                        # outside expected range → skip


# ── Match fetching ────────────────────────────────────────────────────────────

def fetch_matches(event_key: str) -> list[dict]:
    """Fetch and filter matches for an event to qual/semi/finals only."""
    matches = tba_get(f"/event/{event_key}/matches") or []
    return [m for m in matches if m.get("comp_level") in {"qm", "sf", "f"}]


def make_record(m: dict, event: dict) -> dict:
    """
    Build a slim output record for a match.

    Includes geographic metadata (country, state_prov) sourced from the
    parent event object, and the comp_level so downstream reports can
    break down by match type.
    """
    return {
        "match_key":  m["key"],
        "comp_level": m.get("comp_level", ""),
        "country":    event.get("country", ""),
        "state_prov": event.get("state_prov", ""),
        "videos":     [video_url(v) for v in m.get("videos", [])],
    }


# ── Main sampling logic ───────────────────────────────────────────────────────

def build_strata(years: list[int]) -> dict[str, list[dict]]:
    """
    Fetch all events for each year and bin them into strata.
    Returns { stratum_key: [event, ...] }
    """
    strata: dict[str, list[dict]] = defaultdict(list)

    for year in years:
        print(f"  Fetching events for {year}…")
        events = tba_get(f"/events/{year}") or []
        for ev in events:
            label = week_label(ev)
            if label is None:
                continue
            key = stratum_key(year, label)
            strata[key].append(ev)

    return dict(strata)


def cluster_sample_events(events: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Return up to n randomly chosen events (clusters) from the stratum."""
    if len(events) <= n:
        return events[:]
    return rng.sample(events, n)


def srs_matches(
    event: dict,
    n: int,
    rng: random.Random,
    extra_matches: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Fetch all matches for an event, optionally append extra_matches (folded
    finals), SRS n of them, and return slim records.
    """
    matches = fetch_matches(event["key"])
    if extra_matches:
        matches = matches + extra_matches
    if not matches:
        return []
    sampled = matches[:] if len(matches) <= n else rng.sample(matches, n)
    return [make_record(m, event) for m in sampled]


# ── DCMP stratum sampling (with finals folding) ───────────────────────────────

def sample_dcmp_stratum(
    events: list[dict],
    n_events: int,
    n_matches: int,
    rng: random.Random,
    year: int,
) -> list[dict]:
    """
    Handle one {year}_dcmp stratum:
      1. Fetch match counts for all events.
      2. Separate into divisions (≥ DCMP_MIN_MATCHES) and finals (< DCMP_MIN_MATCHES).
      3. Drop finals events — too few matches to sample meaningfully.
      4. Cluster-sample n_events divisions, then SRS n_matches from each.
    """
    # ── Step 1: fetch match lists for every dcmp event ────────────────────────
    event_matches: dict[str, list[dict]] = {}
    for ev in events:
        ekey = ev["key"]
        event_matches[ekey] = fetch_matches(ekey)

    # ── Step 2: split into divisions and finals ───────────────────────────────
    divisions = [ev for ev in events if len(event_matches[ev["key"]]) >= DCMP_MIN_MATCHES]
    finals    = [ev for ev in events if len(event_matches[ev["key"]]) <  DCMP_MIN_MATCHES]

    if finals:
        print(f"  [{year}_dcmp] Dropping {len(finals)} finals event(s) with fewer than {DCMP_MIN_MATCHES} matches:")
        for fev in finals:
            fkey = fev["key"]
            print(f"    {fkey} ({len(event_matches[fkey])} matches) → dropped")

    # ── Step 3: cluster-sample divisions, then SRS matches ───────────────────
    sampled_divisions = cluster_sample_events(divisions, n_events, rng)
    records: list[dict] = []
    for ev in sampled_divisions:
        ekey  = ev["key"]
        ename = ev.get("name", ekey)
        pool  = event_matches[ekey]
        print(f"  [{year}_dcmp] Event: {ename} ({ekey})  pool={len(pool)} matches")
        sampled = pool[:] if len(pool) <= n_matches else rng.sample(pool, n_matches)
        batch = [make_record(m, ev) for m in sampled]
        print(f"         → {len(batch)} match(es) sampled")
        records.extend(batch)

    return records


# ── Generic stratum sampling ──────────────────────────────────────────────────

def sample_stratum(
    key: str,
    events: list[dict],
    n_events: int,
    n_matches: int,
    rng: random.Random,
) -> list[dict]:
    """Cluster + SRS for a non-dcmp stratum."""
    sampled_events = cluster_sample_events(events, n_events, rng)
    records: list[dict] = []
    for ev in sampled_events:
        ekey  = ev["key"]
        ename = ev.get("name", ekey)
        print(f"  [{key}] Event: {ename} ({ekey})")
        matches = srs_matches(ev, n_matches, rng)
        print(f"         → {len(matches)} match(es) sampled")
        records.extend(matches)
    return records


# ── Top-level pipeline ────────────────────────────────────────────────────────

def sample_all(
    years: list[int],
    n_events: int,
    n_matches: int,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    """
    Full sampling pipeline.
    Returns (flat list of {match_key, videos} records, stratum_counts dict).
    """
    rng = random.Random(seed)

    print("\n=== Step 1: Building strata (year × week) ===")
    strata = build_strata(years)

    # Sort strata so they appear in chronological order:
    # w1..w7 numerically, then dcmp, then cmp — within each year.
    def sort_key(k: str) -> tuple:
        year, label = k.split("_", 1)
        order = {"dcmp": 8, "cmp": 9}
        # labels like 'w1'..'w7'
        if label.startswith("w") and label[1:].isdigit():
            return (int(year), int(label[1:]))
        return (int(year), order.get(label, 99))

    all_keys = sorted(strata.keys(), key=sort_key)
    print(f"  Found {len(all_keys)} strata with events:")
    for k in all_keys:
        print(f"    {k}: {len(strata[k])} events")

    records: list[dict] = []
    stratum_counts: dict[str, int] = {}

    print(f"\n=== Step 2 & 3: Cluster-sample {n_events} events, SRS {n_matches} matches ===")
    for key in all_keys:
        year_str, label = key.split("_", 1)
        year = int(year_str)

        if label == "dcmp":
            batch = sample_dcmp_stratum(
                strata[key], n_events, n_matches, rng, year
            )
        else:
            batch = sample_stratum(key, strata[key], n_events, n_matches, rng)

        records.extend(batch)
        stratum_counts[key] = len(batch)

    return records, stratum_counts


# ── Representation reports ────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "0.0%"


def report_geography(records: list[dict]) -> None:
    """
    Print a breakdown of sampled matches by country, and — for the USA —
    by state/province.

    Geographic metadata comes from the parent event's country and state_prov
    fields, attached to each record in make_record().
    """
    total = len(records)
    if not total:
        print("  (no records)")
        return

    by_country: dict[str, int] = defaultdict(int)
    us_by_state: dict[str, int] = defaultdict(int)

    for r in records:
        country    = r.get("country") or "Unknown"
        state_prov = r.get("state_prov") or "Unknown"
        by_country[country] += 1
        if country == "USA":
            us_by_state[state_prov] += 1

    print("\n--- Geographic representation ---")
    print(f"  {'Country':<30} {'Matches':>8}  {'Share':>7}")
    print(f"  {'-'*30} {'-'*8}  {'-'*7}")
    for country, n in sorted(by_country.items(), key=lambda x: -x[1]):
        print(f"  {country:<30} {n:>8}  {_pct(n, total):>7}")

    if us_by_state:
        print(f"\n  US breakdown by state/province ({by_country['USA']} matches):")
        print(f"    {'State':<28} {'Matches':>8}  {'Share of USA':>13}")
        print(f"    {'-'*28} {'-'*8}  {'-'*13}")
        us_total = by_country["USA"]
        for state, n in sorted(us_by_state.items(), key=lambda x: -x[1]):
            print(f"    {state:<28} {n:>8}  {_pct(n, us_total):>13}")


def report_match_types(records: list[dict]) -> None:
    """
    Print a breakdown of sampled matches by competition level
    (qualification, semifinal, final).
    """
    total = len(records)
    if not total:
        print("  (no records)")
        return

    by_level: dict[str, int] = defaultdict(int)
    for r in records:
        level = r.get("comp_level") or "unknown"
        by_level[level] += 1

    # Display in a logical competition order
    level_order = ["qm", "sf", "f"]
    other_levels = sorted(k for k in by_level if k not in level_order)

    print("\n--- Match-type representation ---")
    print(f"  {'Type':<22} {'Code':<6} {'Matches':>8}  {'Share':>7}")
    print(f"  {'-'*22} {'-'*6} {'-'*8}  {'-'*7}")
    for code in level_order + other_levels:
        if code not in by_level:
            continue
        n     = by_level[code]
        label = COMP_LEVEL_LABELS.get(code, code)
        print(f"  {label:<22} {code:<6} {n:>8}  {_pct(n, total):>7}")
    print(f"  {'TOTAL':<22} {'':6} {total:>8}  {'100.0%':>7}")


# ── Summary (strata counts + representation reports) ─────────────────────────

def summarise(stratum_counts: dict[str, int], records: list[dict]) -> None:
    print("\n=== Summary ===")
    total = 0
    for key in sorted(stratum_counts):
        n = stratum_counts[key]
        total += n
        print(f"  {key}: {n} matches")
    print(f"  TOTAL: {total} matches across {len(stratum_counts)} strata")

    report_geography(records)
    report_match_types(records)


def save_results(records: list[dict], path: str = "better_sampled_matches.json") -> None:
    # Strip internal-only fields before writing so the output schema stays stable.
    output = [
        {
            "match_key": r["match_key"],
            "videos":    r["videos"],
        }
        for r in records
    ]
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {path}  ({len(output)} records)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Tune sampling here ────────────────────────────────────────────────────

    # Which FRC seasons to include
    YEARS = list(range(2022, 2027))          # [2022, 2023, 2024, 2025, 2026]

    # How many events to cluster-sample per stratum (year × week)
    N_EVENTS = 2

    # How many matches to SRS per sampled event
    N_MATCHES = 3

    # Reproducibility seed — change to get a different random draw
    SEED = 422

    # Where to write the output JSON
    OUTPUT_PATH = "better_sampled_matches.json"

    # ── Run ───────────────────────────────────────────────────────────────────

    print("FRC Match Sampler")
    print(f"  Years           : {YEARS}")
    print(f"  Events/stratum  : {N_EVENTS}")
    print(f"  Matches/event   : {N_MATCHES}")
    print(f"  Random seed     : {SEED}")
    print(f"  Est. matches    : ~{len(YEARS) * N_EVENTS * N_MATCHES}")

    records, stratum_counts = sample_all(YEARS, N_EVENTS, N_MATCHES, SEED)
    summarise(stratum_counts, records)
    save_results(records, OUTPUT_PATH)

    if records:
        print(f"\nFirst record: {json.dumps({'match_key': records[0]['match_key'], 'videos': records[0]['videos']}, indent=2)}")