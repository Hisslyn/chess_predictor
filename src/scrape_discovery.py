#!/usr/bin/env python3
"""
src/scrape_discovery.py

Discovers valid Swiss tournament IDs on chess-results.com.

Targets: completed, FIDE-rated, Swiss-system, classical OR rapid,
         6–11 rounds, 30+ players, standard chess (not xiangqi/blitz),
         avg rating > 1500, ≥ 80 % of players with non-zero FIDE ratings.

Writes: data/interim/tournament_candidates.csv

Strategy
--------
1. POST to TurnierSuche.aspx twice — once for Standard (bedenkzeit=1) and
   once for Rapid (bedenkzeit=2).  Request up to 2 000 results per search,
   sorted by Start-Date so we cover the full year evenly.

2. Pre-filter names from the search table before fetching any pages:
   skip tournaments matching EXCLUSION_RE (blitz, school, test, xiangqi …).

3. For each surviving ID, GET tnr{ID}.aspx?lan=1&art=4 (crosstable) to
   check rounds / player count / system.

4. GET art=0 (starting-rank list) for preliminary passes:
   check FIDE-ID coverage, avg rating, time-control text.

5. Fall back to sequential ID probe if search yields < TARGET_COUNT.

Rate limit: 1.5 s between every HTTP request.
"""

import logging
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
BASE_URL   = "https://s1.chess-results.com"
SEARCH_URL = f"{BASE_URL}/TurnierSuche.aspx"

RATE_LIMIT_SEC = 1.5
TARGET_COUNT   = 150
OUT_PATH       = Path("data/interim/tournament_candidates.csv")

USER_AGENT = (
    "ChessResearchBot/1.0 "
    "(academic ML research; "
    "contact: research@chess-predictor.local)"
)

# ── search IDs for the sequential fallback probe ──────────────────────────────
# Approximate ID ranges by year:
#   ~700 000 = early 2023   ~800 000 = mid 2023   ~900 000 = late 2023
#   ~920 000 = early 2024   ~1 000 000 = Oct 2024  ~1 090 000 = Dec 2024
#   ~1 090 000 = early 2025 ~1 150 000 = mid 2025
PROBE_RANGES = [
    range(1_000_000, 1_090_000, 150), # Q4 2024
    range(920_000,   1_000_000, 150), # mid-2024
    range(800_000,   920_000,   200), # 2023
    range(1_090_000, 1_180_000, 150), # 2025
]

# ── quality criteria ──────────────────────────────────────────────────────────
ROUND_MIN    = 6
ROUND_MAX    = 11
PLAYERS_MIN  = 30
AVG_RTG_MIN  = 1800
PCT_RATED_MIN = 0.80   # ≥ 80 % of players must have a non-zero FIDE rating

# ── name pre-filter (applied before any HTTP page fetch) ─────────────────────
# Excludes: blitz/bullet events, school/youth/club-local, test data,
#           xiangqi / Chinese chess, disability categories, children categories
EXCLUSION_RE = re.compile(
    r"""
    \btest\b                        # TEST tournament
    | school | schule | escuela     # school events
    | gymnáz | gymnase | gymnasium  # school (gymnázia etc.)
    | \bjunior | \byouth | \bu\s*\d{1,2}\b   # youth categories
    | \bchildren | \bkids\b | \bkinder\b
    | blitz | bullet | relámpago | blic | blitts
    | lightning | schnell           # lightning chess / blitz
    | \bopen\s+blitz | blitz\s+open
    | xiangqi | chinese\s+chess
    | cờ\s*tướng | cờ\s*vua        # Vietnamese chess names
    | tướng | xadrez\s+rápido
    | deaf | ciegos | blind | disabled | special\s+needs
    | mini | mikro                  # miniatures / mini tournaments
    | simultan | simul\b            # simuls
    """,
    re.I | re.VERBOSE,
)

# Also exclude if name contains very short time controls (blitz indicator)
BLITZ_TC_RE = re.compile(
    r"\b[1-9]\s*min\b|\b\d\s*\+\s*[0-3]\b",   # "5min", "3+2", etc.
    re.I,
)


# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class TournamentCandidate:
    tournament_id:    int
    name:             str
    date:             str
    n_rounds:         int
    n_players:        int
    country:          str
    fide_rated:       bool
    system:           str
    time_control:     str
    avg_rating:       float
    pct_rated:        float
    url:              str


# ── rate-limited, retrying HTTP session ───────────────────────────────────────
class RateLimitedSession:
    def __init__(self, rate_limit: float = RATE_LIMIT_SEC) -> None:
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": USER_AGENT})
        self._rate = rate_limit
        self._last: float = 0.0

    def _throttle(self) -> None:
        wait = self._rate - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=3, max=60),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def get(self, url: str, **kw) -> requests.Response:
        self._throttle()
        resp = self._s.get(url, timeout=25, **kw)
        resp.raise_for_status()
        return resp

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=3, max=60),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def post(self, url: str, **kw) -> requests.Response:
        self._throttle()
        resp = self._s.post(url, timeout=25, **kw)
        resp.raise_for_status()
        return resp


_session = RateLimitedSession()


# ── search-form approach ──────────────────────────────────────────────────────
def _viewstate_fields(soup: BeautifulSoup) -> dict[str, str]:
    return {
        inp["name"]: inp.get("value", "")
        for inp in soup.select("input[type=hidden]")
        if inp.get("name", "").startswith("__")
    }


def search_swiss(
    year: int = 2024,
    bedenkzeit: int = 1,      # 1=Standard, 2=Rapid, 0=all
    page_size_idx: int = 5,   # 5 = 2000 rows
) -> dict[int, tuple[str, str]]:
    """
    POST TurnierSuche form.
    Returns { tournament_id: (start_date, tournament_name) }.
    Applies EXCLUSION_RE name pre-filter before returning.
    """
    tc_label = {0: "all", 1: "Standard", 2: "Rapid"}.get(bedenkzeit, str(bedenkzeit))
    log.info("Searching: year=%d  time_control=%s  …", year, tc_label)

    resp = _session.get(SEARCH_URL, params={"lan": 1})
    soup = BeautifulSoup(resp.text, "lxml")
    hidden = _viewstate_fields(soup)

    payload = {
        **hidden,
        "lan": "1",
        "ctl00$P1$combo_art":          "0",            # Swiss-System
        "ctl00$P1$cbox_zuEnde":        "on",           # finished only
        "ctl00$P1$txt_von_tag":        f"{year}-01-01",
        "ctl00$P1$txt_bis_tag":        f"{year}-12-31",
        "ctl00$P1$combo_sort":         "3",            # sort by Start-Date
        "ctl00$P1$combo_land":         "-",            # all countries
        "ctl00$P1$combo_bedenkzeit":   str(bedenkzeit),
        "ctl00$P1$combo_anzahl_zeilen": str(page_size_idx),
        "__EVENTTARGET":  "",
        "__EVENTARGUMENT": "",
        "ctl00$P1$cb_suchen": "Search",
    }

    resp = _session.post(SEARCH_URL, data=payload, params={"lan": 1})
    soup = BeautifulSoup(resp.text, "lxml")

    result: dict[int, tuple[str, str]] = {}
    excluded = 0
    for row in soup.select("tr"):
        cells = row.find_all("td")
        tid: Optional[int] = None
        name = ""
        start_date = ""
        for cell in cells:
            a = cell.find("a", href=re.compile(r"tnr\d+\.aspx", re.I))
            if a:
                m = re.search(r"tnr(\d+)\.aspx", a["href"], re.I)
                if m:
                    tid = int(m.group(1))
                    name = a.get_text(strip=True)
            txt = cell.get_text(strip=True)
            if re.match(r"^\d{4}/\d{2}/\d{2}$", txt) and not start_date:
                y, mo, d = txt.split("/")
                start_date = f"{d}.{mo}.{y}"

        if tid is None:
            continue

        # Name pre-filter
        if EXCLUSION_RE.search(name) or BLITZ_TC_RE.search(name):
            log.debug("  name-filter skip: %s", name)
            excluded += 1
            continue

        result[tid] = (start_date, name)

    log.info(
        "  → %d IDs returned, %d excluded by name filter, %d surviving",
        len(result) + excluded, excluded, len(result),
    )
    return result


# ── tournament-page parsing ───────────────────────────────────────────────────
_INT_RE = re.compile(r"\d+")


def _first_int(text: str) -> Optional[int]:
    m = _INT_RE.search(text)
    return int(m.group()) if m else None


def _fetch_soup(
    tid: int, art: int
) -> Optional[tuple[str, BeautifulSoup]]:
    url = f"{BASE_URL}/tnr{tid}.aspx"
    try:
        resp = _session.get(url, params={"lan": 1, "art": art})
    except (requests.HTTPError, RetryError) as exc:
        log.debug("tnr%d art=%d: HTTP error – %s", tid, art, exc)
        return None
    if "Record not found" in resp.text or len(resp.text) < 500:
        return None
    return resp.text, BeautifulSoup(resp.text, "lxml")


def parse_tournament_page(
    tid: int,
    search_name: str = "",
    search_date: str = "",
) -> Optional[TournamentCandidate]:
    """
    Two-request strategy (art=4, then art=0) with strict quality gates.

    Filters applied here (after name pre-filter already passed):
      F1  FIDE-rated  – fide_id_cells / n_players ≥ PCT_RATED_MIN
      F3  Time control – no blitz keywords in page text
      F5  Avg rating  > AVG_RTG_MIN
      F6  Pct rated   ≥ PCT_RATED_MIN  (non-zero starting_rating)
    """
    # ── Request 1: art=4 ─────────────────────────────────────────────────
    result4 = _fetch_soup(tid, art=4)
    if result4 is None:
        return None
    text4, soup4 = result4

    # Name from page (fallback to search name)
    name = search_name
    if not name:
        for sel in ("h2", "h1"):
            tag = soup4.select_one(sel)
            if tag:
                name = tag.get_text(" ", strip=True)
                break
        if not name:
            t = soup4.find("title")
            if t:
                name = t.get_text(strip=True).split(" - ")[-1].strip()
    name = name or f"Tournament {tid}"

    # Apply name filters again (the name on the page may differ from search)
    if EXCLUSION_RE.search(name) or BLITZ_TC_RE.search(name):
        log.debug("tnr%d: name-filtered at art=4 stage (%s)", tid, name[:60])
        return None

    page_text4 = soup4.get_text(" ")

    # System
    if re.search(r"round.?robin", page_text4, re.I):
        system_raw = "Round robin"
    else:
        system_raw = "Swiss"

    # Rounds
    m_rnd = re.search(r"after\s+(\d+)\s+[Rr]ound", page_text4)
    n_rounds: Optional[int] = int(m_rnd.group(1)) if m_rnd else None
    if n_rounds is None:
        rd_hdrs = [
            th.get_text(strip=True) for th in soup4.find_all("th")
            if re.match(r"^\d+\.Rd$", th.get_text(strip=True))
        ]
        n_rounds = len(rd_hdrs) if rd_hdrs else None

    # Players
    n_players: Optional[int] = (
        sum(
            1 for row in soup4.select("table tr")
            if (c := row.find("td"))
            and re.match(r"^\d+$", c.get_text(strip=True))
        ) or None
    )

    # Date
    page_no_lu = re.sub(
        r"Last update[^\n]*\d{2}\.\d{2}\.\d{4}[^\n]*", "", text4, flags=re.I
    )
    m_date = re.search(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b", page_no_lu)
    date_raw = search_date or (m_date.group(1) if m_date else "")

    # Preliminary check: rounds + players + system
    if (
        system_raw.lower() != "swiss"
        or n_rounds is None
        or not (ROUND_MIN <= n_rounds <= ROUND_MAX)
        or n_players is None
        or n_players < PLAYERS_MIN
    ):
        return None

    # ── Request 2: art=0 ─────────────────────────────────────────────────
    result0 = _fetch_soup(tid, art=0)
    if result0 is None:
        return None
    text0, soup0 = result0
    page_text0 = soup0.get_text(" ")

    # F3: time-control blitz check via page text
    if re.search(r"\bblitz\b|\bbullet\b|\brelámpago\b", page_text0, re.I):
        log.debug("tnr%d: blitz keyword in page text", tid)
        return None
    # Also check for very short time controls in page text
    tc_match = re.search(
        r"(\d+)\s*min(?:utes?)?\s*/?\s*(?:game|partie|partida)?",
        page_text0, re.I
    )
    if tc_match:
        tc_mins = int(tc_match.group(1))
        if tc_mins < 25:   # anything < 25 min per player = blitz territory
            log.debug("tnr%d: short time control %d min", tid, tc_mins)
            return None

    # F1 & F6: FIDE ID coverage and rating quality
    # Collect all player ratings from starting-rank table
    player_ratings: list[float] = []
    fide_id_count = 0
    total_players_in_table = 0

    # Identify player rows
    for tr in soup0.find_all("tr"):
        cls = tr.get("class", [])
        if not any(re.match(r"CRg[12]$|CRng[12]$", c) for c in cls):
            continue
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 4:
            continue
        total_players_in_table += 1

        # Collect numeric cells that look like FIDE IDs (5-9 digits)
        for c in cells:
            if re.match(r"^\d{5,9}$", c):
                fide_id_count += 1
                break  # one per player

        # Collect rating: find the first 4-digit number that looks like an ELO
        for c in cells:
            if re.match(r"^\d{4}$", c):
                r = int(c)
                if 800 <= r <= 3000:
                    player_ratings.append(float(r))
                    break

    if total_players_in_table == 0:
        return None

    pct_fide   = fide_id_count / total_players_in_table
    pct_rated  = len(player_ratings) / total_players_in_table
    avg_rating = sum(player_ratings) / len(player_ratings) if player_ratings else 0.0

    # F1: FIDE coverage
    if pct_fide < PCT_RATED_MIN:
        log.debug(
            "tnr%d: FIDE coverage %.0f%% < %.0f%%",
            tid, pct_fide * 100, PCT_RATED_MIN * 100,
        )
        return None

    # F5: avg rating
    if avg_rating < AVG_RTG_MIN:
        log.debug("tnr%d: avg rating %.0f < %d", tid, avg_rating, AVG_RTG_MIN)
        return None

    # F6: pct with non-zero rating
    if pct_rated < PCT_RATED_MIN:
        log.debug(
            "tnr%d: rated %.0f%% < %.0f%%",
            tid, pct_rated * 100, PCT_RATED_MIN * 100,
        )
        return None

    # Country: majority federation code
    fed_cells = [
        td.get_text(strip=True) for td in soup0.find_all("td")
        if re.match(r"^[A-Z]{2,3}$", td.get_text(strip=True))
    ]
    country = Counter(fed_cells).most_common(1)[0][0] if fed_cells else ""

    # Time control label from page or inferred from search
    tc_label = ""
    for tc in ("Standard", "Classical", "Rapid", "Schnell"):
        if re.search(tc, page_text0, re.I):
            tc_label = tc
            break

    return TournamentCandidate(
        tournament_id=tid,
        name=name,
        date=date_raw,
        n_rounds=n_rounds,
        n_players=n_players,
        country=country,
        fide_rated=True,          # guaranteed by pct_fide check above
        system=system_raw,
        time_control=tc_label,
        avg_rating=round(avg_rating, 1),
        pct_rated=round(pct_rated, 3),
        url=f"{BASE_URL}/tnr{tid}.aspx?lan=1",
    )


# ── criteria check (final gate) ───────────────────────────────────────────────
def meets_criteria(t: TournamentCandidate) -> tuple[bool, list[str]]:
    """
    Returns (pass, [list of failure reasons]).
    All filter logic is already enforced in parse_tournament_page;
    this is the final explicit gate logged for transparency.
    """
    fails: list[str] = []
    if not re.search(r"swiss", t.system, re.I):
        fails.append(f"system={t.system}")
    if not (ROUND_MIN <= t.n_rounds <= ROUND_MAX):
        fails.append(f"rounds={t.n_rounds}")
    if t.n_players < PLAYERS_MIN:
        fails.append(f"players={t.n_players}")
    if not t.fide_rated:
        fails.append("not-FIDE-rated")
    if t.avg_rating < AVG_RTG_MIN:
        fails.append(f"avg_rtg={t.avg_rating:.0f}")
    if t.pct_rated < PCT_RATED_MIN:
        fails.append(f"pct_rated={t.pct_rated:.0%}")
    return len(fails) == 0, fails


def _log_candidate(t: TournamentCandidate, verdict: str, fails: list[str]) -> None:
    log.info(
        "  tnr%-8d  rnd=%-2s  ply=%-4s  avg_rtg=%-4.0f  rated=%-4.0f%%  tc=%-10s  → %s%s",
        t.tournament_id,
        t.n_rounds or "?",
        t.n_players or "?",
        t.avg_rating,
        t.pct_rated * 100,
        (t.time_control or "?")[:10],
        verdict,
        f"  ({', '.join(fails)})" if fails else "",
    )


def _save_incremental(passing: list[TournamentCandidate]) -> None:
    df = pd.DataFrame([vars(t) for t in passing])
    df.to_csv(OUT_PATH, index=False)
    log.info("  ↳ checkpoint: %d candidates saved to %s", len(df), OUT_PATH)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    passing: list[TournamentCandidate] = []
    seen_tids: set[int] = set()

    # Resume from existing candidates if the file exists
    if OUT_PATH.exists():
        try:
            existing = pd.read_csv(OUT_PATH)
            fields_needed = [f.name for f in TournamentCandidate.__dataclass_fields__.values()]
            for _, row in existing.iterrows():
                kwargs = {k: row[k] for k in fields_needed if k in existing.columns}
                tc = TournamentCandidate(**kwargs)
                passing.append(tc)
                seen_tids.add(int(row["tournament_id"]))
            log.info("Resumed with %d existing candidates from %s", len(passing), OUT_PATH)
        except Exception as exc:
            log.warning("Could not load existing candidates: %s — starting fresh", exc)

    # ── Strategy 1: search form (Standard + Rapid, 2025/2024/2023) ───────
    search_maps: list[dict[int, tuple[str, str]]] = []
    for year in (2025, 2024, 2023):
        for bedenkzeit in (1, 2):   # Standard then Rapid
            if len(passing) >= TARGET_COUNT:
                break
            try:
                sm = search_swiss(year=year, bedenkzeit=bedenkzeit)
                search_maps.append(sm)
            except Exception as exc:
                log.warning("Search year=%d tc=%d failed: %s", year, bedenkzeit, exc)

    for sm in search_maps:
        for tid, (search_date, search_name) in sm.items():
            if len(passing) >= TARGET_COUNT:
                break
            if tid in seen_tids:
                continue
            seen_tids.add(tid)

            t = parse_tournament_page(tid, search_name=search_name, search_date=search_date)
            if t is None:
                continue
            ok, fails = meets_criteria(t)
            _log_candidate(t, "PASS" if ok else "skip", fails)
            if ok:
                passing.append(t)
                _save_incremental(passing)

    # ── Strategy 2: sequential ID probe ──────────────────────────────────
    if len(passing) < TARGET_COUNT:
        log.info(
            "Have %d/%d; switching to sequential probe …",
            len(passing), TARGET_COUNT,
        )
        for probe_range in PROBE_RANGES:
            if len(passing) >= TARGET_COUNT:
                break
            for tid in probe_range:
                if len(passing) >= TARGET_COUNT:
                    break
                if tid in seen_tids:
                    continue
                seen_tids.add(tid)

                t = parse_tournament_page(tid)
                if t is None:
                    log.debug("tnr%d: not found / filtered", tid)
                    continue
                ok, fails = meets_criteria(t)
                _log_candidate(t, "PASS" if ok else "skip", fails)
                if ok:
                    passing.append(t)
                    if len(passing) % 10 == 0:
                        _save_incremental(passing)

    # ── final save ────────────────────────────────────────────────────────
    if not passing:
        log.error("No qualifying tournaments found.")
        sys.exit(1)

    df = pd.DataFrame([vars(t) for t in passing])
    df.to_csv(OUT_PATH, index=False)

    log.info("Saved %d candidates → %s", len(df), OUT_PATH)
    print("\n" + "=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
