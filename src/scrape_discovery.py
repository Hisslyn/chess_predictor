#!/usr/bin/env python3
"""
src/scrape_discovery.py

Discovers valid Swiss tournament IDs on chess-results.com.
Targets: completed, FIDE-rated, Swiss-system, 6-11 rounds, 30+ players.
Writes: data/interim/tournament_candidates.csv

Strategy
--------
1. POST to TurnierSuche.aspx (ASP.NET form) with Swiss / finished / 2024
   filters to get a large list of IDs cheaply.
2. For each candidate ID, GET tnrXXXXXX.aspx?lan=1&art=0 (overview tab)
   and parse the metadata table for system, rounds, players, FIDE flag.
3. If the form approach fails or yields too few hits, fall back to a
   sequential ID probe over the 2024 range (~880 000 – 990 000, step 500).

Rate limit: 1.5 s between every HTTP request.
Server:     s1.chess-results.com (chess-results.com → 302 to s1.*)
"""

import logging
import re
import sys
import time
from dataclasses import dataclass
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
BASE_URL = "https://s1.chess-results.com"
SEARCH_URL = f"{BASE_URL}/TurnierSuche.aspx"

RATE_LIMIT_SEC = 1.5          # minimum gap between any two HTTP calls
TARGET_COUNT   = 10           # stop after collecting this many candidates
OUT_PATH       = Path("data/interim/tournament_candidates.csv")

USER_AGENT = (
    "ChessResearchBot/1.0 "
    "(academic ML research; "
    "contact: research@chess-predictor.local)"
)

# Probed 2024 ID boundaries:
#   tnr900000 → Mar 2024 (Cantabrian school championship)
#   tnr950000 → Jun 2024 (Orenburg rapid)
#   tnr1001000 → "Record not found"
# We probe with step=500 so a 110 000-wide window = ~220 HTTP calls max.
PROBE_RANGE = range(880_000, 1_000_000, 500)

# Criteria
ROUND_MIN    = 6
ROUND_MAX    = 11
PLAYERS_MIN  = 30


# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class TournamentCandidate:
    tournament_id: int
    name: str
    date: str
    n_rounds: int
    n_players: int
    country: str
    fide_rated: bool
    system: str
    url: str


# ── rate-limited, retrying HTTP session ───────────────────────────────────────
class RateLimitedSession:
    """requests.Session with a minimum inter-request delay and tenacity retries."""

    def __init__(self, rate_limit: float = RATE_LIMIT_SEC) -> None:
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": USER_AGENT})
        self._rate = rate_limit
        self._last: float = 0.0

    # ── internal ──────────────────────────────────────────────────────────
    def _throttle(self) -> None:
        wait = self._rate - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _handle(self, resp: requests.Response) -> requests.Response:
        resp.raise_for_status()
        return resp

    # ── public ────────────────────────────────────────────────────────────
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=3, max=60),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def get(self, url: str, **kw) -> requests.Response:
        self._throttle()
        return self._handle(self._s.get(url, timeout=25, **kw))

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=3, max=60),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def post(self, url: str, **kw) -> requests.Response:
        self._throttle()
        return self._handle(self._s.post(url, timeout=25, **kw))


_session = RateLimitedSession()


# ── search-form approach ──────────────────────────────────────────────────────
def _viewstate_fields(soup: BeautifulSoup) -> dict[str, str]:
    """Extract ASP.NET hidden form fields (__VIEWSTATE etc.)."""
    return {
        inp["name"]: inp.get("value", "")
        for inp in soup.select("input[type=hidden]")
        if inp.get("name", "").startswith("__")
    }


def search_swiss_2024(year: int = 2024, page_size_idx: int = 5) -> dict[int, str]:
    """
    Submit the TurnierSuche form with Swiss-System + finished + year filter.

    Returns a dict  { tournament_id: start_date_string }  for all results.
    Dates are extracted from the results table columns (Start-Date column).

    Actual field names from page inspection (ASP.NET ctl00$P1$... prefix):
      combo_art         – tournament system: '0' = Swiss-System
      cbox_zuEnde       – only finished tournaments (checkbox, send 'on')
      txt_von_tag       – start date range from  (YYYY-MM-DD)
      txt_bis_tag       – start date range to    (YYYY-MM-DD)
      combo_anzahl_zeilen – results per page: '5' = 2000
      cb_suchen         – submit button
    """
    log.info("Fetching search form …")
    resp = _session.get(SEARCH_URL, params={"lan": 1})
    soup = BeautifulSoup(resp.text, "lxml")
    hidden = _viewstate_fields(soup)
    log.info("  ViewState fields: %s", list(hidden.keys()))

    payload = {
        **hidden,
        "lan": "1",
        "ctl00$P1$combo_art": "0",           # Swiss-System
        "ctl00$P1$cbox_zuEnde": "on",         # only finished
        "ctl00$P1$txt_von_tag": f"{year}-01-01",
        "ctl00$P1$txt_bis_tag": f"{year}-12-31",
        "ctl00$P1$combo_sort": "1",           # sort by last update
        "ctl00$P1$combo_land": "-",           # all countries
        "ctl00$P1$combo_bedenkzeit": "0",     # all time controls
        "ctl00$P1$combo_anzahl_zeilen": str(page_size_idx),  # 5 = 2000 rows
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "ctl00$P1$cb_suchen": "Search",
    }

    log.info("Submitting search (year=%d, system=Swiss, finished) …", year)
    resp = _session.post(SEARCH_URL, data=payload, params={"lan": 1})
    soup = BeautifulSoup(resp.text, "lxml")

    # Parse the results table.
    # Observed column order (sorted by Start-Date):
    #   No. | Tournament | [flag] | [state] | Last-update | Start-Date |
    #   End-Date | Director | Organizer | Chief Arbiter | Deputy |
    #   Arbiter | Location | Time control | FED | State | n | dbkey | EventID
    #
    # Dates appear as YYYY/MM/DD.
    # "n"     column = number of rounds.
    # "dbkey" column = number of players  (small integer, NOT the tnr ID).
    # The tnrXXXXXX ID comes from the <a href> in the Tournament cell.
    result: dict[int, str] = {}
    for row in soup.select("tr"):
        cells = row.find_all("td")
        # Find the cell with a tournament link
        tid: Optional[int] = None
        start_date = ""
        for cell in cells:
            a = cell.find("a", href=re.compile(r"tnr\d+\.aspx", re.I))
            if a:
                m = re.search(r"tnr(\d+)\.aspx", a["href"], re.I)
                if m:
                    tid = int(m.group(1))
            # Dates appear as YYYY/MM/DD
            txt = cell.get_text(strip=True)
            if re.match(r"^\d{4}/\d{2}/\d{2}$", txt) and not start_date:
                # Convert to DD.MM.YYYY for consistency
                y, mo, d = txt.split("/")
                start_date = f"{d}.{mo}.{y}"
        if tid is not None:
            result[tid] = start_date

    if not result:
        _debug = Path("data/interim/search_debug.html")
        _debug.parent.mkdir(parents=True, exist_ok=True)
        _debug.write_text(resp.text, encoding="utf-8")
        log.warning(
            "Search returned 0 IDs — saved raw response to %s", _debug
        )
    log.info("Search returned %d unique tournament IDs", len(result))
    return result


# ── tournament-page parsing ───────────────────────────────────────────────────
_INT_RE = re.compile(r"\d+")


def _first_int(text: str) -> Optional[int]:
    m = _INT_RE.search(text)
    return int(m.group()) if m else None


def _fetch_soup(tid: int, art: int) -> Optional[tuple[str, BeautifulSoup]]:
    """Fetch tnrXXX.aspx?lan=1&art=N. Returns (raw_text, soup) or None."""
    url = f"{BASE_URL}/tnr{tid}.aspx"
    try:
        resp = _session.get(url, params={"lan": 1, "art": art})
    except (requests.HTTPError, RetryError) as exc:
        log.debug("tnr%d art=%d: HTTP error – %s", tid, art, exc)
        return None
    if "Record not found" in resp.text or len(resp.text) < 500:
        return None
    return resp.text, BeautifulSoup(resp.text, "lxml")


def parse_tournament_page(tid: int) -> Optional[TournamentCandidate]:
    """
    Two-request parse strategy:

    Request 1  →  art=4 (final-ranking crosstable)
      • "Final Ranking crosstable after N Rounds"  → round count
      • "N.Rd" column headers                      → fallback round count
      • Rank-integer rows                          → player count
      • Page text                                  → system type, date

    Request 2  →  art=0 (starting rank list)  — only for candidates that
      pass the preliminary rounds+players check, to keep request count low.
      • "FideID" column header or ≥10 FIDE-ID cells → fide_rated
      • Player federation codes                    → country (majority)

    Returns None if the page doesn't exist or is unparseable.
    """
    # ── Request 1: art=4 ─────────────────────────────────────────────────
    result = _fetch_soup(tid, art=4)
    if result is None:
        return None
    text4, soup4 = result

    # Tournament name: h2 → h1 → <title>
    name = ""
    for selector in ("h2", "h1"):
        tag = soup4.select_one(selector)
        if tag:
            name = tag.get_text(" ", strip=True)
            break
    if not name:
        t = soup4.find("title")
        if t:
            name = t.get_text(strip=True).split(" - ")[0].strip()
    name = name or f"Tournament {tid}"

    page_text4 = soup4.get_text(" ")

    # System: look for explicit keywords in page text
    if re.search(r"swiss", page_text4, re.I):
        system_raw = "Swiss"
    elif re.search(r"round.?robin", page_text4, re.I):
        system_raw = "Round robin"
    else:
        system_raw = "Swiss"  # art=4 per-round crosstable implies Swiss

    # Rounds: "after N Rounds" subtitle  (primary)
    m_rounds = re.search(r"after\s+(\d+)\s+[Rr]ound", page_text4)
    n_rounds: Optional[int] = int(m_rounds.group(1)) if m_rounds else None

    if n_rounds is None:
        # Fallback: count "N.Rd" column headers
        rd_headers = [
            th.get_text(strip=True) for th in soup4.find_all("th")
            if re.match(r"^\d+\.Rd$", th.get_text(strip=True))
        ]
        n_rounds = len(rd_headers) if rd_headers else None

    # Players: count rows whose first <td> is a rank integer
    n_players: Optional[int] = (
        sum(
            1 for row in soup4.select("table tr")
            if (c := row.find("td"))
            and re.match(r"^\d+$", c.get_text(strip=True))
        ) or None
    )

    # Date: find a DD.MM.YYYY that is NOT part of a "Last update" line.
    # Strip "Last update ..." lines first, then take the first remaining date.
    text4_nodates = re.sub(
        r"Last update[^\n]*\d{2}\.\d{2}\.\d{4}[^\n]*", "", text4, flags=re.I
    )
    m_date = re.search(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b", text4_nodates)
    date_raw = m_date.group(1) if m_date else ""

    # ── preliminary filter: avoid art=0 fetch for obvious misses ─────────
    # We still need FIDE from art=0, but only bother if rounds+players pass.
    passes_preliminary = (
        system_raw.lower().startswith("swiss")
        and n_rounds is not None
        and ROUND_MIN <= n_rounds <= ROUND_MAX
        and n_players is not None
        and n_players >= PLAYERS_MIN
    )

    # ── Request 2: art=0 (only for preliminary passes) ───────────────────
    fide_rated = False
    country_raw = ""

    if passes_preliminary:
        result0 = _fetch_soup(tid, art=0)
        if result0:
            text0, soup0 = result0
            page_text0 = soup0.get_text(" ")

            # FideID column header = strong FIDE signal
            has_fide_col = bool(
                soup0.find("th", string=re.compile(r"FideID", re.I))
            )
            # Many 5-9-digit cells = FIDE IDs in the table
            fide_id_cells = [
                td for td in soup0.find_all("td")
                if re.match(r"^\d{5,9}$", td.get_text(strip=True))
            ]
            # Explicit mention of FIDE evaluation in page text
            fide_text = bool(
                re.search(r"fide\s+(elo|rapid|blitz|evaluation)", page_text0, re.I)
            )
            fide_rated = has_fide_col or len(fide_id_cells) >= 10 or fide_text

            # Country: majority federation code among players
            fed_cells = [
                td.get_text(strip=True) for td in soup0.find_all("td")
                if re.match(r"^[A-Z]{2,3}$", td.get_text(strip=True))
            ]
            if fed_cells:
                from collections import Counter
                country_raw = Counter(fed_cells).most_common(1)[0][0]

    return TournamentCandidate(
        tournament_id=tid,
        name=name,
        date=date_raw,
        n_rounds=n_rounds or 0,
        n_players=n_players or 0,
        country=country_raw,
        fide_rated=fide_rated,
        system=system_raw,
        url=f"{BASE_URL}/tnr{tid}.aspx?lan=1",
    )


# ── criteria check ────────────────────────────────────────────────────────────
def meets_criteria(t: TournamentCandidate) -> bool:
    # Note: fide_rated is only checked via art=0 when rounds+players already pass,
    # so this check is always meaningful when called after parse_tournament_page.
    return (
        bool(re.search(r"swiss", t.system, re.I))
        and ROUND_MIN <= t.n_rounds <= ROUND_MAX
        and t.n_players >= PLAYERS_MIN
        and t.fide_rated
    )


def _log_candidate(t: TournamentCandidate, verdict: str) -> None:
    log.info(
        "  tnr%-7d  rnd=%-2s  ply=%-4s  fide=%-5s  sys=%-20s  → %s",
        t.tournament_id,
        t.n_rounds or "?",
        t.n_players or "?",
        t.fide_rated,
        (t.system or "?")[:20],
        verdict,
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    passing: list[TournamentCandidate] = []

    # ── Strategy 1: search form ───────────────────────────────────────────
    search_map: dict[int, str] = {}  # {tid: start_date}
    try:
        search_map = search_swiss_2024(year=2024)
    except Exception as exc:
        log.warning("Search form strategy failed (%s); will probe IDs directly.", exc)

    if search_map:
        log.info("Checking %d search-result IDs for criteria …", len(search_map))
        for tid, search_date in search_map.items():
            if len(passing) >= TARGET_COUNT:
                break
            t = parse_tournament_page(tid)
            if t is None:
                continue
            # Override date with the accurate value from the search results table
            if search_date:
                t.date = search_date
            verdict = "PASS" if meets_criteria(t) else "skip"
            _log_candidate(t, verdict)
            if verdict == "PASS":
                passing.append(t)

    # ── Strategy 2: sequential ID probe ──────────────────────────────────
    if len(passing) < TARGET_COUNT:
        log.info(
            "Have %d/%d; switching to sequential probe (step=%d) …",
            len(passing), TARGET_COUNT, PROBE_RANGE.step,
        )
        for tid in PROBE_RANGE:
            if len(passing) >= TARGET_COUNT:
                break
            t = parse_tournament_page(tid)
            if t is None:
                log.debug("tnr%d: not found / no data", tid)
                continue
            verdict = "PASS" if meets_criteria(t) else "skip"
            _log_candidate(t, verdict)
            if verdict == "PASS":
                passing.append(t)

    # ── save ──────────────────────────────────────────────────────────────
    if not passing:
        log.error(
            "No qualifying tournaments found. "
            "Check: network access, ID range, and field-name assumptions."
        )
        sys.exit(1)

    df = pd.DataFrame([vars(t) for t in passing])
    df.to_csv(OUT_PATH, index=False)

    log.info("Saved %d candidates → %s", len(df), OUT_PATH)
    print("\n" + "=" * 72)
    print(df.to_string(index=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
