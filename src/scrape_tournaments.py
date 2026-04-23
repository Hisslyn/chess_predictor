#!/usr/bin/env python3
"""
src/scrape_tournaments.py

Downloads raw HTML for each tournament in data/interim/tournament_candidates.csv.

For each tournament ID, fetches 3 views:
  tnr{ID}_art1.html  – starting rank / player list
  tnr{ID}_art2.html  – pairings and results per round
  tnr{ID}_art5.html  – final crosstable with performance ratings

Saves HTML to data/raw/ and logs every attempt to data/interim/scrape_log.csv.
Idempotent: skips files that already exist.
"""

import csv
import logging
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
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
BASE_URL       = "https://s1.chess-results.com"
RATE_LIMIT_SEC = 1.5
RAW_DIR        = Path("data/raw")
CANDIDATES_CSV = Path("data/interim/tournament_candidates.csv")
SCRAPE_LOG     = Path("data/interim/scrape_log.csv")

ARTS = [1, 2, 5]   # art=1 player list, art=2 pairings, art=5 crosstable

USER_AGENT = (
    "ChessResearchBot/1.0 "
    "(academic ML research; "
    "contact: research@chess-predictor.local)"
)

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


_session = RateLimitedSession()

# ── log record ────────────────────────────────────────────────────────────────
@dataclass
class LogRecord:
    timestamp: str
    tournament_id: int
    art: int
    filename: str
    status: str        # "ok" | "skipped" | "error"
    bytes_saved: int
    error_msg: str


LOG_FIELDS = [f.name for f in fields(LogRecord)]


class ScrapeLogger:
    """Appends one CSV row per fetch attempt; creates header if file is new."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._is_new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=LOG_FIELDS)
        if self._is_new:
            self._writer.writeheader()

    def write(self, rec: LogRecord) -> None:
        self._writer.writerow(
            {f.name: getattr(rec, f.name) for f in fields(rec)}
        )
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ── fetch one view ─────────────────────────────────────────────────────────────
def fetch_view(tid: int, art: int, logger: ScrapeLogger) -> str:
    """
    Download one HTML view for a tournament.
    Returns "ok", "skipped", or "error".
    """
    filename = f"tnr{tid}_art{art}.html"
    out_path = RAW_DIR / filename
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if out_path.exists():
        log.info("  SKIP  %s (already exists)", filename)
        logger.write(LogRecord(
            timestamp=ts, tournament_id=tid, art=art,
            filename=filename, status="skipped",
            bytes_saved=out_path.stat().st_size, error_msg="",
        ))
        return "skipped"

    url = f"{BASE_URL}/tnr{tid}.aspx"
    try:
        resp = _session.get(url, params={"lan": 1, "art": art})
    except (requests.HTTPError, RetryError, Exception) as exc:
        msg = str(exc)
        log.warning("  ERROR tnr%d art=%d: %s", tid, art, msg)
        logger.write(LogRecord(
            timestamp=ts, tournament_id=tid, art=art,
            filename=filename, status="error",
            bytes_saved=0, error_msg=msg[:200],
        ))
        return "error"

    if "Record not found" in resp.text or len(resp.text) < 500:
        msg = "page too short or 'Record not found'"
        log.warning("  ERROR tnr%d art=%d: %s", tid, art, msg)
        logger.write(LogRecord(
            timestamp=ts, tournament_id=tid, art=art,
            filename=filename, status="error",
            bytes_saved=0, error_msg=msg,
        ))
        return "error"

    out_path.write_text(resp.text, encoding="utf-8")
    nbytes = out_path.stat().st_size
    log.info("  OK    %s  (%d bytes)", filename, nbytes)
    logger.write(LogRecord(
        timestamp=ts, tournament_id=tid, art=art,
        filename=filename, status="ok",
        bytes_saved=nbytes, error_msg="",
    ))
    return "ok"


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CANDIDATES_CSV)
    tournament_ids = df["tournament_id"].tolist()
    log.info("Loaded %d tournament IDs from %s", len(tournament_ids), CANDIDATES_CSV)

    logger = ScrapeLogger(SCRAPE_LOG)

    counts = {"ok": 0, "skipped": 0, "error": 0}
    total = len(tournament_ids) * len(ARTS)
    done = 0

    for tid in tournament_ids:
        log.info("── tnr%d ──────────────────────────────", tid)
        for art in ARTS:
            result = fetch_view(tid, art, logger)
            counts[result] += 1
            done += 1
            log.info("  Progress: %d/%d", done, total)

    logger.close()

    log.info("")
    log.info("═" * 60)
    log.info("Done.  ok=%d  skipped=%d  error=%d  (total=%d)",
             counts["ok"], counts["skipped"], counts["error"], total)
    log.info("Raw HTML → %s", RAW_DIR)
    log.info("Scrape log → %s", SCRAPE_LOG)
    log.info("═" * 60)

    # Print a summary of what landed in data/raw/
    files = sorted(RAW_DIR.glob("*.html"))
    log.info("Files in data/raw/ (%d):", len(files))
    for f in files:
        log.info("  %s  (%d bytes)", f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
