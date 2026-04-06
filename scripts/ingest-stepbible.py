"""Download STEPBible and TSK data, then index into LanceDB via BibleRAG.

Data sources (all CC BY 4.0):
  STEPBible-Data: https://github.com/STEPBible/STEPBible-Data
  TSK (Treasury of Scripture Knowledge): public domain cross-reference data

Pipeline:
  1. Download TIPNR, NASB, and ESV datasets from STEPBible-Data GitHub
  2. Download TSK cross-references from STEPBible-Data
  3. Convert to CSV format expected by BibleRAG (columns: OsisRef, EnglishText)
  4. Index into LanceDB via BibleRAG.index_stepbible()

Usage:
    uv run python scripts/ingest-stepbible.py
    uv run python scripts/ingest-stepbible.py --data-dir data/stepbible --skip-download
    uv run python scripts/ingest-stepbible.py --dry-run  # download only, no embed
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import httpx

# Use stdlib logging for the script header; structlog picks up after setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── STEPBible-Data GitHub raw URLs ────────────────────────────────────────────

_REPO_RAW = "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/master"

# TIPNR Tagged texts — ESV (English Standard Version)
_TIPNR_NT = (  # noqa: E501
    f"{_REPO_RAW}/Translators%20Amalgamated%20NT/"
    "TANTT%20-%20Translators%20Amalgamated%20NT%20-%20STEPBible.org%20CC%20BY.txt"
)
_TIPNR_OT = (  # noqa: E501
    f"{_REPO_RAW}/Translators%20Amalgamated%20OT/"
    "TOTHT%20-%20Translators%20Amalgamated%20OT%20Heb%20%26%20Tran%20-%20STEPBible.org%20CC%20BY.txt"
)

# NASB Simple (plain text, each verse on a line)
_NASB_ZIP = (  # noqa: E501
    f"{_REPO_RAW}/NASB%20plain%20text/"
    "NASB%20-%20New%20American%20Standard%20Bible%20-%20STEPBible.org%20CC%20BY.zip"
)

# English translation (ESV-like) from STEP
_ESVS_URL = (  # noqa: E501
    f"{_REPO_RAW}/ESVS%20-%20English%20Standard%20Version%20Selections/"
    "ESVS%20-%20English%20Standard%20Version%20Selections%20-%20STEPBible.org%20CC%20BY.txt"
)

# TSK cross-references (tab-delimited)
_TSK_URL = (  # noqa: E501
    f"{_REPO_RAW}/TSK%20-%20Treasury%20of%20Scripture%20Knowledge/"
    "TSK%20-%20Treasury%20of%20Scripture%20Knowledge%20-%20STEPBible.org%20CC%20BY.txt"
)

# OT/NT OSIS book abbreviation map (first 3 chars → OSIS)
_BOOK_ABBREV: dict[str, str] = {
    # OT
    "Gen": "Gen",
    "Exo": "Exod",
    "Lev": "Lev",
    "Num": "Num",
    "Deu": "Deut",
    "Jos": "Josh",
    "Jdg": "Judg",
    "Rut": "Ruth",
    "1Sa": "1Sam",
    "2Sa": "2Sam",
    "1Ki": "1Kgs",
    "2Ki": "2Kgs",
    "1Ch": "1Chr",
    "2Ch": "2Chr",
    "Ezr": "Ezra",
    "Neh": "Neh",
    "Est": "Esth",
    "Job": "Job",
    "Psa": "Ps",
    "Pro": "Prov",
    "Ecc": "Eccl",
    "Son": "Song",
    "Isa": "Isa",
    "Jer": "Jer",
    "Lam": "Lam",
    "Eze": "Ezek",
    "Dan": "Dan",
    "Hos": "Hos",
    "Joe": "Joel",
    "Amo": "Amos",
    "Oba": "Obad",
    "Jon": "Jonah",
    "Mic": "Mic",
    "Nah": "Nah",
    "Hab": "Hab",
    "Zep": "Zeph",
    "Hag": "Hag",
    "Zec": "Zech",
    "Mal": "Mal",
    # NT
    "Mat": "Matt",
    "Mar": "Mark",
    "Luk": "Luke",
    "Joh": "John",
    "Act": "Acts",
    "Rom": "Rom",
    "1Co": "1Cor",
    "2Co": "2Cor",
    "Gal": "Gal",
    "Eph": "Eph",
    "Phi": "Phil",
    "Col": "Col",
    "1Th": "1Thess",
    "2Th": "2Thess",
    "1Ti": "1Tim",
    "2Ti": "2Tim",
    "Tit": "Titus",
    "Phm": "Phlm",
    "Heb": "Heb",
    "Jam": "Jas",
    "1Pe": "1Pet",
    "2Pe": "2Pet",
    "1Jo": "1John",
    "2Jo": "2John",
    "3Jo": "3John",
    "Jud": "Jude",
    "Rev": "Rev",
}


# ── Download helpers ──────────────────────────────────────────────────────────


def download_text(url: str, *, timeout: float = 60.0) -> str:
    """Download a URL and return as UTF-8 text."""
    logger.info("Downloading %s", url)
    resp = httpx.get(url, follow_redirects=True, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def download_bytes(url: str, *, timeout: float = 120.0) -> bytes:
    logger.info("Downloading binary %s", url)
    resp = httpx.get(url, follow_redirects=True, timeout=timeout)
    resp.raise_for_status()
    return resp.content


# ── STEP plain-text parser ────────────────────────────────────────────────────
# Format: lines like  "Gen.1.1  In the beginning God created..."
# Tab or spaces between ref and text, with optional Hebrew/Greek columns

_STEP_LINE_RE = re.compile(
    r"^([A-Z][a-z]{1,2})\s+(\d+):(\d+)\s+(.+)$"  # "Gen 1:1  text..."
)
_OSIS_LINE_RE = re.compile(
    r"^([A-Za-z0-9]+)\.(\d+)\.(\d+)\s+(.+)$"  # "Gen.1.1  text..."
)


def _to_osis_ref(book: str, chapter: str, verse: str) -> str:
    osis_book = _BOOK_ABBREV.get(book[:3], book)
    return f"{osis_book}.{chapter}.{verse}"


def parse_step_text(raw: str) -> list[tuple[str, str]]:
    """Parse STEP plain-text format → list of (osis_ref, text)."""
    results = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("="):
            continue
        m = _OSIS_LINE_RE.match(line)
        if m:
            book, chap, verse, text = m.groups()
            ref = f"{book}.{chap}.{verse}"
            clean = text.strip()
            if clean:
                results.append((ref, clean))
            continue
        m = _STEP_LINE_RE.match(line)
        if m:
            book, chap, verse, text = m.groups()
            ref = _to_osis_ref(book, chap, verse)
            clean = text.strip()
            if clean:
                results.append((ref, clean))
    return results


# ── TSK parser ────────────────────────────────────────────────────────────────
# Format: tab-delimited — Ref \t CrossRefs


def parse_tsk(raw: str) -> list[tuple[str, str]]:
    """Parse TSK → list of (osis_ref, cross_ref_string)."""
    results = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            ref, xrefs = parts
            ref = ref.strip().replace(" ", ".").replace(":", ".")
            if xrefs.strip():
                results.append((ref, xrefs.strip()[:500]))
    return results


# ── CSV writers ───────────────────────────────────────────────────────────────


def write_csv(
    rows: list[tuple[str, str]], dest: Path, col_a: str = "OsisRef", col_b: str = "EnglishText"
) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([col_a, col_b])
        for ref, text in rows:
            if ref and text:
                writer.writerow([ref, text])
    return len(rows)


# ── Main pipeline ─────────────────────────────────────────────────────────────


def download_and_parse(data_dir: Path, tsk_dir: Path) -> tuple[int, int]:
    """Download STEPBible + TSK, write CSVs. Returns (bible_verses, tsk_refs)."""

    data_dir.mkdir(parents=True, exist_ok=True)
    tsk_dir.mkdir(parents=True, exist_ok=True)

    verse_rows: list[tuple[str, str]] = []

    # Try NT (TANTT is the main translators' NT text)
    try:
        raw_nt = download_text(_TIPNR_NT)
        nt_rows = parse_step_text(raw_nt)
        logger.info("NT parsed: %d verses", len(nt_rows))
        verse_rows.extend(nt_rows)
    except Exception:
        logger.error("Failed to download NT text", exc_info=True)

    # Try OT
    try:
        raw_ot = download_text(_TIPNR_OT)
        ot_rows = parse_step_text(raw_ot)
        logger.info("OT parsed: %d verses", len(ot_rows))
        verse_rows.extend(ot_rows)
    except Exception:
        logger.error("Failed to download OT text", exc_info=True)

    # Deduplicate by ref (keep first occurrence)
    seen: set[str] = set()
    unique_verses: list[tuple[str, str]] = []
    for ref, text in verse_rows:
        if ref not in seen:
            seen.add(ref)
            unique_verses.append((ref, text))

    bible_csv = data_dir / "stepbible_verses.csv"
    n_verses = write_csv(unique_verses, bible_csv)
    logger.info("Wrote %d verses → %s", n_verses, bible_csv)

    # TSK cross-references
    tsk_rows: list[tuple[str, str]] = []
    try:
        raw_tsk = download_text(_TSK_URL)
        tsk_rows = parse_tsk(raw_tsk)
        logger.info("TSK parsed: %d cross-reference entries", len(tsk_rows))
        tsk_csv = tsk_dir / "tsk_xrefs.csv"
        n_tsk = write_csv(tsk_rows, tsk_csv, col_a="OsisRef", col_b="CrossRefs")
        logger.info("Wrote %d TSK entries → %s", n_tsk, tsk_csv)
    except Exception:
        logger.error("Failed to download TSK data", exc_info=True)

    return n_verses, len(tsk_rows)


def index_into_lancedb(data_dir: Path) -> int:
    """Run BibleRAG.index_stepbible() to embed and store verses in LanceDB."""
    import sys

    # Add src paths so we can import workspace packages
    project_root = Path(__file__).parent.parent
    for src_path in project_root.glob("packages/*/src"):
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
    for src_path in project_root.glob("agents/*/src"):
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

    # Override stepbible path before importing BibleRAG
    import spiritual.bible_rag as rag_module

    rag_module._STEPBIBLE_PATH = data_dir  # type: ignore[attr-defined]

    from spiritual.bible_rag import BibleRAG

    rag = BibleRAG()

    if rag.is_indexed():
        logger.info(
            "LanceDB bible table already exists — skipping re-index (use --force to override)"
        )
        return 0

    logger.info("Embedding and indexing %s into LanceDB...", data_dir)
    logger.info("This will take several minutes — Ollama must be running with nomic-embed-text")
    logger.info("Start Ollama: ollama pull nomic-embed-text && ollama serve")
    n = rag.index_stepbible()
    logger.info("Indexed %d chunks into LanceDB", n)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and index STEPBible data into LanceDB")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/stepbible"),
        help="Directory for Bible CSVs (default: data/stepbible)",
    )
    parser.add_argument(
        "--tsk-dir",
        type=Path,
        default=Path("data/tsk"),
        help="Directory for TSK CSVs (default: data/tsk)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; use existing CSVs in data-dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and parse CSVs only; skip LanceDB indexing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index even if LanceDB table already exists",
    )
    args = parser.parse_args()

    if not args.skip_download:
        n_verses, n_tsk = download_and_parse(args.data_dir, args.tsk_dir)
        logger.info("Download complete: %d bible verses, %d TSK refs", n_verses, n_tsk)
    else:
        csv_files = list(args.data_dir.glob("*.csv"))
        logger.info(
            "Skipping download — using %d existing CSVs in %s", len(csv_files), args.data_dir
        )
        if not csv_files:
            logger.error(
                "No CSV files found in %s — remove --skip-download to fetch data", args.data_dir
            )
            raise SystemExit(1)

    if args.dry_run:
        logger.info("--dry-run: stopping before LanceDB indexing")
        return

    # Force re-index by temporarily renaming existing table check
    if args.force:
        logger.info("--force: will drop existing LanceDB table if present")
        try:
            from memory.store import MemoryStore

            store = MemoryStore()
            db = store._get_lance()
            if "bible_verses" in db.table_names():
                db.drop_table("bible_verses")
                logger.info("Dropped existing bible_verses table")
        except Exception:
            logger.warning("Could not drop existing table", exc_info=True)

    n_indexed = index_into_lancedb(args.data_dir)

    logger.info("=" * 60)
    if n_indexed > 0:
        logger.info("STEPBible ingestion complete: %d verses indexed", n_indexed)
    else:
        logger.info("STEPBible already indexed (use --force to re-index)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
