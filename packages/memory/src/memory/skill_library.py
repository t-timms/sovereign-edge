from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    intent      TEXT    NOT NULL,
    description TEXT    NOT NULL,
    score       REAL    DEFAULT 1.0,
    use_count   INTEGER DEFAULT 0,
    created_at  REAL    DEFAULT (unixepoch())
);
CREATE INDEX IF NOT EXISTS idx_skills_intent_score ON skills (intent, score DESC);
"""

# Curated seed patterns — proven approaches before any runtime learning.
_SEED_SKILLS: list[tuple[str, str]] = [
    ("spiritual", "Retrieve via BibleRAG first; cite chapter:verse in every answer"),
    ("spiritual", "Devotional: key verse → 2-sentence reflection → practical application today"),
    ("career", "Lead with role + company, map to John's LangGraph/GRPO/vLLM stack"),
    ("career", "Tailor resume bullets: mirror JD language, never fabricate"),
    ("intelligence", "Market brief: biggest % mover first; flag news-driven vs technical"),
    ("intelligence", "Research digest: practical implication first, then paper + arXiv link"),
    ("creative", "LinkedIn hook: surprising stat → insight → engagement question"),
    ("creative", "Video script: problem hook (30s) → walkthrough → demo → CTA"),
]


class SkillLibrary:
    """Self-improving 3rd-tier memory: stores and ranks proven response patterns.

    Episodic memory (Mem0) stores *what happened*.
    Vector memory (LanceDB) stores *what was said*.
    Skill memory stores *how to do things well* — automatically reinforced
    by positive HITL outcomes and successful completions.

    Backed by SQLite (stdlib, no extra deps).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or Path("data/skills.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            self._seed_if_empty()
            logger.info("memory.skill_library_ready", path=str(self._db_path))
        return self._conn

    def _seed_if_empty(self) -> None:
        conn = self._conn
        if conn is None:
            return
        count = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        if count == 0:
            conn.executemany(
                "INSERT INTO skills (intent, description) VALUES (?, ?)",
                _SEED_SKILLS,
            )
            conn.commit()
            logger.info("memory.skill_library_seeded", count=len(_SEED_SKILLS))

    def get_top_skills(self, intent: str, *, limit: int = 2) -> list[str]:
        """Return top-scored skill descriptions for the given intent class."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT description FROM skills
               WHERE intent = ?
               ORDER BY score DESC, use_count DESC
               LIMIT ?""",
            (intent, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def record_outcome(self, intent: str, *, success: bool) -> None:
        """Reinforce or penalize the top-ranked skill for this intent.

        Called automatically:
        - success=True when user approves a HITL action or non-HITL task completes
        - success=False when user rejects a HITL action
        """
        conn = self._get_conn()
        delta = 0.1 if success else -0.05
        conn.execute(
            """UPDATE skills
               SET score     = MAX(0.1, score + ?),
                   use_count = use_count + 1
               WHERE id = (
                   SELECT id FROM skills WHERE intent = ?
                   ORDER BY score DESC LIMIT 1
               )""",
            (delta, intent),
        )
        conn.commit()
        logger.info(
            "memory.skill_outcome_recorded",
            intent=intent,
            success=success,
            delta=delta,
        )

    def add_skill(self, intent: str, description: str) -> None:
        """Persist a newly learned skill pattern (score starts at 1.0)."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO skills (intent, description) VALUES (?, ?)",
            (intent, description),
        )
        conn.commit()
        logger.info("memory.skill_added", intent=intent)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
