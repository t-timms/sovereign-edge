from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from career.scraper import JobListing, format_listings, scrape_jobs

# ── JobListing.is_dfw ─────────────────────────────────────────────────────────

DFW_CITIES = ["dallas", "fort worth", "plano", "irving", "frisco", "arlington", "allen"]


def test_is_dfw_matches_city_in_location() -> None:
    job = JobListing(
        title="ML Engineer",
        company="Acme",
        location="Dallas, TX",
        description="",
        url="",
        source="linkedin",
        date_posted="2026-04-03",
    )
    assert job.is_dfw(DFW_CITIES) is True


def test_is_dfw_matches_remote() -> None:
    job = JobListing(
        title="ML Engineer",
        company="Acme",
        location="Remote",
        description="",
        url="",
        source="indeed",
        date_posted="2026-04-03",
    )
    assert job.is_dfw(DFW_CITIES) is True


def test_is_dfw_rejects_non_dfw() -> None:
    job = JobListing(
        title="ML Engineer",
        company="Acme",
        location="San Francisco, CA",
        description="",
        url="",
        source="linkedin",
        date_posted="2026-04-03",
    )
    assert job.is_dfw(DFW_CITIES) is False


def test_is_dfw_case_insensitive() -> None:
    job = JobListing(
        title="Senior Engineer",
        company="ACME",
        location="PLANO, TX",
        description="",
        url="",
        source="linkedin",
        date_posted="2026-04-03",
    )
    assert job.is_dfw(DFW_CITIES) is True


def test_is_dfw_matches_partial_city_name() -> None:
    job = JobListing(
        title="Data Scientist",
        company="Corp",
        location="Fort Worth, Texas",
        description="",
        url="",
        source="glassdoor",
        date_posted="2026-04-03",
    )
    assert job.is_dfw(DFW_CITIES) is True


def test_is_dfw_empty_city_list_rejects_non_remote() -> None:
    job = JobListing(
        title="Engineer",
        company="Corp",
        location="New York, NY",
        description="",
        url="",
        source="linkedin",
        date_posted="2026-04-03",
    )
    assert job.is_dfw([]) is False


# ── scrape_jobs ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_scrape_jobs_returns_empty_when_jobspy_not_installed() -> None:
    with patch.dict(sys.modules, {"jobspy": None}):
        result = await scrape_jobs(["ML Engineer"], "Dallas, TX")

    assert result == []


@pytest.mark.asyncio()
async def test_scrape_jobs_returns_listings() -> None:
    # Build a minimal pandas-like DataFrame mock without importing pandas
    row = {
        "title": "ML Engineer",
        "company": "Capital One",
        "location": "Plano, TX",
        "description": "Build ML systems.",
        "job_url": "https://example.com/job/1",
        "site": "linkedin",
        "date_posted": "2026-04-03",
        "salary_source": "$150k",
        "is_remote": False,
    }
    mock_df = MagicMock()
    mock_df.iterrows.return_value = iter([(0, row)])

    mock_jobspy = MagicMock()
    mock_jobspy.scrape_jobs.return_value = mock_df

    with patch.dict(sys.modules, {"jobspy": mock_jobspy}):
        result = await scrape_jobs(["ML Engineer"], "Dallas, TX")

    assert len(result) == 1
    assert result[0].title == "ML Engineer"
    assert result[0].company == "Capital One"
    assert result[0].location == "Plano, TX"
    assert result[0].source == "linkedin"


@pytest.mark.asyncio()
async def test_scrape_jobs_continues_on_per_role_failure() -> None:
    ok_row = {
        "title": "AI Engineer",
        "company": "AT&T",
        "location": "Dallas, TX",
        "description": "AI work",
        "job_url": "https://example.com/job/2",
        "site": "indeed",
        "date_posted": "2026-04-03",
        "salary_source": "",
        "is_remote": False,
    }
    ok_df = MagicMock()
    ok_df.iterrows.return_value = iter([(0, ok_row)])

    call_count = 0

    def fake_scrape(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("rate limited")
        return ok_df

    mock_jobspy = MagicMock()
    mock_jobspy.scrape_jobs = fake_scrape

    with patch.dict(sys.modules, {"jobspy": mock_jobspy}):
        result = await scrape_jobs(["ML Engineer", "AI Engineer"], "Dallas, TX")

    # First role failed silently, second succeeded
    assert len(result) == 1
    assert result[0].title == "AI Engineer"


# ── format_listings ───────────────────────────────────────────────────────────


def test_format_listings_empty() -> None:
    assert format_listings([]) == "No jobs found matching your criteria."


def test_format_listings_includes_title_and_company() -> None:
    listings = [
        JobListing(
            title="ML Engineer",
            company="Capital One",
            location="Plano, TX",
            description="",
            url="https://example.com/job/1",
            source="linkedin",
            date_posted="2026-04-03",
            salary="$150k",
        )
    ]
    result = format_listings(listings)
    assert "ML Engineer" in result
    assert "Capital One" in result
    assert "Plano, TX" in result
    assert "$150k" in result


def test_format_listings_respects_limit() -> None:
    listings = [
        JobListing(
            title=f"Job {i}",
            company="Corp",
            location="Dallas, TX",
            description="",
            url=f"https://example.com/{i}",
            source="linkedin",
            date_posted="2026-04-03",
        )
        for i in range(20)
    ]
    result = format_listings(listings, limit=5)
    # Count job entries — each starts with a number
    count = sum(1 for line in result.splitlines() if line and line[0].isdigit())
    assert count == 5


def test_format_listings_omits_salary_when_empty() -> None:
    listings = [
        JobListing(
            title="Analyst",
            company="Corp",
            location="Dallas, TX",
            description="",
            url="https://example.com/j",
            source="glassdoor",
            date_posted="2026-04-03",
            salary="",  # no salary
        )
    ]
    result = format_listings(listings)
    assert "💰" not in result
