"""Does the ground truth survive the trip from the markdown table into the comparison?

The manual count is the only external reference the project has, and it reaches the code as two
numbers typed into a markdown table by hand. Both failure modes here are silent. If the parser
does not recognise the way the number was written, `evaluate` reports "not recorded yet" for a
count that *was* recorded - the work looks undone. If the two directions do not add up to the
total written underneath them, the comparison prints one total while the README, NOTES.md and
the slides all quote another, and nothing says so.

Neither needs a video, a checkpoint or a GPU: the table is the input.

Run: `uv run pytest tests/test_manual_count.py -v`

Author: The Vinh Nguyen Trong
"""

from __future__ import annotations

import pytest

import config
from src.part2.evaluate import read_manual_count

TOWARD, AWAY = config.DIRECTION_LABELS[1], config.DIRECTION_LABELS[-1]


def table(toward, away, total):
    """A manual_count.md whose table rows are written exactly as given."""
    return (
        "# Manual vehicle count\n\n"
        "| Direction | Manual count |\n"
        "|---|---|\n"
        f"| {TOWARD} | {toward} |\n"
        f"| {AWAY} | {away} |\n"
        f"| **total** | {total} |\n"
    )


def write(tmp_path, text):
    path = tmp_path / "manual_count.md"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Still pending
# --------------------------------------------------------------------------------------


def test_untouched_template_reads_as_not_recorded(tmp_path):
    """`TODO` in both rows must give None, so evaluate refuses to print an accuracy."""
    assert read_manual_count(write(tmp_path, table("`TODO`", "`TODO`", 43))) is None


def test_one_direction_filled_in_is_still_not_a_count(tmp_path):
    """Half a tally is not a ground truth - the task sheet wants the comparison per direction."""
    assert read_manual_count(write(tmp_path, table(31, "`TODO`", 43))) is None


def test_missing_file_is_not_an_error(tmp_path):
    assert read_manual_count(tmp_path / "nothing_here.md") is None


# --------------------------------------------------------------------------------------
# How the number is written must not matter
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("toward, away", [
    ("31", "12"),                # plain
    ("`31`", "`12`"),            # backticks, the way the template writes TODO
    ("**31**", "**12**"),        # bold, the way the total row is written
    ("  31  ", "  12  "),        # padded
])
def test_formatting_around_the_number_is_ignored(tmp_path, toward, away):
    """Whoever fills the table in should not have to strip markdown for the parser's benefit.

    The template shows `TODO` in backticks, so keeping them is the natural thing to do - and it
    used to return None, which reads as "you have not done the count yet".
    """
    counts = read_manual_count(write(tmp_path, table(toward, away, 43)))
    assert counts == {TOWARD: 31, AWAY: 12, "total": 43}


# --------------------------------------------------------------------------------------
# The two directions and the total have to agree
# --------------------------------------------------------------------------------------


def test_directions_are_summed_into_the_total(tmp_path):
    counts = read_manual_count(write(tmp_path, table(30, 13, 43)))
    assert counts[TOWARD] + counts[AWAY] == counts["total"] == 43


def test_a_split_that_contradicts_the_total_is_refused(tmp_path):
    """31 + 5 is 36, not the 43 in the row below - and 43 is what every document quotes."""
    with pytest.raises(ValueError, match="36.*43|43.*36"):
        read_manual_count(write(tmp_path, table(31, 5, 43)))


def test_a_table_without_a_total_row_is_accepted(tmp_path):
    """The total is derived from the split; only a *contradicting* total is an error."""
    text = (f"| {TOWARD} | 31 |\n| {AWAY} | 12 |\n")
    assert read_manual_count(write(tmp_path, text)) == {TOWARD: 31, AWAY: 12, "total": 43}
