"""Cleaning and feature steps for the hotel cancellation model."""

import pandas as pd

LEAKY_COLUMNS = ["reservation_status", "reservation_status_date"]


def drop_duplicate_bookings(df):
    """Remove rows that are exact duplicates of an earlier row.

    Args:
        df: Raw bookings DataFrame.

    Returns:
        A new DataFrame with duplicate rows removed and the index reset to 0..n-1.
    """
    return df.drop_duplicates().reset_index(drop=True)


def fill_missing_agent(df):
    """Replace missing travel-agent IDs with 0. Rows are kept, not dropped.

    A value of 0 marks a booking made without a travel agent.

    Args:
        df: Bookings DataFrame with an ``agent`` column.

    Returns:
        A copy of ``df`` where missing ``agent`` values are set to 0.
    """
    out = df.copy()
    out["agent"] = out["agent"].fillna(0)
    return out


def fix_lead_time_units(df):
    """Convert City Hotel lead times from 2017 from hours to days.

    The booking system logged City Hotel lead times in hours during 2017, so
    those values are divided by 24. All other rows are unchanged.

    Args:
        df: Bookings DataFrame with ``hotel``, ``arrival_date_year``, and
            ``lead_time`` columns.

    Returns:
        A copy of ``df`` with City Hotel 2017 ``lead_time`` values in days.
    """
    # The booking system logged City Hotel lead times in hours during 2017.
    out = df.copy()
    mask = (out["hotel"] == "City Hotel") & (out["arrival_date_year"] == 2017)
    out.loc[mask, "lead_time"] = out.loc[mask, "lead_time"] / 24
    return out


def drop_leaky_columns(df):
    """Drop columns that reveal the booking outcome (target leakage).

    Removes ``reservation_status`` and ``reservation_status_date``, which are
    only known after a booking is canceled or checked out. Columns that are
    already absent are ignored.

    Args:
        df: Bookings DataFrame.

    Returns:
        A new DataFrame without the columns in ``LEAKY_COLUMNS``.
    """
    return df.drop(columns=LEAKY_COLUMNS, errors="ignore")


def add_booking_features(df):
    """Add total_nights and has_agent columns.

    ``total_nights`` is weekend nights plus week nights. ``has_agent`` is 1
    when ``agent`` is not 0, otherwise 0.

    Args:
        df: Bookings DataFrame with ``stays_in_weekend_nights``,
            ``stays_in_week_nights``, and ``agent`` columns.

    Returns:
        A copy of ``df`` with the two new columns.
    """
    out = df.copy()
    out["total_nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
    out["has_agent"] = (out["agent"] != 0).astype(int)
    return out


def clean_bookings(df):
    """Run every cleaning and feature step in order.

    Steps: drop duplicate rows, fill missing agent IDs with 0, convert City
    Hotel 2017 lead times to days, drop leaky columns, and add booking features.

    Args:
        df: Raw bookings DataFrame, as loaded from the CSV.

    Returns:
        The cleaned DataFrame, ready for ``train.split_data``.
    """
    out = drop_duplicate_bookings(df)
    out = fill_missing_agent(out)
    out = fix_lead_time_units(out)
    out = drop_leaky_columns(out)
    return add_booking_features(out)