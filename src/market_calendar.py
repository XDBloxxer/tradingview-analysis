"""
Shared, timezone-correct NYSE calendar helpers.

WHY THIS EXISTS
---------------
GitHub Actions runners run in UTC, and scheduled jobs are sometimes started
hours late. The old code used `datetime.now()` (runner-local = UTC) to decide
"which trading day is this run for?". A winners run that was supposed to
happen at 16:17 ET but actually started at 01:00-05:00 ET therefore stamped
its data with the NEXT calendar day (or a weekend date, which then got
skipped entirely). Everything downstream (predictor date, non-winners,
accuracy tracker) inherits that wrong date.

Rule used everywhere now: the *session date* is derived from the clock in
America/New_York, never from the runner's local time.
"""
import calendar
from datetime import date, datetime, timedelta, time as dt_time

import pytz

NY = pytz.timezone("America/New_York")
MARKET_CLOSE = dt_time(16, 0)
# TradingView's "gainers" list rolls over to the next session when
# pre-market opens (4:00 ET). Before that it still reflects the last close.
LIVE_ROLLOVER = dt_time(4, 0)


def now_et() -> datetime:
    return datetime.now(NY)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year, month, weekday, n) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + (n - 1) * 7)


def _last_weekday(year, month, weekday) -> date:
    last = date(year, month, calendar.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def nyse_holidays(year: int) -> set:
    """Full-day NYSE closures (Columbus/Veterans Day are NOT NYSE holidays;
    Good Friday IS — the old USFederalHolidayCalendar got both wrong)."""
    return {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 6, 19)),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }


def is_trading_day(d) -> bool:
    if isinstance(d, datetime):
        d = d.date()
    if d.weekday() >= 5:
        return False
    return d not in nyse_holidays(d.year)


def next_trading_day(d: date) -> date:
    d = d + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


def prev_trading_day(d: date) -> date:
    d = d - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def last_completed_session(now: datetime = None) -> date:
    """Most recent trading day whose 16:00 ET close has already happened."""
    now = now.astimezone(NY) if now else now_et()
    d = now.date()
    if is_trading_day(d) and now.time() >= MARKET_CLOSE:
        return d
    return prev_trading_day(d)


def upcoming_session(now: datetime = None) -> date:
    """The session a pre-market prediction run should predict for: today if
    today is a trading day and the close hasn't happened yet, otherwise the
    next trading day."""
    now = now.astimezone(NY) if now else now_et()
    d = now.date()
    if is_trading_day(d) and now.time() < MARKET_CLOSE:
        return d
    return next_trading_day(d)


def is_live_window(session_date: date, now: datetime = None) -> bool:
    """
    True if TradingView/yfinance 'gainers' endpoints still describe
    `session_date`'s close. That is: the session has closed, and the next
    session's pre-market (04:00 ET) hasn't started rolling the lists over.
    A run delayed to 01:00-03:59 ET (~08:00-10:59 Estonia) is still inside.
    """
    now = now.astimezone(NY) if now else now_et()
    if session_date != last_completed_session(now):
        return False
    nxt = next_trading_day(session_date)
    # Rollover = 04:00 ET on the calendar day of the next session
    # (weekend/holiday gaps just mean a longer window — still same data).
    rollover = NY.localize(datetime.combine(nxt, LIVE_ROLLOVER))
    return now < rollover
