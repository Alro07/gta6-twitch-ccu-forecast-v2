"""Forecast Grand Theft Auto VI's launch-day Twitch viewers, hour by hour.

For every hour 2..24 of the UTC release day the program writes five
quantiles (5%, 25%, 50%, 75%, 95%) of that hour's concurrent viewers in
the game's Twitch category, anchored on the case's own hour-1 print.

How it works (log space throughout, standard library only, no network):

  1. Every comparable past launch contributes its release-day curve as
     log-ratios ``log(viewers[h] / viewers[1])`` against its own hour-1
     print.  A launch with no complete release day in the file (the GTA VI
     pre-release history is not a launch day) is skipped, as is the case's
     own game.
  2. For each hour a weighted ridge regression fits the log-ratio on four
     terms: an intercept, the launch's size ``log10(hour-1 viewers) - 5``,
     a console-only flag (no PC at launch), and a rolling-midnight-unlock
     flag (1 when the launch notes say "midnight local", else 0).  Size is
     the strongest signal -- a launch that already drew everybody at
     midnight has little headroom left to grow, while a launch that opened
     almost empty before a simultaneous global unlock explodes later --
     and the unlock flag separates the two shapes: rolling-midnight
     launches climb into a US-evening peak, simultaneous ones spike at
     their single unlock hour.  Launches are weighted by how close their
     hour-1 scale is to this case's (Gaussian kernel, 0.6 dex bandwidth,
     floored so every launch still speaks), boosted for rolling-midnight
     launches and discounted for early-access-contaminated days.
  3. The fitted curve is shrunk 15% toward the weighted median curve and
     lightly smoothed across neighbouring hours, which guards against a
     noisy fit when few comparables speak.
  4. The band for an hour comes from the weighted spread of that hour's
     regression residuals, scaled by 0.75 and floored at
     ``0.20 + 0.03 * sqrt(h - 1)`` log units so it widens over the day.
     Quantiles are ``median * exp(z_tau * s_h)`` with the standard-normal
     z's.  The size feature is clamped to the range the comparables span,
     so an hour-1 print beyond every past launch (GTA VI may well be one)
     extrapolates no further than the biggest launch on record.
  5. The case's own history is used only as a guard: a degenerate hour-1
     print is not trusted blindly when the category was already drawing a
     large audience before midnight.

Validation (contest pinball loss on the log scale, same cases for both
models -- see the experiment log in the README):

  * official practice cases (4 past launches, true SullyGnome counts):
    v1 0.188, this entry 0.153
  * leave-one-out over the 37 usable comparables: v1 0.443, this entry
    0.418; rolling-midnight launches only: v1 0.235, this entry 0.221;
    rolling launches with hour-1 >= 50k: v1 0.179, this entry 0.164.
  * house persistence baseline on the practice: about 0.387.

I/O contract: read every ``*.in`` in ``$MAKEFUN_INPUT_DIR`` (else
``/input``), write ``<id>.out`` to ``$MAKEFUN_OUTPUT_DIR`` (else
``/output``) as ``{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}]}``
for hours 2..24.  Deterministic: no randomness anywhere, fixed iteration
order, so two runs write identical bytes.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Knobs, picked by leave-one-out + practice-case scoring (see README).
RIDGE = 0.3          # ridge penalty on the non-intercept regression terms
SPREAD_MULT = 0.75   # scales the residual-based spread
BLEND = 0.85         # weight on the regression curve vs the median curve
SMOOTH = 0.4         # hour-to-hour smoothing of the log-ratio curve
SIZE_BANDWIDTH = 0.6  # dex, Gaussian kernel on log10(hour-1 viewers)
SIZE_FLOOR = 0.02    # every comparable keeps at least this weight
SPREAD_CAP = 2.0     # cap on the residual spread, log units

# Games whose launch is known to unlock at midnight local, rolling by
# timezone (the practice launches and the target itself).  The case's own
# notes name the pattern too; either source marks the target as rolling.
ROLLING_GAMES = {
    "Grand Theft Auto VI",
    "Marvel's Wolverine",
    "Red Dead Redemption 2",
    "God of War Ragnarök",
    "Marvel's Spider-Man 2",
}


def _parse_time(text):
    """ISO-8601 timestamp (with a trailing Z) to a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _day_curve(comp):
    """The comparable's release-day hourly viewer counts, hours 1..24, or None.

    Hours are taken by their offset from the comparable's day_start; a
    launch whose day is incomplete (the GTA VI pre-release history, whose
    day_start is a premiere date, not a release) is skipped.
    """
    try:
        day_start = _parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            when = _parse_time(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(viewers, (int, float)) or viewers is None:
            continue
        offset = (when - day_start).total_seconds() / 3600.0
        if offset == int(offset) and 0 <= offset < 24:
            by_hour[int(offset)] = max(1.0, float(viewers))
    if len(by_hour) < 24:
        return None
    return [by_hour[h] for h in range(24)]


def _is_console_only(comp):
    return "PC" not in (comp.get("platforms") or "").upper()


def _is_rolling(comp):
    return "midnight local" in (comp.get("unlock") or "").lower()


def _target_rolling(case):
    """The target launch's unlock pattern: 1 rolling, 0 simultaneous, else 0.5.

    GTA VI and the four practice launches all unlock at midnight local,
    rolling by timezone; the case notes say so too.  Anything else is
    treated as unknown rather than guessed.
    """
    game_name = (case.get("game") or {}).get("name") or ""
    if game_name in ROLLING_GAMES:
        return 1.0
    notes = (case.get("notes") or "")
    if isinstance(notes, str):
        low = notes.lower()
        if "midnight local" in low:
            return 1.0
        if "simultaneous" in low or "single global" in low:
            return 0.0
    return 0.5


def _launch_weight(comp, p1_case):
    """How much one comparable launch counts for this case."""
    log_gap = math.log10(max(1.0, comp["_p1"])) - math.log10(max(1.0, p1_case))
    w = math.exp(-0.5 * (log_gap / SIZE_BANDWIDTH) ** 2)
    w = max(w, SIZE_FLOOR)
    if _is_rolling(comp):
        w *= 1.2  # rolling midnight unlocks match GTA VI's pattern
    if "early access" in (comp.get("unlock") or "").lower():
        w *= 0.6  # early-access days are not clean release days
    return w


def _weighted_median(pairs):
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return statistics.median([v for v, _ in pairs])
    acc = 0.0
    for value, w in pairs:
        acc += w
        if acc >= total / 2.0:
            return value
    return pairs[-1][0]


def _solve(matrix, rhs):
    """Solve a small dense linear system by Gaussian elimination; None if singular."""
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        if abs(rows[col][col]) < 1e-12:
            return None
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for c in range(col, n + 1):
                rows[r][c] -= factor * rows[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (rows[i][n] - sum(rows[i][j] * out[j] for j in range(i + 1, n))) / rows[i][i]
    return out


def _ridge_fit(rows, ys, ws):
    """Weighted ridge regression; the intercept is not penalised."""
    n = len(rows[0])
    a = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for x, y, w in zip(rows, ys, ws):
        for i in range(n):
            b[i] += w * x[i] * y
            for j in range(n):
                a[i][j] += w * x[i] * x[j]
    for i in range(1, n):
        a[i][i] += RIDGE
    return _solve(a, b)


def _features(p1, console_only, rolling):
    x = math.log10(max(1.0, p1)) - 5.0
    return [1.0, x, 1.0 if console_only else 0.0, rolling]


def _guard_p1(p1, history_hourly):
    """Sanity-guard hour 1's print against the pre-release history.

    A print below 1 counts as 1.  If the print is tiny while the category
    was already drawing thousands of viewers an hour in the hours before
    midnight, the print is almost certainly a partial read, so the
    pre-release level is used instead of a near-zero anchor.
    """
    p1 = max(1.0, float(p1))
    if not history_hourly:
        return p1
    pre = []
    for row in history_hourly:
        viewers = row.get("viewers")
        if isinstance(viewers, (int, float)) and viewers and viewers > 0:
            pre.append(float(viewers))
    if len(pre) >= 3 and p1 < 100.0:
        level = statistics.median(pre[-6:])
        if level > 10.0 * p1:
            return max(p1, level)
    return p1


def forecast_case(case, name):
    """Build the 23-hour forecast for one case dict."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = _guard_p1(first["viewers"], case.get("history_hourly"))

    game_name = (case.get("game") or {}).get("name") or ""
    comps = []
    for comp in case.get("comparables") or []:
        if game_name and game_name in (comp.get("name") or ""):
            continue  # this case's own game is not a past launch
        day = _day_curve(comp)
        if day is None:
            continue
        entry = dict(comp)
        entry["_day"] = day
        entry["_p1"] = max(1.0, day[0])
        entry["_w"] = _launch_weight(entry, p1)
        comps.append(entry)
    if not comps:
        raise ValueError(f"{name} has no usable comparables")

    # The size feature is clamped to the range the comparables actually
    # span, so an hour-1 print beyond every past launch (GTA VI may well
    # be one) flattens the curve no further than the biggest launch on
    # record.  GTA VI is console-only (PS5 / Xbox Series X|S, no PC).
    lo = min(math.log10(c["_p1"]) for c in comps)
    hi = max(math.log10(c["_p1"]) for c in comps)
    case_log = min(max(math.log10(p1), lo), hi)
    case_x = [1.0, case_log - 5.0, 1.0, _target_rolling(case)]

    med_log = []
    spread = []
    for h in range(2, 25):
        rows = []
        ys = []
        ws = []
        for c in comps:
            rows.append(_features(c["_p1"], _is_console_only(c), 1.0 if _is_rolling(c) else 0.0))
            ys.append(math.log(c["_day"][h - 1] / c["_p1"]))
            ws.append(c["_w"])
        beta = _ridge_fit(rows, ys, ws)
        med = _weighted_median(list(zip(ys, ws)))
        if beta is None:
            pred = med
            resid = [y - med for y in ys]
        else:
            pred = sum(beta[i] * case_x[i] for i in range(len(case_x)))
            resid = [y - sum(beta[i] * x[i] for i in range(len(case_x))) for x, y in zip(rows, ys)]
        wsum = sum(ws)
        mean_resid = sum(w * r for w, r in zip(ws, resid)) / wsum
        var = sum(w * (r - mean_resid) ** 2 for w, r in zip(ws, resid)) / wsum
        spread.append(min(math.sqrt(max(var, 0.0)), SPREAD_CAP))
        med_log.append(BLEND * pred + (1.0 - BLEND) * med)

    # Light smoothing across neighbouring hours; endpoints stay as fitted.
    smoothed = list(med_log)
    for i in range(1, len(med_log) - 1):
        smoothed[i] = (1.0 - SMOOTH) * med_log[i] + SMOOTH * 0.5 * (med_log[i - 1] + med_log[i + 1])

    hours = []
    for i, h in enumerate(range(2, 25)):
        median = p1 * math.exp(smoothed[i])
        s = max(spread[i] * SPREAD_MULT, 0.20 + 0.03 * math.sqrt(h - 1))
        qs = []
        for z in Z:
            q = median * math.exp(z * s)
            qs.append(min(max(q, 1.0), 1e9))
        qs.sort()
        hours.append({"hour": h, "q": [round(q, 2) for q in qs]})
    return {"hours": hours}


def _fallback_forecast(case):
    """House-style persistence forecast, used only if the model fails.

    Never let one bad case sink the entry: an invalid case scores as every
    quantile equal to hour 1's print and takes the entry out of the prizes,
    so a crash must degrade to a valid forecast, never to no output.
    """
    p1 = 1.0
    try:
        first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
        if first is not None:
            p1 = max(1.0, float(first["viewers"]))
    except (TypeError, ValueError):
        p1 = 1.0
    hours = []
    for h in range(2, 25):
        s = 0.15 * math.sqrt(h - 1)
        qs = sorted(min(max(p1 * math.exp(z * s), 1.0), 1e9) for z in Z)
        hours.append({"hour": h, "q": [round(q, 2) for q in qs]})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = _fallback_forecast(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
