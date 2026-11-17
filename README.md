# gta6-twitch-ccu-forecast-v2

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Alro07/gta6-twitch-ccu-forecast-v2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Alro07/gta6-twitch-ccu-forecast-v2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, as five quantiles per hour, from the first hour's viewer count and the release-day curves of past big game launches.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=q57bcks6&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

## Methodology

`main.py` (Python 3.13, standard library only, no network, deterministic) reads
every `*.in` case in `$MAKEFUN_INPUT_DIR` and writes `<id>.out` with five
quantiles (5%, 25%, 50%, 75%, 95%) for hours 2–24.

The forecast works in log space, anchored on the case's own hour-1 print:

- For each comparable past launch, it takes that launch's release-day hourly
  curve and forms log-ratios `log(viewers_h / viewers_1)`, hours 2–24.
- For each hour it fits a small weighted ridge regression of the log-ratio on
  four terms: an intercept, `log10(hour-1 viewers)` centred at 100k, a
  console-only flag, and a rolling-midnight-unlock flag (1 when the launch
  unlocked at midnight local, rolling by timezone). Size is the strongest
  signal — a launch that already drew everybody at midnight has little
  headroom left to grow, while one that opened almost empty before a
  simultaneous global unlock explodes later — and the unlock flag separates
  the two shapes: rolling-midnight launches climb into a US-evening peak,
  simultaneous ones spike at their single unlock hour.
- Launches are weighted by how close their hour-1 scale is to this case's
  (Gaussian kernel, 0.6 dex bandwidth, floored so every launch still
  speaks), boosted for rolling-midnight launches and discounted for
  early-access-contaminated days.
- The fitted curve is shrunk 15% toward the weighted median curve and lightly
  smoothed across neighbouring hours. The case's own game (the GTA VI
  pre-release history) is never used as a launch.
- Spreads come from the weighted spread of the regression residuals per hour,
  floored at `0.20 + 0.03·sqrt(h−1)` log units, so the band widens over the
  day; quantiles are `median · exp(z_tau · s_h)`.

## Validation

Scored with the contest's own pinball loss on the log scale. The practice
score uses the four official practice cases against their true SullyGnome
release-day hourly counts (hour-1 values verified to match the case files
exactly); the leave-one-out (LOO) score forecasts each of the 37 usable
comparable launches from all the others.

| model | practice (4 cases) | LOO all (37) | LOO rolling (12) | LOO rolling, hour-1 ≥ 50k (10) |
|---|---:|---:|---:|---:|
| house persistence baseline | 0.387 | 0.930 | — | — |
| v1 entry (regression on size + console only) | 0.188 | 0.443 | 0.235 | 0.179 |
| **this entry** | **0.153** | **0.418** | **0.221** | **0.164** |

What changed vs v1, and why it helps (each step measured on the same
practice + LOO sets before adopting it):

- **Rolling-midnight unlock as a regression feature.** The strongest single
  change: practice 0.188 → 0.161, rolling LOO 0.235 → 0.222. Verified
  rolling launches and simultaneous launches genuinely have different
  day shapes, and the flag lets the fit use that instead of blurring them.
- **Tighter size kernel (1.0 → 0.6 dex).** Launches near the target's
  hour-1 scale predict its shape better than distant ones; practice
  0.161 → 0.154, rolling-big LOO 0.172 → 0.164.
- **Lighter ridge (1.0 → 0.3).** Less shrinkage of the size and unlock
  slopes; small consistent gains on both sets.
- **No console weight boost.** The console flag now enters the regression
  itself, so the extra 1.5× weight is redundant; dropping it improved LOO
  slightly with no practice cost.

Ideas tested and rejected (they helped one set and hurt the other, or hurt
both): excluding tiny hour-1 launches (LOO 0.44 → 0.75 — the small
launches carry the "small hour-1, explosive growth" signal); a quadratic
size term (helps LOO, hurts practice); peak-anchored two-stage forecasting;
empirical residual quantiles instead of normal z's; tier-based weights;
viewers-per-channel and pre-release-max features; median-shifting the
prediction; IQR-based spreads. The pre-release trend was tested and does
not predict the day's shape, so it is used only to guard a degenerate
hour-1 print.
