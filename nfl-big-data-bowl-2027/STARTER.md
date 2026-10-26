# STARTER — local-first discovery

Goal for the first rounds: get every table onto a common frame **locally**, prove one combine→game link end to end,
then pick the position group + outcome for the real analysis. Kaggle only enters at the end (the required public notebook).

Every number below was measured on the downloaded data (2026-10-23); every code cell was run end to end.

## 0. Workflow

```powershell
cd nfl-big-data-bowl-2027
.\venv\Scripts\Activate.ps1
mkdir notebooks; jupyter lab        # notebooks live in notebooks/, data in data/ (gitignored)
```

- Work off a **Parquet cache** (`data/parquet/`, ~0.8 GB, builds in ~20 s), never the raw CSVs. Use `pl.scan_parquet` +
  filters; only `.collect()` what you need.
- Nothing in discovery needs a GPU. This is an analytics comp (no leaderboard), so the only Kaggle limits that matter are
  notebook session/accelerator hours — and you don't spend them until you port the final notebook (§7).
- The same cells run on Kaggle unchanged: paths switch on `ON_KAGGLE`.

## 1. What's in the box

| File | Grain | Key | Rows |
|---|---|---|---|
| `players.csv` | player | `nfl_id` | 510 (2023: 160, 2024: 177, 2025: 173 draft classes) |
| `combine_results.csv` | player | `nfl_id` | 510 — official measurables + NGS scores |
| `player_career_successes.csv` | player | `nfl_id` | 510 — career snaps/starts/All-Pro/Pro Bowl |
| `combine_tracking.csv` | frame of a combine rep, 10 Hz | `event_id` + `time` | 463k (6,310 reps; 32k `BALL` rows) |
| `games.csv` | game | `game_id` | 1,002 (2023–25 × PRE/REG/POST) |
| `player_play.csv` | player × play | `game_id, play_id, nfl_id` | 314k |
| `game_tracking_{2023,2024,2025}.csv` | frame of a player-play, 10 Hz | `game_id, play_id, nfl_id, time` | 3.7M / 7.8M / 11.2M |

```
players ─┬─ combine_results ─ player_career_successes          (1 row per nfl_id, all 510)
         ├─ combine_tracking (event_id = one rep)                → combine layer
         └─ player_play ──(game_id, play_id, nfl_id)── game_tracking_*   → game layer
                 └─(game_id)── games
```

**Drill coverage** (players with tracking): forty 418 · 3-cone 160 · shuttle 170 · skill drills DB 122, DL 114,
OL 121, WR 107, TE 42, LB 4. Skill drill names by group:

- **WR**: SLANT, DAGGER, CURL, SLOT_STRIKE, GO, COMEBACK, SLOT_SAIL, RED_ZONE_FADE, POST_CORNER, SPEED_OUT, CIRCUS routes; GAUNTLET; OVER_SHOULDER_ADJUST
- **TE**: FLAT, HOOK, IN, WHEEL, CORNER, END_ZONE_FADE routes; GAUNTLET; BLOCK_EXPLOSION; OVER_SHOULDER_ADJUST
- **DB**: BACK_PEDAL_AND_TRANSITION_45, BACK_PEDAL_AND_90_BREAK, BOX, W_DRILL, GAUNTLET, TERYL_AUSTIN_1/2, LINE_DRILL
- **DL**: PASS_RUSH, FOUR_BAG_AGILITY, FRONT_START_WAVE, RUN_AND_CLUB, BODY_CONTROL, RUN_THE_HOOP; edge/OLB types also BACK_PEDAL_AND_REACT, SHORT_ZONE_BREAK*
- **OL**: PASS_RUSH_DROP, FIVE_YARD_WAVE_SLIDE_AND_SHUFFLE, SCREEN, OL_PULL_* (3), PASS_PRO_MIRROR

**`player_play` metrics are position-specific** (share non-null):

| lined_up_position | Populated outcome columns |
|---|---|
| WR, SLOT_WR, TE (RB/FB rare) | `route_ran` ~90–100%, `separation_at_pass_forward` ~80–88%, `cushion`, `rec_yards`/`yards_after_catch` when targeted |
| T, G, C | `pressure_allowed`, `sack_allowed`, `time_to_pressure_allowed`, `peak_pressure_probability_allowed` (~56% = pass plays) |
| EDGE, INTERIOR_LINE | `player_get_off` 57–68%, `time_to_pressure` (sparse, ~5%), `coverage_assignment` ~45% |
| CB, SLOT_CB, safeties | `cushion` 86–93% (CB), `coverage_assignment` ~50% |
| everyone | `tackle`, `assist`, EPA/WPA, down/distance, formation, `team_coverage_type` |

## 2. Gotchas found while profiling

1. **Only the 510 drafted players exist anywhere.** Game tracking holds just them — not all 22, and no ball. A play has
   1–11 tracked cohort players. Anything relative to an opponent comes from `player_play`'s precomputed metrics, or from
   plays where two cohort players face each other: **17,283** plays with a cohort receiver + cohort DB (4,848 with the DB in
   MAN), **36,443** with a cohort OL + cohort DL.
2. **Exposure is unbalanced.** Players in game tracking: 156 (2023) → 318 (2024) → 448 (2025); the 2023 class has three
   seasons, the 2025 class one. 337 of 497 players have ≥200 snaps.
3. **Rep key is `event_id`** (exactly one player, one drill, one attempt). `(nfl_id, drill_name, attempt)` is *not* unique.
4. **Combine `dis` is unreliable** — median 0.58× the actual x/y step. Recompute step distance from x/y. `s` (yd/s) and
   `dir` agree with x/y in both datasets. Combine has no `o`.
5. **`dir` is the NGS convention**: degrees clockwise from +y, so 90° = moving toward +x (verified against motion, ~1° error).
6. **Combine drills run at fixed field stations, some in both directions under one name** (gauntlet ±100°, pass rush
   ±70°, backpedal-90 ±120°) → normalize every rep (§3).
7. **Drill names changed between years**: `LINE`→`LINE_DRILL`, `PASS_PRO_MIRROR_DRILL`↔`PASS_PRO-MIRROR_DRILL`,
   `OVER_THE_SHOULDER_ADJUST`→`OVER_SHOULDER_ADJUST`, `SPEED_OUT_ROUTE_LEFT`↔`SPEED_OUT_LEFT` (handled in the cell below).
8. **Game x/y are raw** (offense goes left on half the plays). `play_direction` lives in `player_play`, not tracking.
9. **Game windows start ~1.0 s (10 frames) before `ball_snap`**, median ~7 s; a few run past 60 s → cap at e.g.
   `0 ≤ t_snap ≤ 10`. ~0.2% of frames sit on plays with no snap event.
10. **`player_play` ↔ tracking mismatch is small**: 312,447 matched; 1,750 `player_play` rows lack tracking; 2,497 tracked
    player-plays lack a `player_play` row (they get no `play_direction`, so they're unaligned — drop them).
11. **`player_play` quirks**: `pass_result` is `""` (not NA) on runs; `visitor_team_win_probility_added` is misspelled;
    play-level columns repeat on every player row (dedupe into a `plays` table).
12. **PRE and POST games are included** (~16% of player-plays are preseason) — decide deliberately; preseason usage differs.
13. **Combine position ≠ NFL position**: DL→OLB (30), DB→ILB/MLB/RB (4), WR→TE, TE→FB. Compare drills within
    `combine_position`; describe game roles with `lined_up_position`.
14. **Official measurables are sparse**: missing forty 86, 3-cone 330, shuttle 309, bench 318 (of 510).
15. **Tracked 40 ≠ stopwatch 40**: time-to-40-yards from tracking runs ~0.35 s slow (r = 0.90) because the window starts
    before first movement. **Top speed is the clean metric**: r = −0.98 with the official 40.

## 3. Alignment approach

Put each domain on its own common frame, build features at the natural grain, and only meet at `nfl_id`.

```
COMBINE  (unit = rep, event_id)                    GAME  (unit = player-play)
combine_tracking                                   game_tracking_*  +  plays (deduped from player_play)
 └ origin = rep start, +x = initial heading         └ flip "left" plays → offense always moves toward +x
   t_rep = s since rep start, step from x/y           x_los = yd past line of scrimmage, t_snap = s since snap
 └ rep features (top speed, accel, cut loss…)       └ the SAME kinematics per play + player_play outcomes
 └ player × drill (best / mean / sd over reps)      └ player × season (shrunk toward position mean)
                  \                                  /
                   └──── nfl_id ── players + combine_results + career ────┘
```

**Game frame** (standard NGS flip):

- if `play_direction == "left"`: `x' = 120 − x`, `y' = 53.33 − y`, `dir' = (dir + 180) % 360`, `o' = (o + 180) % 360`
- `los_x = 10 + yardline_number` if `yardline_side == possession_team`, else `110 − yardline_number` (60 at midfield)
- `x_los = x' − los_x` (negative = offense's backfield), `t_snap = time − snap time of the play`, `frame_snap = round(10·t_snap)`

Check at the snap: every offensive position has median `x_los` < 0 (T −1.3, WR −1.4, RB −5.3), every defensive one > 0
(EDGE +0.6, CB +4.9, HIGH_SAFETY +12.2); offense faces `o' ≈ 90°`, defense `≈ 270°`. GO routes are +5.3 yd at +2.0 s
heading 89°.

**Combine frame** (per rep):

- θ = angle from the start point to the first point ≥ 1.5 yd away
- `(x_rep, y_rep) = rotate(x − x0, y − y0, −θ)`, `dir_rep = (dir + θ°) % 360`, `t_rep = time − first time`
- `step = √(Δx² + Δy²)` replaces `dis`
- The ball (WR/TE drills) gets the same per-rep transform, so catch-point geometry stays consistent.
- Optional for two-way drills: mirror `y_rep` by the sign of lateral displacement to pool sides — but keep side as a
  feature (left/right asymmetry may be interesting).

Check: forty and gauntlet reps all end near 0° after rotation (gauntlet was ±100°); `dir_rep` matches motion to ~1°;
`step·10 / s` ≈ 1.00.

## 4. Starter cells (`notebooks/00_build_cache.ipynb`)

**Setup**

```python
from pathlib import Path
import polars as pl

ON_KAGGLE = Path("/kaggle/input").exists()
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
RAW = Path("/kaggle/input/nfl-big-data-bowl-2027") if ON_KAGGLE else ROOT / "data/nfl-big-data-bowl-2027"
CACHE = Path("/kaggle/working/parquet") if ON_KAGGLE else ROOT / "data/parquet"
CACHE.mkdir(parents=True, exist_ok=True)

READ = dict(null_values=["NA"], infer_schema_length=100_000)
TIME = pl.col("time").str.to_datetime(time_unit="ms")
FIELD_W = 160 / 3  # 53.33 yd
```

**Small tables, combine tracking, player_play**

```python
for name in ["players", "combine_results", "player_career_successes", "games"]:
    pl.read_csv(RAW / f"{name}.csv", **READ).write_parquet(CACHE / f"{name}.parquet")

pl.scan_csv(RAW / "combine_tracking.csv", **READ).with_columns(TIME).sink_parquet(CACHE / "combine_tracking.parquet")

pp = (pl.read_csv(RAW / "player_play.csv", **READ)
        .rename({"visitor_team_win_probility_added": "visitor_team_win_probability_added"})
        .with_columns(pl.col(pl.String).replace("", None)))      # pass_result is "" (not NA) on runs
pp.write_parquet(CACHE / "player_play.parquet")
```

**Play-level table with line of scrimmage**

```python
PLAY_COLS = ["game_id", "play_id", "possession_team", "play_direction", "yardline_side", "yardline_number",
             "quarter", "down", "yards_to_go", "game_clock", "offense_formation", "receiver_alignment",
             "pass_result", "pass_length", "team_coverage_man_zone", "team_coverage_type",
             "expected_points", "expected_points_added", "play_description"]
plays = (pp.group_by("game_id", "play_id")
           .agg(pl.col(c).drop_nulls().first() for c in PLAY_COLS[2:])
           .with_columns(los_x=pl.when(pl.col("yardline_number") == 50).then(60)
                                 .when(pl.col("yardline_side") == pl.col("possession_team")).then(10 + pl.col("yardline_number"))
                                 .otherwise(110 - pl.col("yardline_number"))))
plays.write_parquet(CACHE / "plays.parquet")
```

**Aligned game tracking** (22.6M rows → ~700 MB Parquet)

```python
gt = pl.concat([pl.scan_csv(RAW / f"game_tracking_{yr}.csv", **READ).with_columns(TIME, season=pl.lit(yr))
                for yr in (2023, 2024, 2025)])
flip = pl.col("play_direction") == "left"
aligned = (gt.join(plays.lazy().select("game_id", "play_id", "play_direction", "los_x"), on=["game_id", "play_id"], how="left")
             .with_columns(snap_time=pl.col("time").filter(pl.col("event") == "ball_snap").min().over("game_id", "play_id"))
             .with_columns(x_std=pl.when(flip).then(120 - pl.col("x")).otherwise(pl.col("x")),
                           y_std=pl.when(flip).then(FIELD_W - pl.col("y")).otherwise(pl.col("y")),
                           dir_std=pl.when(flip).then((pl.col("dir") + 180) % 360).otherwise(pl.col("dir")),
                           o_std=pl.when(flip).then((pl.col("o") + 180) % 360).otherwise(pl.col("o")),
                           t_snap=(pl.col("time") - pl.col("snap_time")).dt.total_milliseconds() / 1000)
             .with_columns(frame_snap=(pl.col("t_snap") * 10).round(0).cast(pl.Int32),
                           x_los=pl.col("x_std") - pl.col("los_x"))      # + = downfield of LOS (offense's view)
             .sort("game_id", "play_id", "nfl_id", "time"))
aligned.sink_parquet(CACHE / "game_tracking.parquet")
```

**Normalized combine reps**

```python
DRILL_ALIASES = {"LINE": "LINE_DRILL", "PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL",   # renamed between years
                 "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST", "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT"}
ct = pl.scan_parquet(CACHE / "combine_tracking.parquet").with_columns(pl.col("drill_name").replace(DRILL_ALIASES))
d0 = ((pl.col("x") - pl.col("x").first()) ** 2 + (pl.col("y") - pl.col("y").first()) ** 2).sqrt()
xf = (ct.filter(pl.col("entity_type") == "PLAYER").sort("event_id", "time")
        .group_by("event_id")
        .agg(x0=pl.col("x").first(), y0=pl.col("y").first(), t0=pl.col("time").first(),
             theta=pl.arctan2(pl.col("y") - pl.col("y").first(), pl.col("x") - pl.col("x").first()).filter(d0 >= 1.5).first()))
dx, dy = pl.col("x") - pl.col("x0"), pl.col("y") - pl.col("y0")
reps = (ct.join(xf, on="event_id", how="left")
          .with_columns(t_rep=(pl.col("time") - pl.col("t0")).dt.total_milliseconds() / 1000,
                        x_rep=dx * pl.col("theta").cos() + dy * pl.col("theta").sin(),
                        y_rep=-dx * pl.col("theta").sin() + dy * pl.col("theta").cos(),
                        dir_rep=(pl.col("dir") + pl.col("theta").degrees()) % 360)
          .sort("event_id", "entity_type", "time")
          .with_columns(step=((pl.col("x").diff() ** 2 + pl.col("y").diff() ** 2).sqrt()).over("event_id", "entity_type"))
          .drop("x0", "y0", "t0"))
reps.sink_parquet(CACHE / "combine_reps.parquet")
```

(3 reps never move 1.5 yd and get a null `theta` — drop them.)

**Player table**

```python
players = (pl.read_parquet(CACHE / "players.parquet")
             .join(pl.read_parquet(CACHE / "combine_results.parquet"), on=["nfl_id", "draft_year"], how="left")
             .join(pl.read_parquet(CACHE / "player_career_successes.parquet"), on="nfl_id", how="left"))
```

## 5. Sanity checks + first link

```python
reps = pl.scan_parquet(CACHE / "combine_reps.parquet")
forty = (reps.filter(pl.col("drill_name") == "FORTY_YARD_DASH", pl.col("entity_type") == "PLAYER")
             .group_by("nfl_id").agg(combine_top_speed=pl.col("s").max()).collect())
forty.join(players, on="nfl_id").select(pl.corr("combine_top_speed", "forty"))     # expect ≈ -0.98
```

```python
gt = pl.scan_parquet(CACHE / "game_tracking.parquet")
game_speed = (gt.filter(pl.col("t_snap").is_between(0, 8))
                .group_by("nfl_id", "game_id", "play_id").agg(pl.col("s").max())
                .group_by("nfl_id").agg(game_top_speed=pl.col("s").quantile(0.99), n_plays=pl.len())
                .collect())
link = forty.join(game_speed, on="nfl_id").join(players.select("nfl_id", "combine_position"), on="nfl_id")
(link.filter(pl.col("n_plays") >= 100)
     .group_by("combine_position")
     .agg(pl.len(), r=pl.corr("combine_top_speed", "game_top_speed"))
     .sort("combine_position"))
```

| combine_position | n | r (combine top speed vs in-game p99 top speed) |
|---|---|---|
| DB | 86 | 0.49 |
| DL | 77 | 0.70 |
| OL | 84 | 0.26 |
| TE | 20 | 0.42 |
| WR | 69 | 0.60 |

That's the full pipeline proven: combine tracking → aligned game tracking → per-player link. Top speed is the trivial
version (it's also in the official 40); the competition rewards what the stopwatch *can't* see.

## 6. Building the real link

Two kinds of link, both satisfy the pass/fail "combine tracking → NFL game performance" requirement:

- **Transfer — "does the movement show up on Sundays?"** Compute the *same* kinematic measure in both domains and
  correlate by player (like §5). Candidates: accel to 90% of top speed, peak decel, speed kept through a cut,
  backpedal→drive transition time, lateral shuffle speed, first-step burst. Clean, visual, cheap.
- **Outcome — "does it matter?"** Combine-tracking features → `player_play` outcomes, *beyond* what the official
  measurables and draft slot already explain.

| Group | Combine drills | In-game analog (tracking, aligned) | `player_play` outcome |
|---|---|---|---|
| WR / TE | route drills (SLANT, GO, CURL, COMEBACK, SPEED_OUT, POST_CORNER, IN, WHEEL, FLAT…), GAUNTLET | same route via `route_ran` (SLANT, GO, HITCH, OUT, CORNER, IN, WHEEL, FLAT): decel into the break, speed out of it | `separation_at_pass_forward`, `target`, `rec_yards`, YAC vs `expected_yards_after_catch` |
| DB | BACK_PEDAL_*, W_DRILL, BOX, TERYL_AUSTIN_* | backpedal→drive transition on pass plays, by `coverage_assignment` | `cushion`; separation allowed on cohort WR-vs-DB plays (17k; 4.8k man) |
| DL / EDGE | PASS_RUSH, RUN_AND_CLUB, FOUR_BAG_AGILITY, FRONT_START_WAVE | first-step burst (`t_snap` 0–1 s), bend/path around the corner | `player_get_off`, `time_to_pressure`, pressure/sack rate |
| OL | PASS_PRO_MIRROR, FIVE_YARD_WAVE_SLIDE_AND_SHUFFLE, PASS_RUSH_DROP, OL_PULL_* | kick-slide speed/depth after snap, pull speed | `pressure_allowed`, `sack_allowed`, `time_to_pressure_allowed`, `peak_pressure_probability_allowed` |

Traps to design around from day one:

- **Draft capital confound.** Teams drafted these players partly *on* the combine, and draft slot drives snaps and
  teammates. Always report incremental value over a baseline of `draft_overall_pick` + official measurables / `ngs_athleticism_score`.
- **Opportunity ≠ performance.** Career snaps/starts mostly measure opportunity. Prefer per-opportunity rates
  (separation per route, pressure per rush) with shrinkage — empirical Bayes or a mixed model with a player random effect
  fit at play level (more power than one row per player).
- **Small n.** ~80–120 players per position group. Pick a few pre-declared features per group; validate with
  leave-one-draft-class-out CV.
- **Play context.** Control for route, coverage, down/distance, alignment, and the opponent when he's a cohort player.

## 7. Notebook sequence

| Notebook | Purpose |
|---|---|
| `00_build_cache` | §4 cells. Rerun only when the cache logic changes. |
| `01_coverage` | Per player: drills/reps done, plays by season × `lined_up_position`. Choose the group + outcome here. |
| `02_combine_reps` | Spaghetti plots of normalized reps per drill; decide mirroring; define rep features. |
| `03_game_alignment` | Animate/plot a few plays in aligned coords; snap-relative kinematics per position. |
| `04_transfer` | Same-measure transfer table (top speed → accel → cut speed → transition time). |
| `05_outcomes` | Chosen group + outcome, with the draft-pick baseline and class-out CV. |

## 8. Porting to Kaggle (end game)

- Same cells; `ON_KAGGLE` switches paths. Confirm the competition mount path in the notebook's *Input* panel and fix `RAW`
  if it differs. The cache writes to `/kaggle/working/parquet` (~0.8 GB).
- Check `import polars` in the Kaggle image first; if missing, `pip install polars` (needs Internet on in notebook settings).
- CPU session is enough — leave accelerators off.
- Keep the public notebook lean: build cache → features → the handful of figures in the writeup.
