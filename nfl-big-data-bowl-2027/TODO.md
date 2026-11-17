# TODO — NFL Big Data Bowl 2027

Follow-ups from the deep dive. Details and numbers are in [DEEPDIVE.md](DEEPDIVE.md) (section refs below); code is in
`scripts/deepdive/`, numbered in run order.

## Housekeeping

- [ ] **Commit the work.** Nothing is committed yet. Candidates: `STARTER.md`, `DEEPDIVE.md`, `TODO.md`,
      `scripts/deepdive/`, `docs/deepdive/` (figures F1–F6), `notebooks/00_build_cache.ipynb`. `data/` and `.venv/` are
      gitignored.
- [ ] **Pick one Python environment.** The Anaconda-based `venv/` can't load torch (its bundled MSVC runtime 14.27 is too
      old, `WinError 1114 … c10.dll`). `.venv/` (python.org 3.11.8, torch 2.14.1+cu126) runs everything. Either retire
      `venv/` or rebuild it the same way (setup commands in DEEPDIVE §0).
- [ ] **Kaggle token.** If you re-save `~/.kaggle/access_token` from PowerShell, use `Set-Content -Encoding ascii -NoNewline`;
      `echo >` writes UTF-16 with a BOM and breaks the CLI (`Invalid header value b'Bearer \xff\xfe…'`). Make sure the old,
      partially pasted token is revoked.

## Write-up (decisions)

- [ ] **Choose the angle.** Recommended: lead with (1) the combine measures speed, not agility (reliability split, A3),
      (2) the braking blind spot (F1, A2), (3) DL as the exception (A3, A5). Present the fingerprint encoder as the novel
      method, and the betting layer as a short, well-tested null plus the Thursday finding (A8, C.1).
- [ ] **Decel-from-speed drill proposal.** Strongest evidence: F1 plus the DL disattenuated braking transfer (0.98) (C.4).
- [ ] **Don't claim** that combine tracking predicts career success (A7) or that the encoder beats simple features (A5).

## Robustness checks (cheap, do before publishing)

- [ ] **Bootstrap the disattenuated correlations** in `12_reliability.py`. Wide CIs with ~100 players/group; some can exceed 1.
- [ ] **3-seed encoder run** (`06_encoder.py`, ~5 min/seed). Only seed 0 has been trained.
- [ ] **REG/POST-only rerun of A5–A7.** Preseason is 16% of plays and mostly backups (C.2).
- [ ] **Trim `08_play_models.py`** to the `ctx` model. The single-stage feature sets are a documented failure and slow;
      only `ctx` feeds 08b/09.

## Open questions

- [ ] **SNF/MNF distance dip** (−0.12 SD, p = 0.006). Rest and weather don't explain it. Test kickoff time vs body clock
      and travel direction using team time zones (B.1, C.3).

## Betting layer: next steps

- [ ] **Extend B.3/B.4 to all 546 Kalshi prop players** using nflverse box scores. That's ~10× the power for the
      well-known vs lesser-known pricing question, with no tracking needed.
- [ ] **Replicate the two marginal leads on 2026 data** (the season is underway; re-pull with `fetch_markets.py`):
      - parlay-legs coefficient +0.12 (p = 0.022): more-parlayed props hit more often (B.3)
      - late taker shift toward the underdog +0.75 (p = 0.046) (B.4)
- [ ] **Play-level anomaly map.** Score every snap against the player's own encoder norm and join "out-of-character" snaps
      to in-game market candles (C, next steps #2).
- [ ] **Sportsbook data (optional).** FanDuel, Hard Rock, DraftKings etc. have no free historical prop feed, and scraping
      violates their terms. A paid historical odds/props feed is the only route.

**Guardrail:** player-level flag files (`data/deepdive/prop_markets.parquet`, `wallet_scores.parquet`,
`player_games.parquet`) stay local. Report aggregate tests only and don't name individuals in anything public (B.0, B.5).

## Kaggle notebook

- [ ] **Port `01`–`04`, `12` and figures F1–F3** (CPU-only, ~5 min). Present the encoder and the market layer as described
      results with code links, since a Kaggle notebook can't reproduce the market API pulls (C, next steps #3).
