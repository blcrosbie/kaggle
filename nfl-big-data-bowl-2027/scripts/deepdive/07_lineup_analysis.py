"""Step 7: post-hoc lineup analysis on out-of-fold embeddings (no retraining).

Variants of player prototypes:
  raw       mean embedding per player per domain
  centered  subtract the context mean first (combine: drill_name; game: lined_up_position x route_ran x pass/run x
            window half), so the prototype is the player's deviation from what anyone does in that context
  envelope  hand-crafted capacity envelope (same-named features, z-scored within domain and position)
  fused     rank-average of centered-DL similarity and envelope similarity
Also split-half reliability inside each domain (combine: alternate reps per player; game: odd vs even game_id).
Output: data/deepdive/lineup_variants.parquet
"""
import numpy as np
import polars as pl
from scipy.stats import rankdata

from common import OUT, POS_GROUPS, players

E = np.load(OUT / "emb_oof.npy")
meta = (pl.read_parquet(OUT / "windows_meta.parquet")
          .join(players().select("nfl_id", "draft_year", "combine_position"), on="nfl_id", how="left")
          .with_columns(ctx=pl.when(pl.col("domain") == "combine").then(pl.col("drill_name"))
                              .otherwise(pl.concat_str([pl.col("lined_up_position").fill_null("?"), pl.col("route_ran").fill_null("-"),
                                                        pl.col("pass_result").is_null().cast(pl.String), pl.col("w").cast(pl.String)],
                                                       separator="|")),
                        half=pl.when(pl.col("domain") == "combine")
                               .then(pl.col("event_id").rank("dense").over("nfl_id", "domain") % 2 == 0)
                               .otherwise(pl.col("game_id") % 2 == 0)))
pos_map = dict(zip(meta["nfl_id"], meta["combine_position"]))
rng = np.random.default_rng(0)
env = pl.read_parquet(OUT / "envelopes.parquet")
FEATS = ["vmax", "as_A0", "as_V0", "brake", "lat_4_6", "lat_6_8", "acc_low", "omega_4_6"]


def center(E, ctx):
    Ec = E.copy()
    for c in np.unique(ctx):
        m = ctx == c
        Ec[m] -= E[m].mean(0)
    return Ec


def protos(E, ids):
    u = np.unique(ids)
    P = np.stack([E[ids == i].mean(0) for i in u])
    return u, P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)


def within_pct(S, pos):
    pct, top1, chance = [], [], []
    for p in np.unique(pos):
        idx = np.where(pos == p)[0]
        if len(idx) < 5:
            continue
        s = S[np.ix_(idx, idx)]
        r = (s > s.diagonal()[:, None]).sum(1)
        pct += list(1 - r / (len(idx) - 1))
        top1 += list(r == 0)
        chance += [1 / len(idx)] * len(idx)
    return np.mean(pct), np.mean(top1), np.mean(chance)


def perm_p(S, pos, obs, n=2000):
    null = []
    for _ in range(n):
        Sp = S.copy()
        for p in np.unique(pos):
            idx = np.where(pos == p)[0]
            Sp[np.ix_(idx, idx)] = S[np.ix_(idx, idx)][:, rng.permutation(len(idx))]
        null.append(within_pct(Sp, pos)[0])
    return (np.sum(np.array(null) >= obs) + 1) / (n + 1)


def env_sim(ids):
    e = env.select(["nfl_id", "combine_position"] + [f"{d}_{f}" for d in "cg" for f in FEATS])
    e = e.with_columns([((pl.col(c) - pl.col(c).mean().over("combine_position")) / pl.col(c).std().over("combine_position")).fill_null(0)
                        for c in e.columns[2:]])
    e = pl.DataFrame({"nfl_id": ids}).join(e, on="nfl_id", how="left").fill_null(0)
    C, G = e.select([f"c_{f}" for f in FEATS]).to_numpy(), e.select([f"g_{f}" for f in FEATS]).to_numpy()
    return -((G[:, None] - C[None]) ** 2).sum(-1)


def add(rows, yr, variant, S, pos, perm=True):
    pct, t1, ch = within_pct(S, pos)
    rows.append(dict(year=yr, variant=variant, n=len(pos), within_pct=pct, within_top1=t1, chance=ch,
                     p=perm_p(S, pos, pct) if perm else np.nan))


rows = []
for yr in (2023, 2024, 2025):
    m = (meta["draft_year"] == yr).to_numpy()
    Ey, my = E[m], meta.filter(pl.col("draft_year") == yr)
    g = (my["domain"] == "game").to_numpy()
    ids, ctx, half = my["nfl_id"].to_numpy(), my["ctx"].to_numpy(), my["half"].to_numpy()
    Ec = Ey.copy()
    Ec[~g], Ec[g] = center(Ey[~g], ctx[~g]), center(Ey[g], ctx[g])
    for variant, Ev in (("dl_raw", Ey), ("dl_centered", Ec)):
        gi, gp = protos(Ev[g], ids[g])
        ci, cp = protos(Ev[~g], ids[~g])
        both = np.array([i for i in np.intersect1d(gi, ci) if pos_map[i] in POS_GROUPS])
        pos = np.array([pos_map[i] for i in both])
        S = gp[np.searchsorted(gi, both)] @ cp[np.searchsorted(ci, both)].T
        add(rows, yr, variant, S, pos)
    Se = env_sim(both)
    add(rows, yr, "envelope", Se, pos)
    add(rows, yr, "fused", rankdata(S, axis=1) + rankdata(Se, axis=1), pos)
    for dom, mask in (("combine", ~g), ("game", g)):
        a_i, a_p = protos(Ec[mask & half], ids[mask & half])
        b_i, b_p = protos(Ec[mask & ~half], ids[mask & ~half])
        ab = np.array([i for i in np.intersect1d(a_i, b_i) if pos_map[i] in POS_GROUPS])
        add(rows, yr, f"split_half_{dom}", a_p[np.searchsorted(a_i, ab)] @ b_p[np.searchsorted(b_i, ab)].T,
            np.array([pos_map[i] for i in ab]), perm=False)

res = pl.DataFrame(rows)
res.write_parquet(OUT / "lineup_variants.parquet")
pl.Config.set_tbl_rows(40)
print(res.with_columns(pl.selectors.float().round(4)))
print(res.group_by("variant").agg(pl.col("within_pct").mean().round(3), pl.col("within_top1").mean().round(3),
                                   pl.col("chance").mean().round(3), pl.col("n").sum()).sort("within_pct"))

# ---- per-position breakdown (pooled over held-out classes): which groups carry the cross-domain signal?
pp_rows = []
for yr in (2023, 2024, 2025):
    m = (meta["draft_year"] == yr).to_numpy()
    Ey, my = E[m], meta.filter(pl.col("draft_year") == yr)
    g = (my["domain"] == "game").to_numpy()
    ids = my["nfl_id"].to_numpy()
    gi, gp = protos(Ey[g], ids[g])
    ci, cp = protos(Ey[~g], ids[~g])
    both = np.array([i for i in np.intersect1d(gi, ci) if pos_map[i] in POS_GROUPS])
    pos = np.array([pos_map[i] for i in both])
    for name, S in (("dl_raw", gp[np.searchsorted(gi, both)] @ cp[np.searchsorted(ci, both)].T), ("envelope", env_sim(both))):
        for p in POS_GROUPS:
            idx = np.where(pos == p)[0]
            s = S[np.ix_(idx, idx)]
            r = (s > s.diagonal()[:, None]).sum(1)
            pp_rows += [dict(method=name, pos=p, pct=v) for v in 1 - r / (len(idx) - 1)]
by_pos = (pl.DataFrame(pp_rows).group_by("method", "pos").agg(pl.col("pct").mean().round(3), pl.len().alias("n"))
            .pivot(on="method", index=["pos", "n"], values="pct").sort("pos"))
by_pos.write_parquet(OUT / "lineup_by_position.parquet")
print(by_pos)
