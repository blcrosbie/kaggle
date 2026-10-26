"""Step 6: cross-domain movement fingerprint encoder + "lineup" test on unseen draft classes.

Encoder: dilated residual 1D CNN over [v, a_lon, a_lat, omega] x 32 frames -> 128-d L2-normalized embedding.
Loss: SupCon on player id (all windows) + CLIP-style InfoNCE between each player's in-batch combine and game
prototypes + domain-adversarial head (gradient reversal) so embeddings can't tell combine from game.
Folds: leave-one-draft-class-out. For the held-out class: does a player's GAME prototype retrieve his own
COMBINE prototype among his peers (overall and within position group)? Baselines: chance, permutation null,
hand-crafted capacity envelope (same-named features, z-scored within domain and position).
Outputs: data/deepdive/encoder_fold{yr}.pt, emb_oof.npy (out-of-fold embeddings for every window), encoder_eval.json
"""
import json
import math
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import OUT, POS_GROUPS, players

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
P, KC, KG, TAU, LAM_DOM, EMB = 64, 4, 4, 0.1, 0.3, 128

X = np.load(OUT / "windows.npz")["X"]
meta = pl.read_parquet(OUT / "windows_meta.parquet")
pinfo = players().select("nfl_id", "draft_year", "combine_position")
meta = meta.join(pinfo, on="nfl_id", how="left")
is_game = (meta["domain"] == "game").to_numpy()
nfl = meta["nfl_id"].to_numpy()
mu = X.astype(np.float32).mean(axis=(0, 2), keepdims=True)
sd = X.astype(np.float32).std(axis=(0, 2), keepdims=True)
XT = torch.tensor((X.astype(np.float32) - mu) / sd, device=DEV)
MIRROR = torch.tensor([1.0, 1.0, -1.0, -1.0], device=DEV).view(1, 4, 1)   # left/right mirror flips a_lat, omega


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Res(nn.Module):
    def __init__(self, d, dil):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
        self.b1 = nn.BatchNorm1d(d)
        self.c2 = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
        self.b2 = nn.BatchNorm1d(d)

    def forward(self, x):
        return F.gelu(x + self.b2(self.c2(F.gelu(self.b1(self.c1(x))))))


class Encoder(nn.Module):
    def __init__(self, d=96, emb=EMB):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, d, 5, padding=2), nn.BatchNorm1d(d), nn.GELU())
        self.blocks = nn.Sequential(*[Res(d, k) for k in (1, 2, 4, 8, 1, 2, 4, 8)])
        self.head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, emb))
        self.dom = nn.Sequential(nn.Linear(emb, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return F.normalize(self.head(torch.cat([h.mean(-1), h.amax(-1)], 1)), dim=1)


def supcon(z, y, tau=TAU):
    sim = z @ z.T / tau
    sim = sim - torch.eye(len(z), device=z.device) * 1e9
    pos = (y[:, None] == y[None, :]).float() - torch.eye(len(z), device=z.device)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -((logp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()


def embed(model, rows, bs=8192):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(rows), bs):
            x = XT[rows[i:i + bs]]
            out.append(F.normalize(model(x).float() + model(x * MIRROR).float(), dim=1))   # mirror-averaged
    return torch.cat(out).cpu().numpy()


def prototypes(E, ids):
    u = np.unique(ids)
    P_ = np.stack([E[ids == i].mean(0) for i in u])
    return u, P_ / np.linalg.norm(P_, axis=1, keepdims=True)


def lineup(S, pos, rng, n_perm=2000):
    """S[i, j] = sim(game_i, combine_j), true match on the diagonal. Returns overall + within-position metrics."""
    n = len(S)
    rank_all = (S > S.diagonal()[:, None]).sum(1)                         # 0 = top-1
    res = dict(n=n, top1=float((rank_all == 0).mean()), top5=float((rank_all < 5).mean()),
               mrr=float((1 / (rank_all + 1)).mean()), chance_top1=1 / n)
    within_pct, top1_w, top3_w, chance_w = [], [], [], []
    for p in np.unique(pos):
        idx = np.where(pos == p)[0]
        if len(idx) < 5:
            continue
        s = S[np.ix_(idx, idx)]
        r = (s > s.diagonal()[:, None]).sum(1)
        within_pct += list(1 - r / (len(idx) - 1))                         # 1 = best, 0.5 = chance
        top1_w += list(r == 0)
        top3_w += list(r < 3)
        chance_w += [1 / len(idx)] * len(idx)
    res.update(within_pct=float(np.mean(within_pct)), within_top1=float(np.mean(top1_w)),
               within_top3=float(np.mean(top3_w)), within_chance_top1=float(np.mean(chance_w)))
    # permutation null for the within-position percentile: shuffle combine identities within position
    null = []
    for _ in range(n_perm):
        pct = []
        for p in np.unique(pos):
            idx = np.where(pos == p)[0]
            if len(idx) < 5:
                continue
            s = S[np.ix_(idx, idx)][:, rng.permutation(len(idx))]
            r = (s > s.diagonal()[:, None]).sum(1)
            pct += list(1 - r / (len(idx) - 1))
        null.append(np.mean(pct))
    res["within_pct_null_p"] = float((np.sum(np.array(null) >= res["within_pct"]) + 1) / (n_perm + 1))
    return res


def train_fold(test_year):
    tr_mask = (meta["draft_year"] != test_year).to_numpy()
    by_pc, by_pg = {}, {}
    for r in np.where(tr_mask)[0]:
        (by_pg if is_game[r] else by_pc).setdefault(nfl[r], []).append(r)
    ids = np.array([i for i in by_pc if len(by_pc[i]) >= 2 and len(by_pg.get(i, [])) >= 2])
    by_pc = {i: np.array(v) for i, v in by_pc.items()}
    by_pg = {i: np.array(v) for i, v in by_pg.items()}
    model = Encoder().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    t0 = time.time()
    for step in range(STEPS):
        model.train()
        pl_ = np.random.choice(ids, P, replace=False)
        rows = np.concatenate([np.random.choice(by_pc[i], KC) for i in pl_] + [np.random.choice(by_pg[i], KG) for i in pl_])
        y = torch.tensor(np.concatenate([np.repeat(np.arange(P), KC), np.repeat(np.arange(P), KG)]), device=DEV)
        dom = torch.cat([torch.zeros(P * KC), torch.ones(P * KG)]).to(DEV)
        x = XT[rows]
        flip = (torch.rand(len(x), 1, 1, device=DEV) < 0.5).float()
        x = x * (flip * MIRROR + (1 - flip)) + 0.05 * torch.randn_like(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x).float()
            zc = F.normalize(z[:P * KC].view(P, KC, -1).mean(1), dim=1)
            zg = F.normalize(z[P * KC:].view(P, KG, -1).mean(1), dim=1)
            logits = zg @ zc.T / TAU
            tgt = torch.arange(P, device=DEV)
            clip = 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))
            d_logit = model.dom(GradReverse.apply(z, LAM_DOM)).squeeze(1).float()
            loss = supcon(z, y) + clip + F.binary_cross_entropy_with_logits(d_logit, dom)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == STEPS - 1:
            acc = ((d_logit > 0).float() == dom).float().mean().item()
            print(f"  [{test_year}] step {step:5d} loss {loss.item():.3f} clip {clip.item():.3f} dom_acc {acc:.2f} ({time.time() - t0:.0f}s)")
    torch.save(model.state_dict(), OUT / f"encoder_fold{test_year}.pt")
    return model


def envelope_baseline(test_ids, pos_of):
    env = pl.read_parquet(OUT / "envelopes.parquet").filter(pl.col("nfl_id").is_in(test_ids.tolist()))
    feats = ["vmax", "as_A0", "as_V0", "brake", "lat_4_6", "lat_6_8", "acc_low", "omega_4_6"]
    env = env.select(["nfl_id", "combine_position"] + [f"c_{f}" for f in feats] + [f"g_{f}" for f in feats])
    z = env.with_columns([((pl.col(c) - pl.col(c).mean().over("combine_position")) / pl.col(c).std().over("combine_position")).fill_null(0)
                          for c in env.columns[2:]])
    z = z.join(pl.DataFrame({"nfl_id": test_ids}), on="nfl_id", how="right").fill_null(0)
    C = z.select([f"c_{f}" for f in feats]).to_numpy()
    G = z.select([f"g_{f}" for f in feats]).to_numpy()
    return -((G[:, None, :] - C[None, :, :]) ** 2).sum(-1)               # similarity = -squared distance


if __name__ == "__main__":
    E_oof = np.zeros((len(meta), EMB), dtype=np.float32)
    rng = np.random.default_rng(0)
    results = {}
    pos_map = dict(zip(pinfo["nfl_id"], pinfo["combine_position"]))
    for yr in (2023, 2024, 2025):
        model = train_fold(yr)
        te_rows = np.where((meta["draft_year"] == yr).to_numpy())[0]
        E = embed(model, te_rows)
        E_oof[te_rows] = E
        g_ids, g_proto = prototypes(E[is_game[te_rows]], nfl[te_rows][is_game[te_rows]])
        c_ids, c_proto = prototypes(E[~is_game[te_rows]], nfl[te_rows][~is_game[te_rows]])
        both = np.intersect1d(g_ids, c_ids)
        G = g_proto[np.searchsorted(g_ids, both)]
        C = c_proto[np.searchsorted(c_ids, both)]
        pos = np.array([pos_map[i] for i in both])
        keep = np.isin(pos, POS_GROUPS)
        both, G, C, pos = both[keep], G[keep], C[keep], pos[keep]
        res_dl = lineup(G @ C.T, pos, rng)
        res_env = lineup(envelope_baseline(both, pos_map), pos, rng)
        # within-game split-half (odd vs even games) - how strong is the in-game fingerprint itself?
        tg = te_rows[is_game[te_rows]]
        gid = meta["game_id"].to_numpy()[tg]
        Eg = E[is_game[te_rows]]
        odd = (gid % 2 == 1)
        a_ids, a_p = prototypes(Eg[odd], nfl[tg][odd])
        b_ids, b_p = prototypes(Eg[~odd], nfl[tg][~odd])
        ab = np.intersect1d(np.intersect1d(a_ids, b_ids), both)
        res_half = lineup(a_p[np.searchsorted(a_ids, ab)] @ b_p[np.searchsorted(b_ids, ab)].T,
                          np.array([pos_map[i] for i in ab]), rng, n_perm=200)
        results[yr] = dict(dl=res_dl, envelope=res_env, game_split_half=res_half)
        print(f"== held-out class {yr}: n={res_dl['n']}")
        for k, r in results[yr].items():
            print(f"   {k:16s} top1 {r['top1']:.3f} (chance {r['chance_top1']:.3f})  within-pos top1 {r['within_top1']:.3f} "
                  f"(chance {r['within_chance_top1']:.3f}) top3 {r['within_top3']:.3f}  pct {r['within_pct']:.3f}  p={r['within_pct_null_p']:.4f}")
    np.save(OUT / "emb_oof.npy", E_oof)
    json.dump(results, open(OUT / "encoder_eval.json", "w"), indent=1)
