"""Paired supervised map-context ablation; whole-layout validation/test splits."""

from pathlib import Path
import sys, json, time, copy, argparse
import numpy as np

from pursuit_evasion.reversible.footprint_map import FootprintMap

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--steps", type=int, default=16000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
H = args.dataset.resolve()
import torch
from pursuit_evasion.reversible.map_value import MapValue

torch.set_num_threads(4)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_float32_matmul_precision("high")
device = torch.device(args.device)
out = args.output.resolve()
if out.exists() and any(out.iterdir()):
    raise FileExistsError("Choose a new output directory")
out.mkdir(parents=True, exist_ok=True)
entries = json.loads((H / "maps.json").read_text())
by_split = {}
cache = {}
for entry in entries:
    f = H / "results" / entry["id"]
    z = np.load(f / "labels.npz")
    if len(z["times"]) != 5 or not np.allclose(z["times"], [0, 0.5, 1, 1.5, 2]):
        raise ValueError(
            "This training example expects the full two-second suite, not smoke-test labels."
        )
    states = z["states"].astype(np.float32)
    m = FootprintMap.from_dict(entry["geometry"])
    pc = m.clearance(states[:, :4], "pursuer")
    ec = m.clearance(states[:, 4:], "evader")
    geom = np.stack([z["values"][0], pc, ec], -1).astype(np.float32)
    values = z["values"].astype(np.float32)
    pool = np.flatnonzero((abs(values[-1]) < 0.2) | ((values[0] - values[-1]) > 0.2))
    if not len(pool):
        pool = np.arange(len(states))
    cache[entry["id"]] = {
        "s": torch.tensor(states, device=device),
        "g": torch.tensor(geom, device=device),
        "v": torch.tensor(values, device=device),
        "map": torch.tensor(np.load(f / "map_sdf.npy")[None], device=device),
        "pool": torch.tensor(pool, device=device),
    }
    by_split.setdefault(entry["split"], []).append(entry["id"])
models = {
    "clearance_mlp": MapValue(False, hidden=576).to(device),
    "map_value": MapValue(True, hidden=256).to(device),
}
optim = {
    k: torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-5)
    for k, m in models.items()
}
counts = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
meta = dict(
    seed=args.seed,
    steps=args.steps,
    models=counts,
    training_maps=by_split["train"],
    validation_maps=by_split["validation"],
    test_maps=by_split["test"],
    ood_maps=by_split["ood"],
    input="full8D queried states+time+currentP/E clearance+initialpayoff; map model also receives SDF image",
    output="initial payoff + time * learned correction (signed game value, not seconds)",
    training_sampling="half uniform nodes; half nearzero final field or larger future change; uniform time budgets0.5,1,1.5,2",
    selection="best mean validation MAE across held-out entire layouts; no test-based selection",
    loss="MSE weighted by1+4exp(-abs(reference)/.2)",
    device=str(device),
    status="running",
)
(out / "meta.json").write_text(json.dumps(meta, indent=2))
history = []
best = {k: float("inf") for k in models}
started = time.perf_counter()


@torch.no_grad()
def evaluate(split, limit=8000):
    result = {}
    for name, model in models.items():
        model.eval()
        errors = []
        false_wins = []
        sign = []
        near = []
        by_map = {}
        for key in by_split[split]:
            d = cache[key]
            states = d["s"][:limit][None]
            geo = d["g"][:limit][None]
            feat = model.encode(d["map"][None])
            es = []
            ss = []
            fw = []
            for ti in range(1, 5):
                times = torch.full(states.shape[:2], ti * 0.5, device=device)
                target = d["v"][ti, :limit][None]
                pred = model.query(feat, states, times, geo)
                error = abs(pred - target)
                es.append(error.flatten())
                ss.append(((pred <= 0) != (target <= 0)).flatten())
                fw.append(((pred <= 0) & (target > 0)).flatten())
                mask = abs(target) < 0.2
                if mask.any():
                    near.append(error[mask])
            allerror = torch.cat(es)
            allsign = torch.cat(ss)
            allfw = torch.cat(fw)
            by_map[key] = {
                "mae": allerror.mean().item(),
                "sign_disagreement": allsign.float().mean().item(),
                "false_win_fraction": allfw.float().mean().item(),
            }
            errors.append(allerror)
            sign.append(allsign)
            false_wins.append(allfw)
        result[name] = {
            "mae": torch.cat(errors).mean().item(),
            "sign_disagreement": torch.cat(sign).float().mean().item(),
            "false_win_fraction": torch.cat(false_wins).float().mean().item(),
            "near_boundary_mae": torch.cat(near).mean().item() if near else None,
            "by_map": by_map,
        }
        model.train()
    return result


for step in range(1, args.steps + 1):
    keys = np.random.choice(by_split["train"], 4, replace=True)
    S = []
    G = []
    V = []
    T = []
    M = []
    for key in keys:
        d = cache[key]
        u = torch.randint(len(d["s"]), (256,), device=device)
        p = d["pool"][torch.randint(len(d["pool"]), (256,), device=device)]
        q = torch.cat([u, p])
        ti = torch.randint(1, 5, (512,), device=device)
        S.append(d["s"][q])
        G.append(d["g"][q])
        V.append(d["v"][ti, q])
        T.append(ti.float() * 0.5)
        M.append(d["map"])
    s, g, v, t, m = map(torch.stack, [S, G, V, T, M])
    losses = {}
    for name, model in models.items():
        optim[name].zero_grad(set_to_none=True)
        pred = model(m, s, t, g)
        weight = 1 + 4 * torch.exp(-v.abs() / 0.2)
        loss = (weight * (pred - v).square()).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optim[name].step()
        losses[name] = loss.item()
    if step == 1 or step % 400 == 0:
        val = evaluate("validation")
        record = {
            "step": step,
            "seconds": time.perf_counter() - started,
            "loss": losses,
            "validation": val,
        }
        history.append(record)
        for name in models:
            if val[name]["mae"] < best[name]:
                best[name] = val[name]["mae"]
                torch.save(
                    {
                        "state_dict": models[name].state_dict(),
                        "use_map": models[name].use_map,
                        "hidden": 576 if name == "clearance_mlp" else 256,
                        "step": step,
                        "validation": val[name],
                        "physics": json.loads((H / "config.json").read_text()),
                        "footprints": {
                            role: entries[0]["geometry"][role]
                            for role in ("pursuer", "evader")
                        },
                        "max_budget": 2.0,
                    },
                    out / (name + "_best.pt"),
                )
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(
            "step",
            step,
            "seconds",
            round(record["seconds"], 1),
            "valMAE",
            {k: round(v["mae"], 4) for k, v in val.items()},
            flush=True,
        )
for name in models:
    models[name].load_state_dict(
        torch.load(out / (name + "_best.pt"), map_location=device, weights_only=False)[
            "state_dict"
        ]
    )
results = {
    split: evaluate(split, limit=16000)
    for split in ["train", "validation", "test", "ood"]
}
(out / "metrics.json").write_text(json.dumps(results, indent=2))
meta.update(
    status="complete",
    wall_seconds=time.perf_counter() - started,
    best_validation=best,
    gpu_max_allocated=(
        torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    ),
    gpu_max_reserved=(
        torch.cuda.max_memory_reserved() if device.type == "cuda" else None
    ),
)
(out / "meta.json").write_text(json.dumps(meta, indent=2))
print("COMPLETE", results["test"], flush=True)
