"""Supervised local game-value fitting, with whole-map train/validation separation.

This fits signed capture fields, not RL policies. The image model and the
state/observed-clearance baseline share targets, samples and checkpoint rules.
"""

from pathlib import Path
from types import SimpleNamespace
import json
import time
import os
import numpy as np
import torch
from .learning import ContactValueNet
from ..reversible.map_game_field import first_crossing


def fit_contact_values(
    train_folders,
    validation_folders,
    output,
    *,
    steps=80000,
    seed=0,
    hidden=256,
    channels=16,
    device="cuda",
):
    """Train both models and retain the best accumulated-observation validation fit.

    Inputs are directories from reference.solve plus dataset.build_observations.
    Labels and observations remain outside the code repository. Validation
    selects checkpoints by field error; chase outcomes are separate evaluation.
    """
    args = SimpleNamespace(steps=steps, seed=seed, hidden=hidden, channels=channels)
    paths = {
        "train": [Path(x) for x in train_folders],
        "validation": [Path(x) for x in validation_folders],
    }
    if not all(paths.values()):
        raise ValueError("Both train and validation folders are required")
    names = [p.name for group in paths.values() for p in group]
    if len(set(names)) != len(names):
        raise ValueError("Dataset folder names must be distinct across splits")
    folders = {p.name: p for group in paths.values() for p in group}
    splits = {key: [p.name for p in group] for key, group in paths.items()}
    first_meta = json.loads((paths["train"][0] / "meta.json").read_text())
    config = first_meta["spec"]["config"]
    arena = first_meta["spec"]["arena"]
    footprints = {role: arena[role] for role in ["pursuer", "evader"]}
    extent = json.loads((paths["train"][0] / "observations_meta.json").read_text())[
        "sensor"
    ]["extent"]
    map_ids = {}
    for split, group in paths.items():
        map_ids[split] = set()
        for folder in group:
            meta = json.loads((folder / "meta.json").read_text())
            observed = json.loads((folder / "observations_meta.json").read_text())
            if (
                meta["spec"]["config"] != config
                or meta["times"] != first_meta["times"]
                or meta["spec"]["exit_radius"] != first_meta["spec"]["exit_radius"]
            ):
                raise ValueError("All references must share physics and time budgets")
            if any(
                meta["spec"]["arena"][role] != footprints[role] for role in footprints
            ):
                raise ValueError("All references must share physical footprints")
            if observed.get("version") != 2 or observed["sensor"]["extent"] != extent:
                raise ValueError("Need matching version2 observation contracts")
            source = meta["spec"]["arena"].get("source") or {}
            identity = source.get("sha256") or json.dumps(
                meta["spec"]["arena"], sort_keys=True
            )
            map_ids[split].add(identity)
    if map_ids["train"] & map_ids["validation"]:
        raise ValueError("Whole maps must be separated between training and validation")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        previous = json.loads((out / "meta.json").read_text())
        if previous.get("status") == "complete":
            if previous["steps"] == steps and previous["seed"] == seed:
                return previous
            raise ValueError("Completed output belongs to another training request")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_float32_matmul_precision("high")
    cache = {}
    times = np.asarray(first_meta["times"])
    for split in ["train", "validation"]:
        for mid in splits[split]:
            folder = folders[mid]
            meta = json.loads((folder / "meta.json").read_text())
            if meta["status"] != "complete" or meta["spec"].get("query_version") != 3:
                raise ValueError("Need completed continuous-query data_v3")
            o = np.load(folder / "observations.npz")
            labels = np.load(folder / "labels.npz")["values"].T.astype(np.float32)
            raw = labels.copy()
            labels = np.minimum.accumulate(np.minimum(labels, labels[:, :1]), axis=1)
            nframes = len(o["frames"])
            per = len(labels) // nframes
            if per * nframes != len(labels):
                raise ValueError("Expected fixed queries per observation")
            pool = []
            for f in range(nframes):
                local = labels[f * per : (f + 1) * per]
                ids = np.flatnonzero(
                    (abs(local[:, -1]) < 0.2) | ((local[:, 0] - local[:, -1]) > 0.5)
                )
                pool.append(ids if len(ids) else np.arange(per))
            cache[mid] = dict(
                s=torch.tensor(
                    o["local_states"].reshape(nframes, per, 8), device=device
                ),
                gp=torch.tensor(
                    o["geometry_partial"].reshape(nframes, per, 3), device=device
                ),
                gm=torch.tensor(
                    o["geometry_memory"].reshape(nframes, per, 3), device=device
                ),
                gi=torch.tensor(
                    o["geometry_ideal"].reshape(nframes, per, 3), device=device
                ),
                v=torch.tensor(labels.reshape(nframes, per, len(times)), device=device),
                partial=torch.tensor(o["partial"], device=device),
                memory=torch.tensor(o["memory"], device=device),
                ideal=torch.tensor(o["ideal_sdf"], device=device),
                pool=pool,
                nframes=nframes,
                per=per,
                projection_mean_change=float(abs(raw - labels).mean()),
            )
        print("loaded", split, mid, flush=True)
    models = {
        "state_value": ContactValueNet(
            False, extent=extent, hidden=576, times=times
        ).to(device),
        "map_value": ContactValueNet(
            True, extent=extent, hidden=args.hidden, channels=args.channels, times=times
        ).to(device),
    }
    optim = {
        k: torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-5)
        for k, m in models.items()
    }
    meta = dict(
        status="running",
        seed=args.seed,
        steps=args.steps,
        hidden=args.hidden,
        channels=args.channels,
        parameter_counts={
            k: sum(p.numel() for p in m.parameters()) for k, m in models.items()
        },
        splits={k: splits[k] for k in ["train", "validation"]},
        observations="balanced single-scan, accumulated-scan and ideal local maps; same labels; whole BARN maps held out",
        sampling="per frame: half uniform queries, half near capture boundary or substantial predicted progress",
        target="monotone signed capture field at9budgets; exact initial rectangle geometry; interpolated targets clipped by exact initial payoff",
        selection="lowest accumulated-observation validation MAE at2s, without test-world results",
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    best = {k: np.inf for k in models}
    history = []
    start_step = 0
    elapsed_offset = 0.0
    if (out / "resume.pt").exists():
        resume = torch.load(out / "resume.pt", map_location=device, weights_only=False)
        for name in models:
            models[name].load_state_dict(resume["models"][name])
            optim[name].load_state_dict(resume["optimizers"][name])
        rng.bit_generator.state = resume["rng"]
        torch.set_rng_state(resume["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([x.cpu() for x in resume["cuda_rng"]])
        start_step = resume["step"]
        history = json.loads((out / "history.json").read_text())
        history = [x for x in history if x["step"] <= start_step]
        best = resume.get(
            "best",
            {
                k: min(x["validation"][k]["memory"]["mae"] for x in history)
                for k in models
            },
        )
        elapsed_offset = history[-1]["seconds"] if history else 0.0
        print("Resuming step", start_step, flush=True)
    start = time.perf_counter()

    def atomic_save(payload, path):
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temp)
        os.replace(temp, path)

    @torch.no_grad()
    def evaluate():
        result = {}
        for name, model in models.items():
            model.eval()
            modes = {}
            for mode in ["partial", "memory", "ideal"]:
                errs = []
                signs = []
                false = []
                missed = []
                time_errors = []
                by_map = {}
                for mid in splits["validation"]:
                    d = cache[mid]
                    ev = []
                    sv = []
                    fv = []
                    mv = []
                    te = []
                    for frame in range(0, min(d["nframes"], 96), 4):
                        frames = slice(frame, frame + 4)
                        s = d["s"][frames, :128]
                        g = d[{"partial": "gp", "memory": "gm", "ideal": "gi"}[mode]][
                            frames, :128
                        ]
                        y = d["v"][frames, :128]
                        image = (
                            d[mode][frames].float()
                            if mode != "ideal"
                            else torch.stack(
                                [
                                    d["ideal"][frames].float(),
                                    torch.ones_like(
                                        d["ideal"][frames], dtype=torch.float32
                                    ),
                                ],
                                1,
                            )
                        )
                        pred = model(image, s, g)
                        ev.append(abs(pred[..., -1] - y[..., -1]).flatten())
                        sv.append(((pred[..., -1] <= 0) != (y[..., -1] <= 0)).flatten())
                        fv.append(((pred[..., -1] <= 0) & (y[..., -1] > 0)).flatten())
                        mv.append(((pred[..., -1] > 0) & (y[..., -1] <= 0)).flatten())
                        pp = pred.reshape(-1, len(times)).cpu().numpy().T
                        yy = y.reshape(-1, len(times)).cpu().numpy().T
                        a = first_crossing(times, pp)
                        b = first_crossing(times, yy)
                        valid = np.isfinite(a) & np.isfinite(b)
                        if valid.any():
                            te.extend(abs(a[valid] - b[valid]).tolist())
                    ee = torch.cat(ev)
                    ss = torch.cat(sv)
                    ff = torch.cat(fv)
                    mm = torch.cat(mv)
                    by_map[str(mid)] = {
                        "mae": ee.mean().item(),
                        "sign_error": ss.float().mean().item(),
                        "false_finite": ff.float().mean().item(),
                        "missed_finite": mm.float().mean().item(),
                        "time_mae_both_finite": float(np.mean(te)) if te else None,
                    }
                    errs.append(ee)
                    signs.append(ss)
                    false.append(ff)
                    missed.append(mm)
                    time_errors.extend(te)
                modes[mode] = {
                    "mae": torch.cat(errs).mean().item(),
                    "sign_error": torch.cat(signs).float().mean().item(),
                    "false_finite": torch.cat(false).float().mean().item(),
                    "missed_finite": torch.cat(missed).float().mean().item(),
                    "time_mae_both_finite": (
                        float(np.mean(time_errors)) if time_errors else None
                    ),
                    "by_map": by_map,
                }
            result[name] = modes
            model.train()
        return result

    for step in range(start_step + 1, args.steps + 1):
        S = []
        G = []
        V = []
        M = []
        for _ in range(4):
            mid = str(rng.choice(splits["train"]))
            d = cache[mid]
            f = int(rng.integers(d["nframes"]))
            ids = np.r_[rng.integers(d["per"], size=128), rng.choice(d["pool"][f], 128)]
            q = torch.tensor(ids, device=device)
            mode = ["partial", "memory", "ideal"][int(rng.integers(3))]
            S.append(d["s"][f, q])
            G.append(d[{"partial": "gp", "memory": "gm", "ideal": "gi"}[mode]][f, q])
            V.append(d["v"][f, q])
            M.append(
                torch.stack(
                    [
                        d["ideal"][f].float(),
                        torch.ones_like(d["ideal"][f], dtype=torch.float32),
                    ]
                )
                if mode == "ideal"
                else d[mode][f].float()
            )
        s, g, y, image = map(torch.stack, [S, G, V, M])
        losses = {}
        for name, m in models.items():
            optim[name].zero_grad(set_to_none=True)
            pred = m(image, s, g)
            weight = 1 + 4 * torch.exp(-abs(y[..., 1:]) / 0.2)
            loss = (weight * (pred[..., 1:] - y[..., 1:]).square()).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step{step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5)
            optim[name].step()
            losses[name] = loss.item()
        if step == 1 or step % 2000 == 0 or step == args.steps:
            validation = evaluate()
            row = dict(
                step=step,
                seconds=elapsed_offset + time.perf_counter() - start,
                loss=losses,
                validation=validation,
            )
            history.append(row)
            (out / "history.json").write_text(json.dumps(history, indent=2))
            for name, m in models.items():
                criterion = validation[name]["memory"]["mae"]
                if criterion < best[name]:
                    best[name] = criterion
                    atomic_save(
                        dict(
                            state_dict=m.state_dict(),
                            physics=config,
                            footprints=footprints,
                            times=times.tolist(),
                            exit_radius=first_meta["spec"]["exit_radius"],
                            extent=extent,
                            hidden=args.hidden if name == "map_value" else 576,
                            channels=args.channels if name == "map_value" else 16,
                            use_map=(name == "map_value"),
                            seed=args.seed,
                            step=step,
                            selection="accumulated-observation validation2sMAE",
                        ),
                        out / (name + "_best.pt"),
                    )
            atomic_save(
                {
                    "step": step,
                    "best": best,
                    "models": {k: m.state_dict() for k, m in models.items()},
                    "optimizers": {k: o.state_dict() for k, o in optim.items()},
                    "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (
                        torch.cuda.get_rng_state_all()
                        if device.startswith("cuda")
                        else []
                    ),
                },
                out / "resume.pt",
            )
            print(
                "step",
                step,
                "seconds",
                round(row["seconds"], 1),
                "val memory MAE",
                {k: round(v["memory"]["mae"], 4) for k, v in validation.items()},
                flush=True,
            )
    meta.update(
        status="complete",
        seconds=elapsed_offset + time.perf_counter() - start,
        best_validation_mae=best,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("TRAINING COMPLETE", out, flush=True)
    return meta
