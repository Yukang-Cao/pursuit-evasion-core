"""Training/checkpoint contract checks on tiny synthetic targets, not research data."""

from dataclasses import replace
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pursuit_evasion.local_game.training import fit_contact_values
from pursuit_evasion.local_game.learning import ContactNeuralField
from pursuit_evasion.local_game.geometry import CircleArena
from pursuit_evasion.reversible.model import ReversibleConfig
from pursuit_evasion.reversible.planning_frame import PlanningFrame


def fixture(folder, identity):
    folder.mkdir()
    c = replace(
        ReversibleConfig(),
        point_vehicles=False,
        capture_rule="rectangle_contact",
        boundary_rule="collision_constrained",
    )
    arena = CircleArena([], 4, source={"sha256": identity})
    times = np.arange(9) * 0.25
    meta = {
        "status": "complete",
        "times": times.tolist(),
        "spec": {
            "query_version": 3,
            "config": c.to_dict(),
            "arena": arena.to_dict(),
            "exit_radius": 3,
        },
    }
    (folder / "meta.json").write_text(json.dumps(meta))
    (folder / "observations_meta.json").write_text(
        json.dumps({"version": 2, "sensor": {"extent": 6}})
    )
    s = np.tile([0, 0, 0, 0, 1, 0, 0, 0], (64, 1)).astype(np.float64)
    g = np.tile([0.4, 2, 2], (64, 1)).astype(np.float32)
    image = np.ones((4, 2, 32, 32), np.float16)
    image[:, 0] = 2
    np.savez(
        folder / "observations.npz",
        local_states=s,
        geometry_partial=g,
        geometry_memory=g,
        geometry_ideal=g,
        partial=image,
        memory=image,
        ideal_sdf=image[:, 0],
        frames=np.zeros((4, 4)),
    )
    values = 0.4 - times[:, None] * np.linspace(0.1, 0.4, 64)[None]
    np.savez(folder / "labels.npz", values=values.astype(np.float32))
    return c, image[0].astype(np.float32)


def test_fit_checkpoint_runs_without_reference_at_inference(tmp_path):
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    c, image = fixture(train, "train-map")
    fixture(validation, "validation-map")
    out = tmp_path / "model"
    result = fit_contact_values(
        [train], [validation], out, steps=2, hidden=16, channels=4, device="cpu"
    )
    assert result["status"] == "complete"
    source = ContactNeuralField(out / "map_value_best.pt", c, device="cpu")
    source.prepare(image, PlanningFrame())
    fields, valid, shape = source.fields([0, 0, 0, 0, 1, 0, 0, 0])
    assert fields.shape == (9, 1) and valid.all()
    assert np.all(np.diff(fields[:, 0]) <= 1e-7)
    assert fields[0, 0] == pytest.approx(0.4)
    # Reusing a path must not silently return weights fitted to different labels.
    same = fit_contact_values(
        [train], [validation], out, steps=2, hidden=16, channels=4, device="cpu"
    )
    assert same["input_signature"] == result["input_signature"]
    labels = np.load(train / "labels.npz")["values"]
    labels[1:, 0] -= 0.1
    np.savez(train / "labels.npz", values=labels)
    with pytest.raises(ValueError, match="input data/settings"):
        fit_contact_values(
            [train], [validation], out, steps=2, hidden=16, channels=4, device="cpu"
        )
    # Duplicate geometry under another directory must not leak across splits.
    metadata = json.loads((validation / "meta.json").read_text())
    metadata["spec"]["arena"]["source"]["sha256"] = "train-map"
    (validation / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Whole maps"):
        fit_contact_values(
            [train], [validation], tmp_path / "bad", steps=1, device="cpu"
        )
