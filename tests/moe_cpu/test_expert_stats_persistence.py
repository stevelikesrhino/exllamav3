"""Placement sweeps must not prevent periodic or unload-time profile persistence."""
import json
from types import SimpleNamespace

import pytest
import torch

import exllamav3.modules.block_sparse_mlp_cpu as bsm_cpu
from exllamav3.model.model import Model

pytestmark = pytest.mark.nogpu


def make_module(tmp_path):
    m = bsm_cpu.BlockSparseMLP_CPU()
    ip = SimpleNamespace(moe_cpu_swap_modules = [m])
    m.config = SimpleNamespace(directory = str(tmp_path / "model"), infer_params = ip)
    m.key = "model.layers.0.mlp"
    m.num_experts = 4
    m._split_hist = torch.tensor([2.0, 3.0, 5.0, 7.0])
    m._split_perm = [2, 0, 3, 1]
    m._split_spares = 0
    m._swap_tick_count = 0
    m._split_step_count = 0
    return m


def test_save_after_sweep_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(bsm_cpu, "EXPERT_STATS_DIR", str(tmp_path))
    m = make_module(tmp_path)
    for _ in range(bsm_cpu._STATS_MIN_STEPS):
        m._split_swap_tick()
    m._split_sweep_layer_reset()
    assert m._swap_tick_count == 0
    bsm_cpu.save_expert_stats(m.config.infer_params)
    with open(bsm_cpu._expert_stats_path(m.config, m.num_experts)) as f:
        saved = json.load(f)
    assert saved["layers"][m.key] == [3.0, 7.0, 2.0, 5.0]


def test_periodic_save_survives_sweep_resets(tmp_path, monkeypatch):
    m = make_module(tmp_path)
    saved = []
    monkeypatch.setattr(bsm_cpu, "_STATS_SAVE_INTERVAL", 4)
    monkeypatch.setattr(bsm_cpu, "save_expert_stats", lambda ip: saved.append(m._split_step_count))
    for i in range(8):
        m._split_swap_tick()
        if i % 2 == 1:
            m._split_sweep_layer_reset()
    assert saved == [4, 8]


def test_short_session_preserves_existing_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(bsm_cpu, "EXPERT_STATS_DIR", str(tmp_path))
    m = make_module(tmp_path)
    path = bsm_cpu._expert_stats_path(m.config, m.num_experts)
    original = '{"layers": {"settled": [1, 2, 3, 4]}}'
    with open(path, "w") as f:
        f.write(original)
    for _ in range(bsm_cpu._STATS_MIN_STEPS - 1):
        m._split_swap_tick()
    bsm_cpu.save_expert_stats(m.config.infer_params)
    with open(path) as f:
        assert f.read() == original


def test_main_unload_saves_before_shared_mtp_registry_loses_layers(tmp_path, monkeypatch):
    monkeypatch.setattr(bsm_cpu, "EXPERT_STATS_DIR", str(tmp_path))
    draft = make_module(tmp_path)
    draft.key = "mtp.layers.0.mlp"
    draft._split_step_count = bsm_cpu._STATS_MIN_STEPS
    main = make_module(tmp_path)
    main.config = draft.config
    registry = draft.config.infer_params.moe_cpu_swap_modules
    registry.append(main)
    main.unload = lambda: registry.remove(main)
    draft.config.stc = SimpleNamespace(release_arena = lambda: None)
    model = SimpleNamespace(config = draft.config, modules = [main], cache_weakrefs = {},
                            unload_tp = lambda: None)
    Model.unload(model)
    assert registry == [draft]
    with open(bsm_cpu._expert_stats_path(main.config, main.num_experts)) as f:
        saved = json.load(f)
    assert set(saved["layers"]) == {draft.key, main.key}
    assert saved["layers"][main.key] == [3.0, 7.0, 2.0, 5.0]
