"""Decode-focused placement must ignore prefill counts without changing expert routing."""
import pytest
import torch

import exllamav3.modules.block_sparse_mlp_cpu as bsm_cpu


@pytest.mark.parametrize("decode_only", [False, True])
@torch.inference_mode()
def test_prefill_does_not_pollute_decode_profile(device, monkeypatch, decode_only):
    monkeypatch.setattr(bsm_cpu, "_split_stats_decode_only", decode_only)
    m = bsm_cpu.BlockSparseMLP_CPU()
    m.cpu_split_first = 4
    m._split_map = torch.tensor([4, 1, 6, 2, 5, 0, 7, 3], device = device)
    m._split_hist = torch.zeros(8, device = device)
    m._split_prefill_hist = torch.zeros_like(m._split_hist)
    m._split_selcpu_t = None
    expected = torch.zeros_like(m._split_hist)
    for rows in (2, 64, 1):
        selected = torch.arange(rows * 2, device = device).reshape(rows, 2) % 8
        physical = m._split_map[selected]
        if not decode_only or rows <= 2:
            expected += torch.bincount(selected.flatten(), minlength = 8).float()
        cpu = m._split_translate(selected)
        assert torch.equal(selected, physical)
        assert torch.equal(cpu, torch.where(physical >= 4, physical - 4, -1))
        assert torch.equal(m._split_hist, expected)
