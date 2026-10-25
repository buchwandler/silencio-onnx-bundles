from __future__ import annotations

from scripts.silencio_contracts import CACHE_ABI, COMPONENT_CONTRACTS, COMPONENT_ORDER


def test_component_order_covers_every_source_backed_abi() -> None:
    assert tuple(COMPONENT_CONTRACTS) == COMPONENT_ORDER
    assert len(COMPONENT_ORDER) == 5


def test_dit_cache_contract_distinguishes_past_and_updated_frames() -> None:
    contract = COMPONENT_CONTRACTS["dit_step"]
    cache_inputs = [name for name in contract["inputs"] if name.startswith("cache_")]
    cache_outputs = [name for name in contract["outputs"] if name.startswith("updated_cache_")]

    assert CACHE_ABI["version"] == 1
    assert len(cache_inputs) == len(cache_outputs) == 8
    assert all(contract["inputs"][name]["shape"][2] == "past_frames" for name in cache_inputs)
    assert all(
        contract["outputs"][name]["shape"][2] == "updated_cache_frames"
        for name in cache_outputs
    )
    assert all(contract["dynamic_axes"][name][2] == "past_frames" for name in cache_inputs)
    assert all(
        contract["dynamic_axes"][name][2] == "updated_cache_frames"
        for name in cache_outputs
    )
    assert contract["inputs"]["position_offset"]["shape"] == []
