"""Unit tests for the simulator-independent SO-101 MCP front process."""

import math

import pytest

from isaaclab_tutorial.mcp_server import _validate_ticks, _validated_actions
from isaaclab_tutorial.so101_joints import JOINTS


def test_validated_actions_clamps_in_canonical_order():
    values = _validated_actions([2.0, -2.0, 0.5, 0.0, 0.25, -0.25])
    assert values == [1.0, -1.0, 0.5, 0.0, 0.25, -0.25]
    assert len(values) == len(JOINTS)


@pytest.mark.parametrize("actions", [[0.0] * 5, [0.0] * 7, [0.0, 0.0, math.nan, 0.0, 0.0, 0.0]])
def test_validated_actions_rejects_invalid_values(actions):
    with pytest.raises(ValueError):
        _validated_actions(actions)


@pytest.mark.parametrize("ticks", [1, 30, 3_000])
def test_validate_ticks_accepts_bounded_values(ticks):
    assert _validate_ticks(ticks) == ticks


@pytest.mark.parametrize("ticks", [0, -1, 3_001])
def test_validate_ticks_rejects_out_of_range_values(ticks):
    with pytest.raises(ValueError, match="between 1 and 3000"):
        _validate_ticks(ticks)
