"""Unit tests for the simulator-independent manual-control command layer."""

import pytest

from isaaclab_tutorial.manual_control import JOINTS, ActionState, CommandError, parse_command


def test_parse_action_and_aliases():
    command = parse_command("action 1 0.5 0 -0.5 0 -1")
    assert command.kind == "action"
    assert command.values == (1.0, 0.5, 0.0, -0.5, 0.0, -1.0)
    assert parse_command("status").kind == "state"
    assert parse_command("home").kind == "reset"
    assert parse_command("exit").kind == "quit"


def test_parse_pulse_and_gripper_shortcuts():
    pulse = parse_command("pulse wrist_roll 0.25 12")
    assert (pulse.kind, pulse.joint, pulse.value, pulse.ticks) == ("pulse", "wrist_roll", 0.25, 12)
    assert parse_command("open").value == 1.0
    assert parse_command("open 7").ticks == 7
    assert parse_command("close").value == -1.0


@pytest.mark.parametrize(
    "line, message",
    [
        ("", "enter a command"),
        ("action 1 2", "exactly 6"),
        ("action 0 0 0 nan 0 0", "finite"),
        ("pulse elbow 1 10", "unknown joint"),
        ("pulse elbow_flex 1 0", "between 1"),
        ("open 1 2", "at most one"),
        ("python print('no')", "unknown command"),
    ],
)
def test_parse_rejects_invalid_commands(line, message):
    with pytest.raises(CommandError, match=message):
        parse_command(line)


def test_action_state_clamps_and_holds():
    state = ActionState()
    state.hold((2.0, -2.0, 0.5, 0.0, 0.0, 0.0))
    assert state.values == (1.0, -1.0, 0.5, 0.0, 0.0, 0.0)
    assert state.remaining_ticks is None
    state.after_step()
    assert state.values[0] == 1.0


def test_action_state_pulse_expires_to_zero():
    state = ActionState()
    state.pulse("gripper", -0.75, 2)
    assert state.values[-1] == -0.75
    assert state.remaining_ticks == 2
    state.after_step()
    assert state.values[-1] == -0.75
    assert state.remaining_ticks == 1
    state.after_step()
    assert state.values == (0.0,) * len(JOINTS)
    assert state.remaining_ticks is None
