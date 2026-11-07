"""Canonical SO-101 joint names shared by terminal and MCP controllers."""

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINT_INDEX = {name: index for index, name in enumerate(JOINTS)}
