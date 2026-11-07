"""Exercise the SO-101 MCP server through its real STDIO protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from isaaclab_tutorial.so101_joints import JOINTS


def payload(result):
    if result.isError:
        raise RuntimeError(result.content)
    return json.loads(next(item.text for item in result.content if item.type == "text"))


async def run_smoke_test(*, simulator: bool, device: str) -> int:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "isaaclab_tutorial.mcp_server"],
    )
    print("Starting MCP process...", flush=True)
    async with stdio_client(parameters) as (read_stream, write_stream):
        print("MCP process started; initializing session...", flush=True)
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("Session initialized; listing tools...", flush=True)
            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            result = {"tools": tool_names}
            assert "move_joint_positions" in tool_names

            if simulator:
                started = await session.call_tool("start_simulator", {"visualizer": "none", "device": device})
                initial = payload(started)["state"]
                target = [initial["joints"][name]["position_rad"] for name in JOINTS]
                target[0] += 0.10
                absolute = payload(
                    await session.call_tool("move_joint_positions", {"positions_rad": target, "ticks": 60})
                )
                assert absolute["reached_joint_tolerance"], absolute["command_error_rad"]
                assert max(abs(v) for v in absolute["command_error_rad"]) <= 0.005
                assert len(absolute["vial_grasp_point_m"]) == 3
                assert set(absolute["jaw_contacts"]) == {"fixed", "moving"}
                invalid = target.copy()
                invalid[0] = absolute["joint_limits_rad"][0][1] + 0.1
                rejected = await session.call_tool("move_joint_positions", {"positions_rad": invalid})
                assert rejected.isError, "Out-of-limit target was accepted"
                result["absolute_command_max_error_rad"] = max(abs(v) for v in absolute["command_error_rad"])
                moved = await session.call_tool(
                    "pulse_joint",
                    {"joint": "shoulder_pan", "value": 0.5, "ticks": 30},
                )
                if moved.isError:
                    raise RuntimeError(moved.content)
                state = await session.call_tool("get_robot_state", {})
                stopped = await session.call_tool("stop_simulator", {})
                if state.isError or stopped.isError:
                    raise RuntimeError(state.content if state.isError else stopped.content)
                result["simulator_round_trip"] = True

            print(f"SO101_MCP_SMOKE_RESULT={json.dumps(result, sort_keys=True)}")
            return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", action="store_true", help="Also launch Isaac and move a joint headlessly.")
    parser.add_argument("--device", default="cuda:0", help="Isaac device used with --simulator.")
    args = parser.parse_args(argv)
    return asyncio.run(run_smoke_test(simulator=args.simulator, device=args.device))


if __name__ == "__main__":
    raise SystemExit(main())
