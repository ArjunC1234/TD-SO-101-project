# SO-101 simulator control

Before controlling or planning manipulation with this arm, read
[docs/SO101_CONTROL_GUIDE.md](docs/SO101_CONTROL_GUIDE.md). It records the calibrated joint conventions,
available tools, measured accuracy, reach limitations, and grasp verification procedure.

Use current state and absolute joint targets for precise motion. Treat vague manipulation requests as
observe, plan, approach, grasp, and verify tasks. Never equate a tool's successful return with a successful
pickup: verify object elevation and retention. The connected MCP tools control simulation only.
