"""Serve one ACT action at a time on stdin/stdout.

Each request is a JSON line with ``state_deg`` and optional base64 RGB images
``up`` and ``side`` (height 480, width 640). Missing images stay black.
Nothing is sent to a robot.
"""

from __future__ import annotations

import base64
import json
import sys

MODEL_ID = "samarth-shukla6/act-svla-so101-pickplace"


def decode_rgb(encoded: str):
    import torch

    raw = base64.b64decode(encoded)
    image = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(480, 640, 3)
    return image.permute(2, 0, 1).float() / 255.0


def main() -> None:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(MODEL_ID)
    config.device = "cpu"
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(MODEL_ID, config=config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.reset()
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        state = request.get("state_deg")
        if not isinstance(state, list) or len(state) != 6:
            print(json.dumps({"error": "state_deg must be 6 numbers"}), flush=True)
            continue
        observation = {
            key: torch.zeros(tuple(feature.shape), dtype=torch.float32)
            for key, feature in policy.config.input_features.items()
        }
        observation["observation.state"] = torch.tensor(state, dtype=torch.float32)
        for name in ("up", "side"):
            encoded = request.get(name)
            if isinstance(encoded, str):
                observation[f"observation.images.{name}"] = decode_rgb(encoded)
        batch = preprocessor(observation)
        with torch.inference_mode():
            normalized = policy.select_action(batch)
        action = postprocessor(normalized).detach().cpu().reshape(-1)
        print(json.dumps({"action_deg": [float(value) for value in action]}), flush=True)


if __name__ == "__main__":
    main()
