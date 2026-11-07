"""Load the public LeRobot ACT checkpoint and run one dummy forward pass.

The checkpoint is ``samarth-shukla6/act-svla-so101-pickplace`` (LeRobot 0.4 ACT).
This script does not talk to Isaac Lab and does not move a robot. It stays on CPU
unless ``--device cuda`` is passed, so a GPU training job can keep the RTX 3060.

Requires LeRobot 0.4.2 (the release current when this checkpoint was uploaded).
The workspace's later LeRobot pin is for dataset recording, not this policy file.

Example:
    python scripts/load_act_svla_so101.py --device cpu
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MODEL_ID = "samarth-shukla6/act-svla-so101-pickplace"
JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="Hub repo id or a local snapshot directory")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="cpu leaves the laptop GPU free (default)",
    )
    return parser.parse_args()


def weight_file(model: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path / "model.safetensors"
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model, "model.safetensors"))


def assert_weights_match(policy, filename: Path) -> int:
    import torch
    from safetensors import safe_open

    state = policy.state_dict()
    mismatches: list[str] = []
    missing: list[str] = []
    with safe_open(filename, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            if key not in state:
                missing.append(key)
                continue
            saved = handle.get_tensor(key)
            loaded = state[key].detach().cpu()
            if saved.shape != loaded.shape or not torch.equal(saved, loaded):
                mismatches.append(key)
    if missing or mismatches:
        raise RuntimeError(f"Checkpoint did not load cleanly: missing={missing[:8]} mismatches={mismatches[:8]}")
    return len(keys)


def dummy_observation(policy) -> dict:
    import torch

    observation = {}
    for key, feature in policy.config.input_features.items():
        observation[key] = torch.zeros(tuple(feature.shape), dtype=torch.float32)
    return observation


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        # This process must not allocate on the GPU while Isaac Lab training is using it.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from lerobot import __version__ as lerobot_version
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    # PreTrainedConfig, not ACTConfig: the saved file's "type" field is how LeRobot
    # picks the subclass, and ACTConfig.from_pretrained rejects that field.
    config = PreTrainedConfig.from_pretrained(args.model)
    config.device = args.device
    # The uploaded model.safetensors already contains the ResNet-18 body.
    config.pretrained_backbone_weights = None

    policy = ACTPolicy.from_pretrained(args.model, config=config)
    policy.eval()
    copied = assert_weights_match(policy, weight_file(args.model))
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.model,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": args.device}},
    )

    policy.reset()
    batch = preprocessor(dummy_observation(policy))
    with torch.inference_mode():
        normalized = policy.select_action(batch)
    action = postprocessor(normalized).detach().cpu().reshape(-1)

    if action.numel() != len(JOINT_NAMES) or not torch.isfinite(action).all():
        raise RuntimeError(f"Unexpected action tensor: shape={tuple(action.shape)} values={action.tolist()}")

    print(f"lerobot {lerobot_version}")
    print(f"policy type: {policy.config.type}")
    print(f"parameters: {parameter_count}")
    print(f"safetensors tensors copied: {copied}")
    print(f"device: {next(policy.parameters()).device}")
    print(f"action chunk remaining after one step: {len(policy._action_queue)}")
    print("unnormalized action (dataset joint units, not sent to a robot):")
    for name, value in zip(JOINT_NAMES, action.tolist(), strict=True):
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
