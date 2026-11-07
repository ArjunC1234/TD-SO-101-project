# Upstream sources and local changes

## IsaacLabTutorial

`IsaacLabTutorial/` is a source snapshot of
[isaac-sim/IsaacLabTutorial](https://github.com/isaac-sim/IsaacLabTutorial), based on commit
`7efef771fa0bfbcb9683193ff100ce9f015fb17e`, with our simulator control, calibration, MCP, pickup curriculum,
tests, and ACT integration changes. Its upstream [Apache-2.0 license](IsaacLabTutorial/LICENSE) is retained.
The nested source snapshot has no nested Git metadata. Small runtime assets are stored directly in Git;
its publication copy removes Git LFS filters so fresh clones have the assets without another asset server.

## STS32XX firmware tools

[roboninecom/STS32XX-Firmware](https://github.com/roboninecom/STS32XX-Firmware), commit
`a995f77cf83175060bfa8fb908347448a5ae45eb`, is an external dependency, not our firmware implementation.
[patches/sts32xx-windows.patch](patches/sts32xx-windows.patch) contains our three-file Windows compatibility
changes. `scripts/setup-firmware.ps1` fetches that revision and applies the patch. Upstream component
license notices remain authoritative; the project-root license does not relicense external code or assets.

## Other references

- [art-e-fact/isaaclab-so101](https://github.com/art-e-fact/isaaclab-so101), commit
  `0284f4a2c31fd86282c58914a524595e6b71e3e7`, was an exploratory checkout with no local modifications.
  It is linked here instead of copying the unchanged repository.
- [samarth-shukla6/act-svla-so101-pickplace](https://huggingface.co/samarth-shukla6/act-svla-so101-pickplace)
  is the public ACT checkpoint used by the experimental loader. Model weights are downloaded separately;
  refer to that model card for its terms, training setup, and limitations.
- [LeRobot](https://github.com/huggingface/lerobot) supplies the ACT policy implementation. The playback
  experiments expect version 0.4.2; they do not vendor LeRobot or its model weights.
