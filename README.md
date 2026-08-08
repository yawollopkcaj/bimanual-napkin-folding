# Bimanual Napkin Folding Robot

**Two 6-DOF robot arms that learn to fold linen napkins from human demonstration — no simulator, no reward engineering, ~$1k of hardware.**

`Python` · `PyTorch` · `LeRobot` · `SmolVLA` · `ACT` · `SARM` · `Hugging Face Hub` · `SO-101`

[**▶ Watch the demo**](https://x.com/jack_polloway/status/2060747250800988349?s=20) · [**Read the paper**](paper/napkin_folding.pdf)

<!-- TODO: hero image — ~/Downloads/real_robot.jpeg -->

---

## What it does

Two 6-DOF SO-101 arms and a three-camera RGB stack learn decorative napkin folding entirely
from human teleoperation demonstrations, running on the open-source
[LeRobot](https://github.com/huggingface/lerobot) framework.

Cloth is the hard case in manipulation. A rigid object has six degrees of freedom; a napkin has
effectively infinite, and linen's dynamics are non-linear enough that simulation-trained policies
do not survive contact with a real table. That rules out the usual sim-to-real reinforcement
learning pipeline, so every policy here is learned directly from real demonstrations on real cloth.

## Results

| | |
|---|---|
| **SmolVLA** (450M-param vision-language-action) | **70%** end-to-end fold success |
| Mean cycle time | 40.2 s |
| Mean edge deviation | 6 mm |
| **ACT** (Action-Chunking Transformer), identical data | **0 / 20** — total failure |
| Tangential-scoop end effector | **0%** cloth crumple, all conditions |
| Force-closure end effectors | 47–100% cloth crumple |
| Dataset | 499 episodes · 550k frames · 3 camera views · 14 action dimensions |

Two results are worth more than the headline success rate.

### 1. Gripper geometry beats gripper force

Force-closure grippers — the default choice, squeeze the thing you want to hold — crumple linen
in 47% to 100% of grasp attempts, because pinching a flat sheet drags material inward before the
fingers close. Replacing force closure with a **geometry-entrapment "tangential scoop"** that
slides under the cloth and traps it against the end effector's own shape drove crumpling to
**zero across every condition tested**. No control-side fix produced anything comparable. The
mechanical design was the bottleneck.

### 2. The stage-composition bottleneck

ACT failing 0-for-20 while SmolVLA hits 70% on the *same* dataset looks like a straightforward
model-capacity story. It isn't.

Stage-isolated ablations show ACT trained on **pickup-only** data succeeds reliably, and ACT
trained on **fold-only** data succeeds reliably. It is only the single monolithic policy spanning
both that collapses. The failure sits precisely at the pickup-to-fold phase boundary, where
pickup-like and fold-like actions are both plausible under near-identical observations — a
multimodal action distribution that ACT's objective cannot resolve, and that averages out into
motion belonging to neither stage.

This is a composition problem, not a capacity problem, and it reframes what "the policy failed"
means for multi-stage manipulation tasks. The paper proposes stage-aware reward modeling and
flow-matching policies as the paths forward.

## My contributions

This was a seven-person team project. My work concentrated on two areas:

**Policy training and deployment**
- Trained the vision-language-action and ACT policy families on cloud GPUs.
- Deployed trained policies for real-time closed-loop inference on the physical arms.
- Owned the pipeline end to end, from raw collected data through to a policy running on hardware.

**Dataset and teleoperation**
- Collected the teleoperation dataset — 3 synchronized camera views, 14 action dimensions.
- Applied Stage-Aware Reward Modeling (SARM) annotation to improve data quality.

**Also**
- Identified and patched two bugs in the upstream LeRobot codebase.
  <!-- TODO: name these two bugs specifically — a concrete named bug is the single most
       credible line available for this page. Link the PRs/commits if they exist. -->

## System architecture

```
2× SO-101 arm (6-DOF)  ──┐
                         ├──► LeRobot ──► SmolVLA / ACT policy ──► closed-loop inference @ real time
3× RGB camera ───────────┘                        ▲
                                                  │
              human teleoperation ──► 499-episode dataset ──► SARM annotation
```

<!-- TODO: replace with the architecture figure from the paper's figs/ -->

## Paper

**Bimanual Imitation Learning for Decorative Napkin Folding on Low-Cost Hardware: End-Effector
Geometry and the Stage-Composition Bottleneck**

Joshua Himmens, Sloan Sobie, Dawson March, Genevieve Merz, Cameron Powell, Jaden Legate, Jack Polloway
*University of British Columbia, Vancouver, BC, Canada*

[📄 Full paper (PDF)](paper/napkin_folding.pdf)

## Notes on this repo

This is a project write-up, not the implementation. The system is built on a fork of
[huggingface/lerobot](https://github.com/huggingface/lerobot); this repo holds the paper and a
summary of the work.
