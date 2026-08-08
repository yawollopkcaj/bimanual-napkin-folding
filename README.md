# Bimanual Napkin Folding Robot

**Two 6-DOF robot arms that learn to fold linen napkins from human demonstration. No simulator, no reward engineering, open low-cost hardware.**

[**▶ Watch the demo**](https://x.com/jack_polloway/status/2060747250800988349?s=20) | [**Read the paper**](paper/napkin_folding.pdf) | [**🤗 Datasets and models**](#open-source-contributions)

> **Everything is public.** 499 episodes and ~545k frames of real bimanual cloth manipulation,
> plus the trained policy weights, released on the Hugging Face Hub in standard LeRobot format
> under Apache-2.0.
> [Jump to the open source ↓](#open-source-contributions)

<img src="assets/prototype.jpeg" width="520" alt="The bimanual napkin folding rig: two SO-101 follower arms on a hardboard base, overhead camera gantry, leader arms in the foreground">

---

## What it does

Two 6-DOF [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arms and three RGB cameras learn to
fold a linen napkin from human teleoperation, running on
[LeRobot](https://github.com/huggingface/lerobot).

Cloth is the hard case in manipulation. A rigid object has six degrees of freedom. A napkin has
effectively infinite, and linen's dynamics depend on weave, weight, and finish in ways current
physics engines do not reproduce. That kills the standard recipe, which is to train a policy in
simulation with RL and transfer it to hardware. The sim-to-real gap for cloth is too wide to cross.
So every policy here is learned directly from real demonstrations on real cloth.

The task is narrow on purpose. Premium hospitality venues spend one to three hours of paid staff
time a day on decorative folding, so it is a real workload, and it bounds the perception and
workspace problem to a table.

## Results

We set the bar at 80% first-attempt success and did not clear it. The best policy folds 70% of the
time.

The bench test mirrors the deployment requirement: a freshly ironed 20"×20" linen napkin, placed
flat in a randomized workspace position, 20 trials per policy. A trial counts as a success only if
it meets *both* the cycle-time and edge-accuracy thresholds.

| Requirement | Target | SmolVLA achieved |
|---|---|---|
| First-attempt success rate | ≥ 80% (16/20) | **70%** (14/20) |
| Cycle time | ≤ 90 s | **40.2 s** mean |
| Fold accuracy (edge deviation) | ≤ 10 mm | **6 mm** mean |
| Footprint | ≤ 36×36 in | met |

| Policy | Success | Rate | Cycle | Edge dev. |
|---|---|---|---|---|
| **SmolVLA** (450M VLA) [🤗 weights](https://huggingface.co/jhimmens/smolvla-napkin-fold) | 14 / 20 | **70%** | 40.2 s | 6 mm |
| **ACT** (Action-Chunking Transformer), identical data | 0 / 20 | **0%** | n/a | n/a |

SmolVLA cleared the speed and accuracy budgets with room to spare and missed the success threshold.
Five of its six failures were bad initial grasps and one was a mid-fold slip. Grasp acquisition is
on the critical path, not folding.

Two things we found are worth more than the headline number.

### 1. Gripper geometry beats gripper force

Force-closure grippers squeeze the thing you want to hold. That is the default choice, and on thin
linen it is the wrong one. Pinching a flat sheet drags material inward before the fingers close. We
3D-printed four single-DOF end effectors in PLA+ and ran teleoperated pickup trials on flat and
pre-folded napkins, approaching from both sides:

| Design | Type | Pickup, flat | Pickup, folded | **Crumple, flat** | **Crumple, folded** |
|---|---|---|---|---|---|
| OG (narrow-tip pinch) | force closure | 4.03 s | 3.60 s | 100% | 100% |
| EEJ (thick-jaw pinch) | force closure | 12.68 s | 9.44 s | 77% | 47% |
| EED (wide parallel pads) | force closure | 5.27 s | 4.17 s | 100% | 100% |
| **EES (tangential scoop)** | **geometry entrapment** | 5.74 s | 4.57 s | **0%** | **0%** |

The scoop slides a curved surface under the cloth edge and lifts before closing, so it captures the
napkin without ever compressing it. Zero crumpling across every condition we tested. It pays a 1 to
2 second pickup penalty against the fastest design, which is nothing against a 90 second budget.

That trade was worth taking because crumpling propagates. A crumpled grasp produces corner
misalignment and crease defects the folding policy cannot correct later. We tried fixing this on the
control side and got nowhere. The bottleneck was mechanical.

### 2. The stage-composition bottleneck

ACT going 0-for-20 while SmolVLA hits 70% on the *same* data looks like a model capacity story. It
is not.

We trained stage-isolated variants and tested them in isolation:

| Variant | Training data | Outcome |
|---|---|---|
| ACT, pickup only | 185 episodes | reliable grasps in nearly all trials |
| ACT, fold only | pre-grasped napkin | folds reliably, clean diagonal crease |
| ACT, full task | 314 episodes | **fails to compose the two** |

Each half is learnable from this data. The whole thing is not, at least not for one monolithic ACT
policy. The failure sits right at the pickup-to-fold boundary, where a near-identical observation
can demand two different actions: keep grasping, or start folding. ACT predicts one action chunk per
timestep and cannot represent that ambiguity, so it collapses the two modes together and produces
motion that belongs to neither.

SmolVLA's pretrained vision-language backbone gives richer observation features that partially
disambiguate the boundary, which is why it works at all. Its 70% ceiling says better representations
alone do not fix stage composition.

<img src="assets/loss.png" width="620" alt="Training loss on log-log axes for ten runs, every one descending smoothly from roughly 50 down to between 0.05 and 0.3 across 200k gradient steps">

*Training loss, log-log. Gradient step on the x axis (100 to 200k), behavior-cloning loss on the y.
Each line is one training run logged during ACT development. Note that every single one of them descends
smoothly.*

**The loss-curve trap.** Every run converged. Loss falls monotonically across nearly three orders of
magnitude with no divergence and no instability, and pushing the longest run out to 200k gradient
steps bought nothing further. That is exactly what a healthy, well-tuned training pipeline is
supposed to look like, and the full-task policy it produced succeeds on 0% of trials. Lower loss did
not help either. The run that converged furthest is not a better folder.

Behavior-cloning loss measures prediction error on demonstrated action chunks. It does not measure
task success. A converged loss is entirely consistent with total end-to-end failure when the failure
is one of composition rather than representation. If we had trusted these curves we would have
concluded the model was fine. Evaluate on the task, not on the loss.

<img src="assets/completed_fold.jpeg" width="460" alt="Overhead view of a completed fold: clean diagonal crease with aligned corners">

*Fold produced by the fold-only policy given a stable grasp. When the pickup bottleneck is
bypassed, fold quality already meets the presentation standard. The system's ceiling is set by
grasp acquisition and stage composition, not by the folding motion.*

## System

<img src="assets/sysarch.png" width="600" alt="System dataflow diagram: inference path from cameras and joint poses through the control PC to the follower arms, and data-collection path from teleoperated leader arms to the logged dataset">

**Hardware.** Two SO-101 6-DOF arms as followers with custom end effectors, plus a matched pair of
SO-101 leader arms for teleoperation. All four bolt to one continuous hardboard base. It is stiff
enough to hold the arms square through a fold and still cheap, portable at around 12 kg, and easy to
modify. We started on an optical breadboard, which was rigid but immobile and expensive, and moved
off it.

**Workspace geometry.** We tried two arm-spacing layouts and picked outer-edge mounting over
center-mounting. It lowers inter-arm collision risk during the fold-over motion and gives each arm
enough independent reach to capture a napkin corner.

<img src="assets/armspacing.png" width="620" alt="Two-panel reach comparison. Left: both arms mounted on the same edge, their semicircular reach envelopes covering only the near half of the workspace. Right: arms mounted on opposite edges, reach circles covering the whole board and overlapping in a tall lens down the center.">

**Perception.** Three RGB cameras at 640×480 and 30 fps. One overhead on the gantry for global
napkin pose, two on the wrists for local gripper-to-cloth geometry. The redundancy means a critical
feature is unlikely to be occluded in every view at once. We hold camera viewpoints, lighting, and
arm-base positions fixed across sessions to limit distribution shift between demonstration and
deployment.

<img src="assets/camera_frames.png" width="600" alt="Synchronized frames from the three-camera stack: overhead view plus two wrist-mounted views">

**Real-time inference at 30 Hz.** LeRobot's default inference path does image preprocessing (resize,
normalize, color-space conversion) on the CPU before handing observations to the GPU. On our first
workstation that starved the GPU and dragged the control loop below target, and at low loop rates
the gap between observation and action grows until the motion turns jerky and unresponsive. I
replaced it with a path that hands raw camera frames straight to the GPU and lets the policy's own
vision encoder preprocess inside the forward pass. That removed the bottleneck and put us back at
30 Hz.

## Data and training

**Dataset.** 499 episodes and 544,906 synchronized frames, all collected by hand. An operator drives
the SO-101 leader arms, the followers mirror the leader poses, and we log 12-dimensional joint-angle
trajectories together with RGB from all three cameras at 30 fps. All of it is
[public](#open-source-contributions).

| Split | Episodes | Frames | Purpose |
|---|---|---|---|
| [Full task](https://huggingface.co/datasets/jhimmens/linique-v2) | 314 | 496,866 | pick up a flat napkin, align corners, execute the complete fold |
| [Pickup only](https://huggingface.co/datasets/jhimmens/linique-v2-pickup) | 185 | 48,040 | added after early ACT runs showed pickup as the dominant failure mode |
| [**Combined**](https://huggingface.co/datasets/jhimmens/linique-v2-fold-pickup) | **499** | **544,906** | both tasks in one corpus |

**Policy families.** ACT is data-efficient with a small memory footprint and emits 100-action chunks
per inference step. We added temporal ensembling at inference (coefficient 0.01) and cosine-annealed
learning-rate scheduling, and both improved trajectory smoothness and training stability over the
baseline. SmolVLA at 450M params and roughly 4 GB of VRAM was the best policy we trained. We tried
X-VLA early and dropped it after it commanded trajectories that drove the arms into the table, which
was an action-space calibration mismatch. Diffusion Policy is the obvious candidate for the
pickup-to-fold branch, because modeling the full multimodal action distribution is exactly what is
missing, and we did not get to it.

The whole thing is constrained by an edge inference budget. A deployable policy has to fit in
roughly 8 GB of VRAM, something like an NVIDIA Jetson Orin, which rules out the largest VLAs.

**Training infrastructure.** We ran across local consumer GPUs (RTX 2070, then a 4070), a cloud
provider (H100 and RTX 5090), and an institutional HPC cluster (A40). Local turned out to have the
fastest end-to-end iteration cycle, which we did not expect. Staging a 545k-frame image dataset and
sitting in HPC job queues ate the compute advantage for short experiments. We saved the cloud GPUs
for long runs like SmolVLA, where the per-step speedup paid for the transfer.

**A known limitation.** We could not merge the pickup-only episodes into SmolVLA training. Episode
metadata and frame indexing differ between the two collection runs, and sampling across the
partition boundary crashed the dataloader intermittently. So SmolVLA trained on the 314 full-task
episodes alone. Fixing that schema mismatch is the highest-leverage thing left, because the data we
are dropping is exactly the pickup coverage the model is worst at.

## Future work

- **Combined-corpus training.** Fix the dataset schema mismatch and train SmolVLA on all 499
  episodes rather than 314.
- **Stage-aware decomposition and RL fine-tuning.** Stage-Aware Reward Modeling (SARM) decomposes
  the task into stages (pickup, align, fold, release) and trains a per-stage reward model, with a
  meta-controller sequencing them. That turns the sparse binary task reward into a dense signal,
  which is a prerequisite for RL fine-tuning on top of an imitation-learned initialization and a
  principled remedy for the composition failure above.
- **A continuous fold-quality metric.** Fold quality is currently scored only as binary
  success/failure. A vision-based continuous metric would give gradient information for both
  evaluation and RL fine-tuning.

## My contributions

This was a seven-person team project. My work concentrated on two areas:

**Policy training and deployment**
- Trained the vision-language-action and ACT policy families on cloud GPUs.
- Deployed trained policies for real-time closed-loop inference on the physical arms.
- **Diagnosed and patched the inference-path bottleneck in upstream LeRobot.** Its default path ran
  image preprocessing (resize, normalize, color-space conversion) on the CPU before handing
  observations to the GPU. That starved the GPU, pulled the control loop below its 30 Hz target, and
  widened the observation-to-action gap until arm motion turned visibly jerky. I rerouted raw camera
  frames straight to the GPU so the policy's own vision encoder preprocesses inside the forward
  pass, which removed the bottleneck and restored the full 30 Hz loop.
- Owned the pipeline end to end, from raw collected data through to a policy running on hardware.

**Dataset and teleoperation**
- Collected the teleoperation dataset across 3 synchronized camera views. That corpus is now public
  as [`linique-v2`](https://huggingface.co/datasets/jhimmens/linique-v2) and
  [`linique-v2-fold-pickup`](https://huggingface.co/datasets/jhimmens/linique-v2-fold-pickup) on the
  Hugging Face Hub.

## Open source contributions

Real bimanual cloth-manipulation data is scarce, and it is the expensive part of a project like
this. Roughly five hours of a human driving leader arms, one napkin at a time. It is all public, in
standard [LeRobot](https://github.com/huggingface/lerobot) format, so it loads in two lines with no
conversion.

**Datasets** ([full profile](https://huggingface.co/jhimmens))

| Dataset | Episodes | Frames | What it is |
|---|---|---|---|
| [**linique-v2-fold-pickup**](https://huggingface.co/datasets/jhimmens/linique-v2-fold-pickup) | **499** | **544,906** | The complete corpus, both tasks. Start here. Citable: [`10.57967/hf/8174`](https://doi.org/10.57967/hf/8174) |
| [linique-v2](https://huggingface.co/datasets/jhimmens/linique-v2) | 314 | 496,866 | Full-task only: pick up a flat napkin, align corners, complete the fold |
| [linique-v2-pickup](https://huggingface.co/datasets/jhimmens/linique-v2-pickup) | 185 | 48,040 | Pickup and corner-capture phase in isolation |
| [linique-v2-pickup-force](https://huggingface.co/datasets/jhimmens/linique-v2-pickup-force) | 189 | 50,077 | Pickup with force sensing: 24-dim state instead of 12 |
| [linique-v2-combined-force](https://huggingface.co/datasets/jhimmens/linique-v2-combined-force) † | 503 | 546,943 | Full corpus with force sensing, 24-dim state |
| [linique-v2-load-padded](https://huggingface.co/datasets/jhimmens/linique-v2-load-padded) † | 314 | 496,866 | Full-task with padded load channels, 24-dim state |

Every episode carries three synchronized 640×480 RGB streams (one overhead, two wrist-mounted) at
30 fps alongside 12-dimensional bimanual joint trajectories, recorded on a `bi_so_follower` robot.

**Trained policies**

| Model | Base | Trained on |
|---|---|---|
| [**smolvla-napkin-fold**](https://huggingface.co/jhimmens/smolvla-napkin-fold) | [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) | `linique-v2`. This is the 70% policy from the results table above. |
| [xvla-linique-v2-fold-pickup](https://huggingface.co/jhimmens/xvla-linique-v2-fold-pickup) | X-VLA | `linique-v2-fold-pickup` |

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# the full 499-episode corpus, LeRobot dataset format v3.0
dataset = LeRobotDataset("jhimmens/linique-v2-fold-pickup")
```

Or browse it in the [dataset
visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset?path=jhimmens/linique-v2-fold-pickup)
without downloading anything.

The full corpus, the two phase splits, the force-sensing pickup set, and both models are
Apache-2.0, so they are usable commercially and for derivative work without asking. If you train
something better on this data, that is the point of releasing it.

† These two variants are public but do not yet carry a declared license, so treat them as
all-rights-reserved until one is added.

## Paper

**Bimanual Imitation Learning for Decorative Napkin Folding on Low-Cost Hardware: End-Effector
Geometry and the Stage-Composition Bottleneck**

Joshua Himmens, Sloan Sobie, Dawson March, Genevieve Merz, Cameron Powell, Jaden Legate, Jack Polloway
*University of British Columbia, Vancouver, BC, Canada*

[📄 Full paper (PDF)](paper/napkin_folding.pdf)
## Notes on this repo

This is a project write-up, not the implementation. The system runs on a fork of
[huggingface/lerobot](https://github.com/huggingface/lerobot). This repo holds the paper, the figures, and a summary of the work.
