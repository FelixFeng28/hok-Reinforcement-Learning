# Honor of Kings 1v1 Reinforcement Learning

**Reward shaping, survival curriculum, and fixed-opponent evaluation**

How can a reinforcement-learning agent learn when to stop fighting, retreat,
recover, and return to productive play? This project investigates that question
in Honor of Kings 1v1 using a PPO policy built on Tencent AI Lab's
[Honor of Kings AI Open Environment (`hok_env`)](https://github.com/tencent-ailab/hok_env).

- **My contribution:** reward shaping for combat, economy, and objectives;
  a staged survival curriculum; and training diagnostics and checkpoint evaluation.
- **Reported result:** V7.22 at step 1,100,000 won **63 of 100 games (63.0%)**
  against a fixed Common AI opponent, with 50 games on each side.
- **Scope:** This checkpoint won 63 of 100 games against a fixed Common AI
  opponent, with 50 games played on each side. Further controlled comparisons
  are needed to determine how much the survival curriculum contributed to
  this performance.

[Evaluation results](#evaluation-results) ·
[Core code](#code-and-reproduction) ·
[Checkpoint release](https://github.com/FelixFeng28/hok-Reinforcement-Learning/releases/tag/v7.22-1100000)

## My Contribution

Tencent's `hok_env` provides the underlying environment and RL framework.
I build on that foundation rather than claiming the environment or PPO as
original contributions. My work focuses on:

| Area | Work in this project | Code entry |
|---|---|---|
| Reward design | Combine environment rewards with capped auxiliary signals for net trades, economic progress, objective progress, and retreat/recovery | [`Agent.get_shaped_reward`](source/v7_22_1100000/agent.py#L1791) |
| Survival curriculum | Define danger conditions and track ordered retreat, safety, and recovery milestones through reward shaping | [`Agent._calc_retreat_cycle_shaping_reward_details`](source/v7_22_1100000/agent.py#L674) |
| Diagnostics and evaluation | Record behavior and reward components; support explicit checkpoint evaluation against Common AI and selection of the model's side | [`actor.py`](source/v7_22_1100000/actor.py), [`entry.py`](source/v7_22_1100000/entry.py) |

These files include inherited framework code as well as project changes; the
table identifies the areas to inspect, not a claim of authorship of every line.

## Evaluation Results

### Fixed Common AI evaluation

| Checkpoint | Opponent | Games | Side allocation | Wins | Win rate |
|---|---|---:|---|---:|---:|
| V7.22 · step 1,100,000 | Fixed Common AI | 100 | 50 per side | 63 | **63.0%** |

This evaluation describes the checkpoint's performance against that opponent.
It does not establish improvement over a baseline, isolate the effect of the
survival curriculum, or measure generalization to other opponents. Repeated
evaluations are needed to assess variability.

### Recorded survival events

The same reported 100-game evaluation recorded the following events:

| Recorded event | Count |
|---|---:|
| Survival-decision cycles | 211 |
| Disengagement events | 204 |
| Confirmed safety events | 87 |
| Arrivals at the own crystal | 20 |
| Health / energy recovery events | 35 |
| Completed resupply events | 11 |

These are aggregate event counts, not counts of games won or independent
successful trajectories. They describe events detected by the instrumentation
along the intended sequence: **danger → disengagement → safety → return →
recovery**. The totals alone do not establish that the agent reliably recognizes
danger or completes the full sequence, and should not be treated as a funnel
without matching events to individual cycles.

The public release contains the checkpoint and a source snapshot, but does not
include the per-game evaluation logs behind these tables. The figures are
reported results; the release checksums verify artifact integrity, not the
evaluation outcome. No gameplay recording is currently available.

## Research Questions and Method

### 1. Can the agent learn to win a 1v1 game?

The initial objective was to train a PPO policy that could choose profitable
trades, take objectives, and win. The base environment exposes signals such as
health, experience, gold, kills, deaths, last hits, and objective health.
I extended the reward with small auxiliary signals for:

- **Net trade quality:** damage dealt to the opponent relative to damage received.
- **Economic progress:** newly earned gold relative to the opponent, computed
  from observable game state.
- **Objective progress:** enemy structure damage relative to damage to our own
  structures.

These terms help evaluate changes in advantage. Their per-step values are
clipped so that one unusually large difference does not dominate the PPO update
or encourage a single risky action.

### 2. Why did win rate stop improving?

Repeated fixed-opponent evaluations did not show stable improvement with
additional training and newer checkpoints. Trajectory inspection suggested
survival as a bottleneck: the agent could continue fighting when low on resources
and in a losing exchange, even when retreating would preserve its ability to
play the next phase.

This shifted the research question from only *"How can the agent win?"* to
*"When should the agent give up immediate combat rewards to survive?"*

### 3. A staged survival curriculum

The curriculum is designed around two connected stages:

**Stage 1 — danger recognition.** Define danger using observable conditions:
low health or energy, nearby enemy pressure, and recent losing trades. The
implementation supports an urgent low-health condition and an exception for a
clearly finishable opponent. During an active retreat cycle, positive auxiliary
combat and economy shaping can be suppressed so that immediate gains do not
override survival.

**Stage 2 — retreat and recovery.** Use ordered milestones for creating distance,
establishing safety, approaching the own crystal, recovering health or energy,
and completing resupply. Damage, lingering in danger, and death can receive
negative signals. The aim is to make the long-term value of recovery observable
through rewards.

PPO selects actions from those available in the environment. The curriculum
changes the learning signal rather than prescribing an action sequence. Returning
to the own crystal does not, by itself, demonstrate use of a recall action.

The source exposes configurable thresholds, weights, and switches. Their defaults
should not be assumed to equal the runtime settings used in the reported run;
the full run configuration is not included in the release.

## Code and Reproduction

### Browse the released source snapshot

The five files below are unchanged copies of `source/` from the
[V7.22 · 1.10M release bundle](https://github.com/FelixFeng28/hok-Reinforcement-Learning/releases/tag/v7.22-1100000).
They are provided for direct inspection and depend on the surrounding `hok_env`
framework; this directory is not a standalone training package.

| File | What to inspect |
|---|---|
| [`agent.py`](source/v7_22_1100000/agent.py) | Reward assembly, survival-cycle logic, and reward-component statistics |
| [`config.py`](source/v7_22_1100000/config.py) | Shaping weights, thresholds, and environment-variable overrides |
| [`actor.py`](source/v7_22_1100000/actor.py) | Episode execution, behavior monitoring, and checkpoint result logging |
| [`entry.py`](source/v7_22_1100000/entry.py) | Actor setup, checkpoint loading, and evaluation-side selection |
| [`train.py`](source/v7_22_1100000/train.py) | Learner entry point and integration with the RL framework |

See the [snapshot notes](source/v7_22_1100000/README.md) for provenance and
reproduction boundaries.

### Download and verify the checkpoint

- [Release and download options](https://github.com/FelixFeng28/hok-Reinforcement-Learning/releases/tag/v7.22-1100000)
- [Checkpoint and source bundle](https://github.com/FelixFeng28/hok-Reinforcement-Learning/releases/download/v7.22-1100000/v7_22_1100000_repro_bundle.tar.gz)
- [Bundle SHA256 checksum](https://github.com/FelixFeng28/hok-Reinforcement-Learning/releases/download/v7.22-1100000/v7_22_1100000_repro_bundle.tar.gz.sha256)

The bundle includes `MANIFEST.txt`, `SHA256SUMS`, the five source files, and
`model/checkpoints_20260828_051101.310174_1100000.tar`. The model archive contains
`checkpoint/model.pth`. Verify the downloaded bundle against its checksum, then
verify the model archive against `SHA256SUMS` inside the extracted bundle.

Running game evaluation requires an independently configured, licensed HoK
1v1/GameCore environment and the compatible framework dependencies. GameCore and
game assets are excluded. The snapshot and model provide inspection materials;
the full environment configuration and evaluation logs are still needed for an
exact reproduction of the reported experiment.

## Limitations and Next Experiments

- **Measure the curriculum's contribution:** compare a baseline and curriculum
  variants under matched opponent, side allocation, and training budget;
  repeat runs to assess variability.
- **Validate behavior at the trajectory level:** publish event definitions and
  per-cycle traces to distinguish partial retreats from successful recovery,
  and examine failure cases as well as successes.
- **Broaden evaluation:** test additional opponents and report results by side
  to examine robustness and side-dependent performance.
- **Complete the experiment record:** add per-game logs, exact runtime settings,
  framework/GameCore version information, and annotated gameplay examples when
  available.

The main lesson from this project is to pair outcome metrics with behavior-level
diagnostics: a win rate helps track performance, while trajectories and reward
components help identify which decisions to investigate next.
