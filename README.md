# Honor of Kings 1v1 Reinforcement Learning

## Reward Design, Survival Curriculum, and Evidence-Based Evaluation

This project studies how a reinforcement-learning agent can learn the behavior pattern of making independt behavior in a 1v1 Honor of Kings environment. The goal is to train an
agent that can recognize when a trade has become dangerous, disengage from the
fight, return to base for recovery, and resume productive play afterwards.

The implementation is built on Tencent AI Lab's
[Honor of Kings AI Open Environment (`hok_env`)](https://github.com/tencent-ailab/hok_env).
My work focuses on reward shaping, curriculum design, training diagnostics, and
reproducible evaluation.

## Research Questions and Project Evolution

This project developed through two connected research questions.

### Part 1 — Can the agent learn to win a 1v1 game?

The initial objective was performance: train a PPO policy that can
choose profitable trades, take objectives, and win. I used reward shaping for trade quality, economic progress, and
objective damage so that the agent would learn productive combat.

### Part 2 — Why did win rate stop improving, and can the agent learn to survive?

Repeated fixed-opponent evaluations showed that additional training and newer
checkpoints did not produce a stable improvement in win rate. I therefore
shifted from asking only *"How can the agent win?"* to asking a more diagnostic
question: *"What decision is preventing the agent from converting combat skill
into reliable wins?"*

Trajectory inspection pointed to survival as a key bottleneck. The agent could
continue fighting when low on resources and in a losing exchange,
even when returning to base would preserve its ability to play the next phase
of the game. The later curriculum therefore teaches the following decision
sequence:

1. Recognize that the current situation is dangerous.
2. Compare the short-term gain of fighting or farming with the long-term value
   of survival.
3. Disengage, establish safety, return to its crystal, and recover when
   recovery has the higher expected return.

PPO remains responsible for selecting concrete movement, skill, and recall
actions. The curriculum changes the learning signal, not the action policy.

## Methodology

### 1. Learning productive combat

The base environment already includes game-level signals such as health,
experience, gold, kills, deaths, last hits, and objective health. I extended
the reward with small auxiliary signals for:

- **Net trade quality:** damage dealt to the opponent minus damage received.
- **Economic progress:** newly earned gold, computed from the observable game
  state.
- **Objective progress:** meaningful tower or objective damage.

Together, these terms help the model evaluate its advantage relative to its
opponent.

I also clip their per-step values so that one
unusually large difference does not dominate the PPO update or cause the model
to overvalue a single risky action.

### 2. Staged survival curriculum

The survival objective is taught in two connected stages.

**Stage 1 — danger recognition.** A dangerous state is defined from observable
conditions: low health or energy, nearby enemy pressure, and a recent losing
trade. Hard low-health states are treated as urgent, while a clearly
finishable opponent remains an exception. When the state is active, positive
auxiliary combat and economy shaping is suppressed so that immediate fighting
or trading does not override survival.

**Stage 2 — deciding whether to return for recovery.** Once danger is
recognized, the agent receives a sparse, ordered set of milestones: create
distance from the threat, remain safe, approach the crystal, recover
health or energy, and complete resupply. Continued damage, lingering in the
danger zone, and death receive negative signals. This makes the intended
long-term choice observable without hard-coding a recall action.

## Training and Monitoring

The project uses fixed-opponent evaluation, a fixed common-AI.



## Best Result

| Checkpoint | Fixed Common AI evaluation | Win rate |
|---|---:|---:|
| V7.22 · 1.10M | 63 / 100 games | **63.0%** |

The evaluation uses 100 games split evenly across the two sides.

## Evidence of Survival Behavior

In the same 100-game evaluation, the V7.22 · 1.10M policy produced the
following recorded survival trajectory events:

| Recorded behavior | Count |
|---|---:|
| Survival-decision cycles | 211 |
| Disengagement events | 204 |
| Confirmed safety events | 87 |
| Arrivals at the own crystal | 20 |
| Health / energy recovery events | 35 |
| Completed resupply events | 11 |

These data do not mean that every dangerous situation ends in a successful
return. They do show that the policy has learned components of the desired
behavioral chain: **danger recognition → disengagement → safety → return and
recovery**.

## What I Learned

- Reinforcement learning requires a behavior-level objective, not only a large
  reward value. A reward should explain *why* an action is useful in the
  current state.
- Reward shaping can create unintended shortcuts. Small, capped rewards and
  explicit monitoring are necessary to prevent the agent from optimizing a
  proxy instead of the intended behavior.
- Self-play can improve local adaptation while degrading performance against a
  fixed opponent. Fixed, side-balanced evaluation is essential for detecting
  this form of policy regression.


## Repository Materials

This repository includes the current trained checkpoint, **V7.22 · 1.10M**,
so that its behavior can be independently evaluated against the fixed Common
AI setup reported above.

## Verified checkpoint

### V7.22 · step 1,100,000

The reproducibility bundle contains the archived checkpoint, manifest, SHA256
checksums, and the corresponding source snapshot.

- [Download and verify the checkpoint bundle](https://github.com/FelixFeng28/hok-Reinforcement-Learning/releases/tag/v7.22-1100000)
- Artifact: `checkpoint/model.pth`
- Validation: SHA256 + manifest + archived source snapshot
- Note: reproducing game evaluation requires an independently configured,
  licensed HoK 1v1/GameCore environment.
