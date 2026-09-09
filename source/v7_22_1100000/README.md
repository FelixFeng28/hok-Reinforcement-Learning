# V7.22 · 1.10M source snapshot

These five Python files are copied without modification from `source/` in the
[public V7.22 release](https://github.com/FelixFeng28/hok-Reinforcement-Learning/releases/tag/v7.22-1100000).
They preserve the released snapshot for inspection, including inherited
[Tencent `hok_env`](https://github.com/tencent-ailab/hok_env) framework code and
project-specific changes. Inclusion does not imply original authorship of every
file or function; refer to the upstream project for its licensing terms.

## Provenance

- Bundle: `v7_22_1100000_repro_bundle.tar.gz`
- Bundle SHA256: `962c6042ed3a6d4f8985f0f20dd238eebbef1968757cb356f39ef9ffa72b0cde`
- Manifest run: `tactical_retreat_home_v7_22`
- Manifest reward profile: `tactical_retreat_home_v7_20`
- Checkpoint step: `1100000`

The run label and reward-profile label are distinct fields in the release
manifest. The manifest does not contain a complete export of the runtime
environment variables.

## Reading guide

- [`agent.py`](agent.py): start with `get_shaped_reward` for reward assembly and
  `_calc_retreat_cycle_shaping_reward_details` for the survival milestones.
- [`config.py`](config.py): inspect the configuration switches and defaults;
  runtime environment variables can override them.
- [`actor.py`](actor.py): inspect behavior statistics and `CHECKPOINT_EVAL_RESULT`
  logging.
- [`entry.py`](entry.py): inspect checkpoint loading and
  `checkpoint_model_slot` for evaluation-side selection.
- [`train.py`](train.py): inspect integration with the framework's learner.

## Reproduction boundaries

This is a partial source snapshot, not a standalone application. Imports depend
on the original framework layout, model code, and supporting configuration
files. The release excludes GameCore and game assets and does not include the
complete runtime configuration, per-game evaluation logs, or a gameplay video.
Code defaults alone cannot reconstruct the experiment settings.

Checksums establish that the downloaded artifacts match the release. They do
not independently verify the reported win rate or behavior counts.
