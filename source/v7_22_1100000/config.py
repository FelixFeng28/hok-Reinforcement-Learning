import os


def _positive_float_env(name, default):
    """Read a positive floating-point training override from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("{} must be a positive float".format(name)) from exc
    if value <= 0:
        raise ValueError("{} must be a positive float".format(name))
    return value


class DimConfig:
    # main camp soldier
    DIM_OF_SOLDIER_1_10 = [18, 18, 18, 18]
    # enemy camp soldier
    DIM_OF_SOLDIER_11_20 = [18, 18, 18, 18]
    # main camp organ
    DIM_OF_ORGAN_1_2 = [18, 18]
    # enemy camp organ
    DIM_OF_ORGAN_3_4 = [18, 18]
    # main camp hero
    DIM_OF_HERO_FRD = [235]
    # enemy camp hero
    DIM_OF_HERO_EMY = [235]
    # public hero info
    DIM_OF_HERO_MAIN = [14]  # main_hero_vec

    DIM_OF_GLOBAL_INFO = [25]


class Config:
    backend = os.getenv("AIARENA_BACKEND", "pytorch")
    actor_num = int(os.getenv("ACTOR_NUM", "1"))
    auto_bind_cpu = os.getenv("AUTO_BIND_CPU", "0") == "1"

    # TODO refactor: learner only config
    use_init_model = os.getenv("AIARENA_USE_INIT_MODEL", "0") == "1"
    init_model_path = os.getenv(
        "AIARENA_INIT_MODEL_PATH", "/aiarena/code/learner/model/init/"
    )
    load_optimizer_state = os.getenv("AIARENA_LOAD_OPTIMIZER_STATE", "1") == "1"
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        809,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        8,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        512,
        512,
    ]
    SERI_VEC_SPLIT_SHAPE = [(725,), (84,)]
    # Keep the historical defaults, but make short stabilization experiments
    # reproducible without editing the checked-in reward/model configuration.
    INIT_LEARNING_RATE_START = _positive_float_env(
        "AIARENA_LEARNING_RATE", 0.00008
    )
    BETA_START = 0.03
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 8]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]  # means each task whether need reinforce

    RMSPROP_DECAY = 0.9
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.01
    CLIP_PARAM = _positive_float_env("AIARENA_PPO_CLIP_PARAM", 0.15)

    # Position shaping is used as a potential difference, never as a reward for
    # merely standing in one place. Healthy heroes should leave home and move
    # toward the lane/objectives, while low-HP heroes should retreat.
    HOME_DISTANCE_REWARD_ENABLE = os.getenv("HOME_DISTANCE_REWARD_ENABLE", "1") == "1"
    HOME_DISTANCE_SAFE_RADIUS = float(os.getenv("HOME_DISTANCE_SAFE_RADIUS", "10000"))
    HOME_DISTANCE_HOME_PENALTY = float(os.getenv("HOME_DISTANCE_HOME_PENALTY", "-0.01"))
    OBJECTIVE_DISTANCE_MAX_RADIUS = float(
        os.getenv("OBJECTIVE_DISTANCE_MAX_RADIUS", "60000")
    )
    OBJECTIVE_DISTANCE_REWARD_WEIGHT = float(
        os.getenv("OBJECTIVE_DISTANCE_REWARD_WEIGHT", "0.008")
    )
    OBJECTIVE_DISTANCE_HEALTHY_HP_RATE = float(
        os.getenv("OBJECTIVE_DISTANCE_HEALTHY_HP_RATE", "0.55")
    )
    OWN_OBJECTIVE_CAMP_RADIUS = float(os.getenv("OWN_OBJECTIVE_CAMP_RADIUS", "12000"))
    OWN_OBJECTIVE_DEFENSE_RADIUS = float(
        os.getenv("OWN_OBJECTIVE_DEFENSE_RADIUS", "14000")
    )
    OWN_OBJECTIVE_CAMP_PENALTY = float(
        os.getenv("OWN_OBJECTIVE_CAMP_PENALTY", "0.0")
    )

    # The environment's raw reward already includes money, HP, kills, deaths,
    # last hits, and objective HP. Keep its defaults authoritative and add only
    # one dense combat signal: signed net hero trade.
    SHAPING_NET_TRADE_WEIGHT = float(
        os.getenv("SHAPING_NET_TRADE_WEIGHT", "0.00001")
    )
    SHAPING_NET_TRADE_DAMAGE_RATIO = float(
        os.getenv("SHAPING_NET_TRADE_DAMAGE_RATIO", "1.0")
    )
    SHAPING_DELTA_REWARD_CLIP = float(os.getenv("SHAPING_DELTA_REWARD_CLIP", "0.04"))

    # GameCore's raw ``reward_money`` slot is zero in the deployed 1v1 SDK.
    # Use the observed hero money counters instead: reward only the change in
    # newly-earned gold relative to the opponent, not total wealth.  This is
    # deliberately small compared with the environment's terminal/objective
    # reward and is clipped per decision frame.
    SHAPING_GOLD_LEAD_WEIGHT = float(
        os.getenv("SHAPING_GOLD_LEAD_WEIGHT", "0.0002")
    )
    SHAPING_GOLD_LEAD_CLIP = float(
        os.getenv("SHAPING_GOLD_LEAD_CLIP", "0.02")
    )

    # Give a small dense signal for net structure progress.  The environment
    # retains its own terminal/objective reward; this only shortens the credit
    # assignment delay by comparing enemy objective HP loss with our own.
    SHAPING_OBJECTIVE_PROGRESS_WEIGHT = float(
        os.getenv("SHAPING_OBJECTIVE_PROGRESS_WEIGHT", "0.00001")
    )
    SHAPING_OBJECTIVE_PROGRESS_CLIP = float(
        os.getenv("SHAPING_OBJECTIVE_PROGRESS_CLIP", "0.02")
    )
    RETREAT_HP_RATE = float(os.getenv("RETREAT_HP_RATE", "0.42"))
    RETREAT_EP_RATE = float(os.getenv("RETREAT_EP_RATE", "0.25"))
    RETREAT_DISTANCE_MAX_RADIUS = float(
        os.getenv("RETREAT_DISTANCE_MAX_RADIUS", "30000")
    )
    RETREAT_HOME_REWARD_WEIGHT = float(
        os.getenv("RETREAT_HOME_REWARD_WEIGHT", "0.01")
    )
    RETREAT_LOW_RESOURCE_FAR_HOME_PENALTY_WEIGHT = float(
        os.getenv("RETREAT_LOW_RESOURCE_FAR_HOME_PENALTY_WEIGHT", "-0.006")
    )
    RETREAT_DANGER_MAX_RADIUS = float(
        os.getenv("RETREAT_DANGER_MAX_RADIUS", "12000")
    )
    RETREAT_DANGER_PENALTY_WEIGHT = float(
        os.getenv("RETREAT_DANGER_PENALTY_WEIGHT", "-0.03")
    )
    # Optional event-level retreat shaping.  Keep both defaults neutral so
    # established experiments are unchanged unless their launch explicitly
    # opts in.  The recovery reward is paid once after a low-resource hero
    # reaches home and restores a meaningful amount of HP or EP relative to
    # the resource level at which the retreat began.  The target rates prevent
    # a tiny regeneration tick at home from being treated as a completed
    # resupply.  The danger penalty applies only while the hero remains
    # low-resource, far from home, and close to an enemy hero or enemy
    # objective.
    RETREAT_RECOVERY_EVENT_REWARD = float(
        os.getenv("RETREAT_RECOVERY_EVENT_REWARD", "0.0")
    )
    RETREAT_RECOVERY_EVENT_MIN_HP_GAIN = float(
        os.getenv("RETREAT_RECOVERY_EVENT_MIN_HP_GAIN", "0.02")
    )
    RETREAT_RECOVERY_EVENT_MIN_EP_GAIN = float(
        os.getenv("RETREAT_RECOVERY_EVENT_MIN_EP_GAIN", "0.03")
    )
    RETREAT_RECOVERY_EVENT_TARGET_HP_RATE = float(
        os.getenv("RETREAT_RECOVERY_EVENT_TARGET_HP_RATE", "0.60")
    )
    RETREAT_RECOVERY_EVENT_TARGET_EP_RATE = float(
        os.getenv("RETREAT_RECOVERY_EVENT_TARGET_EP_RATE", "0.50")
    )
    RETREAT_RECOVERY_EVENT_MAX_FRAMES = int(
        os.getenv("RETREAT_RECOVERY_EVENT_MAX_FRAMES", "600")
    )
    RETREAT_LOW_RESOURCE_DANGER_STEP_PENALTY = float(
        os.getenv("RETREAT_LOW_RESOURCE_DANGER_STEP_PENALTY", "0.0")
    )
    # Reward concrete retreat progress before a hero reaches home. A reward is
    # emitted only when a low-resource, far-from-home hero crosses one full
    # distance step toward the own crystal, so idling and small back-and-forth
    # movement cannot farm it. Regression is likewise event based and only
    # applies while both adjacent states are dangerous.
    RETREAT_PROGRESS_STEP_DISTANCE = float(
        os.getenv("RETREAT_PROGRESS_STEP_DISTANCE", "6000")
    )
    RETREAT_PROGRESS_STEP_REWARD = float(
        os.getenv("RETREAT_PROGRESS_STEP_REWARD", "0.0")
    )
    RETREAT_PROGRESS_MAX_REWARD = float(
        os.getenv("RETREAT_PROGRESS_MAX_REWARD", "0.04")
    )
    RETREAT_DANGER_REGRESSION_PENALTY = float(
        os.getenv("RETREAT_DANGER_REGRESSION_PENALTY", "0.0")
    )
    # Optional recall-cycle shaping.  This is deliberately independent from
    # the older retreat/recovery signals so experiments can enable it without
    # changing established runs.  A cycle starts only for a critical,
    # threatened hero away from home and rewards the full sequence: retreat,
    # walk back to the own crystal, recover the resources that triggered the
    # retreat, then rejoin the map.  All reward defaults are neutral.
    RETREAT_CYCLE_CRITICAL_HP_RATE = float(
        os.getenv("RETREAT_CYCLE_CRITICAL_HP_RATE", "0.30")
    )
    RETREAT_CYCLE_CRITICAL_EP_RATE = float(
        os.getenv("RETREAT_CYCLE_CRITICAL_EP_RATE", "0.15")
    )
    RETREAT_CYCLE_REQUIRE_DANGER = os.getenv(
        "RETREAT_CYCLE_REQUIRE_DANGER", "1"
    ).strip().lower() not in ("0", "false", "no")
    # Optional tactical gate: enter a retreat cycle only when the low-resource
    # hero is actually losing the recent hero-damage exchange.  Keep this off
    # by default so prior retreat experiments remain reproducible.
    RETREAT_CYCLE_REQUIRE_LOSING_TRADE = os.getenv(
        "RETREAT_CYCLE_REQUIRE_LOSING_TRADE", "0"
    ).strip().lower() in ("1", "true", "yes")
    RETREAT_CYCLE_HARD_HP_RATE = float(
        os.getenv("RETREAT_CYCLE_HARD_HP_RATE", "0.18")
    )
    RETREAT_CYCLE_TRADE_WINDOW_FRAMES = int(
        os.getenv("RETREAT_CYCLE_TRADE_WINDOW_FRAMES", "180")
    )
    RETREAT_CYCLE_MIN_TRADE_WINDOW_FRAMES = int(
        os.getenv("RETREAT_CYCLE_MIN_TRADE_WINDOW_FRAMES", "60")
    )
    RETREAT_CYCLE_LOSING_TRADE_RATIO = float(
        os.getenv("RETREAT_CYCLE_LOSING_TRADE_RATIO", "1.25")
    )
    RETREAT_CYCLE_LOSING_TRADE_MIN_HP_RATE = float(
        os.getenv("RETREAT_CYCLE_LOSING_TRADE_MIN_HP_RATE", "0.08")
    )
    RETREAT_CYCLE_FINISHABLE_HP_MARGIN = float(
        os.getenv("RETREAT_CYCLE_FINISHABLE_HP_MARGIN", "0.08")
    )
    # When a retreat is mandatory, suppress positive auxiliary trade/economy
    # shaping.  The environment's raw reward is left untouched.
    RETREAT_CYCLE_SUPPRESS_OPPORTUNITY_REWARD = os.getenv(
        "RETREAT_CYCLE_SUPPRESS_OPPORTUNITY_REWARD", "0"
    ).strip().lower() in ("1", "true", "yes")
    RETREAT_CYCLE_MAX_FRAMES = int(
        os.getenv("RETREAT_CYCLE_MAX_FRAMES", "1800")
    )
    # This must be a tight radius around the own crystal.  It is deliberately
    # separate from RESOURCE_RECOVERY_HOME_RADIUS, whose wider value is used
    # for descriptive monitoring rather than proof of an actual resupply.
    RETREAT_CYCLE_HOME_RADIUS = float(
        os.getenv("RETREAT_CYCLE_HOME_RADIUS", "3000")
    )
    # A reachable first milestone on the base side.  It is valid only after
    # the enemy has been kept beyond SAFE_ENEMY_DISTANCE for SAFE_HOLD_FRAMES.
    # A non-positive radius disables this extra milestone.
    RETREAT_CYCLE_SAFE_HOME_RADIUS = float(
        os.getenv("RETREAT_CYCLE_SAFE_HOME_RADIUS", "0")
    )
    RETREAT_CYCLE_SAFE_ENEMY_DISTANCE = float(
        os.getenv("RETREAT_CYCLE_SAFE_ENEMY_DISTANCE", "14000")
    )
    RETREAT_CYCLE_SAFE_HOLD_FRAMES = int(
        os.getenv("RETREAT_CYCLE_SAFE_HOLD_FRAMES", "60")
    )
    # A base-side location is not safe if the hero is still taking damage.
    # Disabled by default so legacy reward profiles retain their behavior.
    RETREAT_CYCLE_SAFE_REQUIRE_NO_DAMAGE = os.getenv(
        "RETREAT_CYCLE_SAFE_REQUIRE_NO_DAMAGE", "0"
    ).strip().lower() in ("1", "true", "yes")
    RETREAT_CYCLE_SAFE_ARRIVAL_REWARD = float(
        os.getenv("RETREAT_CYCLE_SAFE_ARRIVAL_REWARD", "0.0")
    )
    # Stage 1 can define safety relative to the enemy hero rather than to any
    # enemy structure.  A base-side location plus sufficient hero separation
    # is the tactical "out of combat" condition used by the escape curriculum.
    RETREAT_CYCLE_SAFE_USE_HERO_DISTANCE = os.getenv(
        "RETREAT_CYCLE_SAFE_USE_HERO_DISTANCE", "0"
    ).strip().lower() in ("1", "true", "yes")
    # Curriculum stage 1 may end the cycle once the hero has demonstrably
    # escaped.  Later stages keep this false and continue toward crystal-side
    # recovery.
    RETREAT_CYCLE_SAFE_COMPLETES_CYCLE = os.getenv(
        "RETREAT_CYCLE_SAFE_COMPLETES_CYCLE", "0"
    ).strip().lower() in ("1", "true", "yes")
    RETREAT_CYCLE_PROGRESS_STEP_DISTANCE = float(
        os.getenv("RETREAT_CYCLE_PROGRESS_STEP_DISTANCE", "6000")
    )
    RETREAT_CYCLE_PROGRESS_STEP_REWARD = float(
        os.getenv("RETREAT_CYCLE_PROGRESS_STEP_REWARD", "0.0")
    )
    RETREAT_CYCLE_PROGRESS_MAX_REWARD = float(
        os.getenv("RETREAT_CYCLE_PROGRESS_MAX_REWARD", "0.006")
    )
    # Dense Stage-1 escape shaping.  An event requires both greater distance
    # from the enemy hero and concrete movement toward the own crystal, so an
    # opponent walking away cannot by itself farm the reward.  The maximum is
    # a cumulative cap for one retreat cycle, not a cap applied every frame.
    # Defaults keep existing experiments unchanged.
    RETREAT_CYCLE_ESCAPE_STEP_DISTANCE = float(
        os.getenv("RETREAT_CYCLE_ESCAPE_STEP_DISTANCE", "600")
    )
    RETREAT_CYCLE_ESCAPE_MIN_HOME_PROGRESS = float(
        os.getenv("RETREAT_CYCLE_ESCAPE_MIN_HOME_PROGRESS", "0")
    )
    RETREAT_CYCLE_ESCAPE_STEP_REWARD = float(
        os.getenv("RETREAT_CYCLE_ESCAPE_STEP_REWARD", "0.0")
    )
    RETREAT_CYCLE_ESCAPE_MAX_REWARD = float(
        os.getenv("RETREAT_CYCLE_ESCAPE_MAX_REWARD", "0.0")
    )
    # A reward cap alone can still allow many tiny escape increments.  This
    # optional event cap makes a retreat curriculum explicitly milestone-like:
    # after the first few valid escape segments, the policy must reach safety
    # rather than continue collecting movement feedback.  Zero preserves the
    # legacy reward-only cap.
    RETREAT_CYCLE_ESCAPE_MAX_EVENTS = int(
        os.getenv("RETREAT_CYCLE_ESCAPE_MAX_EVENTS", "0")
    )
    # Stage-1.5 safe-zone approach shaping.  While a retreat is mandatory,
    # reward strictly monotonic movement toward the reachable base-side safe
    # zone.  This fills the otherwise sparse gap between leaving combat and
    # earning SAFE_ARRIVAL_REWARD.  The per-cycle cap prevents repeatedly
    # walking back and forth from farming reward.  Defaults stay neutral so
    # prior experiments preserve their reward function.
    RETREAT_CYCLE_SAFE_APPROACH_STEP_DISTANCE = float(
        os.getenv("RETREAT_CYCLE_SAFE_APPROACH_STEP_DISTANCE", "0")
    )
    RETREAT_CYCLE_SAFE_APPROACH_STEP_REWARD = float(
        os.getenv("RETREAT_CYCLE_SAFE_APPROACH_STEP_REWARD", "0.0")
    )
    RETREAT_CYCLE_SAFE_APPROACH_MAX_REWARD = float(
        os.getenv("RETREAT_CYCLE_SAFE_APPROACH_MAX_REWARD", "0.0")
    )
    RETREAT_CYCLE_SAFE_APPROACH_MAX_EVENTS = int(
        os.getenv("RETREAT_CYCLE_SAFE_APPROACH_MAX_EVENTS", "0")
    )
    # Stage-2 continuation shaping.  Once a retreat has reached a confirmed
    # safe area, reward further movement toward the own crystal.  This is
    # separate from escape shaping: the enemy can already be far away, but
    # the hero must still finish walking home to obtain resupply.  The maximum
    # is cumulative per retreat cycle; zero defaults preserve prior runs.
    RETREAT_CYCLE_HOME_APPROACH_STEP_DISTANCE = float(
        os.getenv("RETREAT_CYCLE_HOME_APPROACH_STEP_DISTANCE", "0")
    )
    RETREAT_CYCLE_HOME_APPROACH_STEP_REWARD = float(
        os.getenv("RETREAT_CYCLE_HOME_APPROACH_STEP_REWARD", "0.0")
    )
    RETREAT_CYCLE_HOME_APPROACH_MAX_REWARD = float(
        os.getenv("RETREAT_CYCLE_HOME_APPROACH_MAX_REWARD", "0.0")
    )
    # A reward budget alone does not say how many progress milestones are
    # intended.  This separate cap lets a curriculum cover the complete
    # safe-zone-to-crystal route with a finite number of forward-only events.
    # Zero preserves the legacy reward-budget-only behavior.
    RETREAT_CYCLE_HOME_APPROACH_MAX_EVENTS = int(
        os.getenv("RETREAT_CYCLE_HOME_APPROACH_MAX_EVENTS", "0")
    )
    RETREAT_CYCLE_DANGER_HOLD_FRAMES = int(
        os.getenv("RETREAT_CYCLE_DANGER_HOLD_FRAMES", "0")
    )
    RETREAT_CYCLE_DANGER_HOLD_PENALTY = float(
        os.getenv("RETREAT_CYCLE_DANGER_HOLD_PENALTY", "0.0")
    )
    RETREAT_CYCLE_DANGER_HOLD_MAX_EVENTS = int(
        os.getenv("RETREAT_CYCLE_DANGER_HOLD_MAX_EVENTS", "0")
    )
    RETREAT_CYCLE_HOME_ARRIVAL_REWARD = float(
        os.getenv("RETREAT_CYCLE_HOME_ARRIVAL_REWARD", "0.0")
    )
    # Stage-3 recovery shaping.  Once the hero has reached the crystal-side
    # home radius after confirmed safety, reward real HP/EP restoration in
    # capped increments.  This is intentionally disabled by default so a
    # passive resource tick elsewhere on the map can never alter older runs.
    RETREAT_CYCLE_HOME_RECOVERY_STEP_GAIN = float(
        os.getenv("RETREAT_CYCLE_HOME_RECOVERY_STEP_GAIN", "0.0")
    )
    RETREAT_CYCLE_HOME_RECOVERY_STEP_REWARD = float(
        os.getenv("RETREAT_CYCLE_HOME_RECOVERY_STEP_REWARD", "0.0")
    )
    RETREAT_CYCLE_HOME_RECOVERY_MAX_REWARD = float(
        os.getenv("RETREAT_CYCLE_HOME_RECOVERY_MAX_REWARD", "0.0")
    )
    # One-time crystal-side readiness milestone.  Existing profiles remain
    # unchanged because all three controls default to neutral values.
    RETREAT_CYCLE_HOME_READY_HOLD_FRAMES = int(
        os.getenv("RETREAT_CYCLE_HOME_READY_HOLD_FRAMES", "0")
    )
    RETREAT_CYCLE_HOME_READY_MIN_RECOVERY_GAIN = float(
        os.getenv("RETREAT_CYCLE_HOME_READY_MIN_RECOVERY_GAIN", "0.0")
    )
    RETREAT_CYCLE_HOME_READY_REWARD = float(
        os.getenv("RETREAT_CYCLE_HOME_READY_REWARD", "0.0")
    )
    # A small terminal bonus for completing the resupply while the hero is
    # still in a valid home-side safe state.  Unlike a readiness gate, this
    # never delays or blocks a completed recovery.  Defaults remain neutral.
    RETREAT_CYCLE_SAFE_COMPLETION_BONUS = float(
        os.getenv("RETREAT_CYCLE_SAFE_COMPLETION_BONUS", "0.0")
    )
    RETREAT_CYCLE_COMPLETION_REWARD = float(
        os.getenv("RETREAT_CYCLE_COMPLETION_REWARD", "0.0")
    )
    RETREAT_CYCLE_REJOIN_REWARD = float(
        os.getenv("RETREAT_CYCLE_REJOIN_REWARD", "0.0")
    )
    # A recovery cycle is useful once the resource that caused the retreat is
    # back to a playable level.  Requiring a near-full bar made completed
    # resupplies too rare: the policy could earn a first recovery tick at the
    # crystal but would usually re-enter danger before reaching 0.70 / 0.60.
    # These remain environment-overridable for curriculum stages.
    RETREAT_CYCLE_TARGET_HP_RATE = float(
        os.getenv("RETREAT_CYCLE_TARGET_HP_RATE", "0.55")
    )
    RETREAT_CYCLE_TARGET_EP_RATE = float(
        os.getenv("RETREAT_CYCLE_TARGET_EP_RATE", "0.45")
    )
    RETREAT_CYCLE_MIN_HP_GAIN = float(
        os.getenv("RETREAT_CYCLE_MIN_HP_GAIN", "0.15")
    )
    RETREAT_CYCLE_MIN_EP_GAIN = float(
        os.getenv("RETREAT_CYCLE_MIN_EP_GAIN", "0.15")
    )
    RETREAT_CYCLE_HOME_MIN_HP_GAIN = float(
        os.getenv("RETREAT_CYCLE_HOME_MIN_HP_GAIN", "0.05")
    )
    RETREAT_CYCLE_HOME_MIN_EP_GAIN = float(
        os.getenv("RETREAT_CYCLE_HOME_MIN_EP_GAIN", "0.05")
    )
    RETREAT_CYCLE_REJOIN_HOME_DISTANCE = float(
        os.getenv("RETREAT_CYCLE_REJOIN_HOME_DISTANCE", "20000")
    )
    RETREAT_CYCLE_DANGER_DAMAGE_PENALTY_WEIGHT = float(
        os.getenv("RETREAT_CYCLE_DANGER_DAMAGE_PENALTY_WEIGHT", "0.0")
    )
    RETREAT_CYCLE_DANGER_DAMAGE_PENALTY_MAX = float(
        os.getenv("RETREAT_CYCLE_DANGER_DAMAGE_PENALTY_MAX", "-0.02")
    )
    # When a mandatory retreat is still taking damage in enemy danger range,
    # do not pay positive distance-progress rewards in the same transition.
    # This keeps a risky forward walk from being rewarded merely because the
    # hero happened to move closer to home while being hit.
    RETREAT_CYCLE_BLOCK_PROGRESS_ON_DAMAGE = os.getenv(
        "RETREAT_CYCLE_BLOCK_PROGRESS_ON_DAMAGE", "0"
    ).strip().lower() in ("1", "true", "yes")
    # Crystal-side recovery can optionally require that the hero remains in
    # the configured safe zone.  The arrival reward already requires a
    # confirmed safety milestone; this flag also prevents continued recovery
    # and completion rewards if an enemy later closes back into threat range.
    RETREAT_CYCLE_HOME_RECOVERY_REQUIRE_SAFE = os.getenv(
        "RETREAT_CYCLE_HOME_RECOVERY_REQUIRE_SAFE", "0"
    ).strip().lower() in ("1", "true", "yes")
    # A tight radius around the own crystal is a sanctuary for recovery. When
    # enabled, use it instead of the broad enemy-distance safe zone. Defaults
    # preserve historical reward profiles.
    RETREAT_CYCLE_HOME_RECOVERY_USE_HOME_SANCTUARY = os.getenv(
        "RETREAT_CYCLE_HOME_RECOVERY_USE_HOME_SANCTUARY", "0"
    ).strip().lower() in ("1", "true", "yes")
    # Keep the later curriculum stages causally clean: moving toward home or
    # counting regeneration while HP is still dropping is not a successful
    # retreat.  This is intentionally separate from the old global progress
    # block, which also disabled early escape shaping and caused regression.
    RETREAT_CYCLE_HOME_PROGRESS_REQUIRE_NO_DAMAGE = os.getenv(
        "RETREAT_CYCLE_HOME_PROGRESS_REQUIRE_NO_DAMAGE", "0"
    ).strip().lower() in ("1", "true", "yes")
    RETREAT_CYCLE_HOME_RECOVERY_REQUIRE_NO_DAMAGE = os.getenv(
        "RETREAT_CYCLE_HOME_RECOVERY_REQUIRE_NO_DAMAGE", "0"
    ).strip().lower() in ("1", "true", "yes")
    RETREAT_CYCLE_STALL_FRAMES = int(
        os.getenv("RETREAT_CYCLE_STALL_FRAMES", "0")
    )
    RETREAT_CYCLE_STALL_MIN_PROGRESS_DISTANCE = float(
        os.getenv("RETREAT_CYCLE_STALL_MIN_PROGRESS_DISTANCE", "500")
    )
    RETREAT_CYCLE_STALL_PENALTY = float(
        os.getenv("RETREAT_CYCLE_STALL_PENALTY", "0.0")
    )
    RETREAT_CYCLE_STALL_MAX_EVENTS = int(
        os.getenv("RETREAT_CYCLE_STALL_MAX_EVENTS", "0")
    )
    RETREAT_CYCLE_TIMEOUT_PENALTY = float(
        os.getenv("RETREAT_CYCLE_TIMEOUT_PENALTY", "0.0")
    )
    RETREAT_CYCLE_DEATH_PENALTY = float(
        os.getenv("RETREAT_CYCLE_DEATH_PENALTY", "0.0")
    )
    RESOURCE_RECOVERY_HOME_RADIUS = float(
        os.getenv("RESOURCE_RECOVERY_HOME_RADIUS", "12000")
    )
    LANE_DISTANCE_MAX_RADIUS = float(os.getenv("LANE_DISTANCE_MAX_RADIUS", "20000"))
    LANE_DISTANCE_REWARD_WEIGHT = float(
        os.getenv("LANE_DISTANCE_REWARD_WEIGHT", "0.01")
    )
    SKILL_MONITOR_TARGET_RADIUS = float(
        os.getenv("SKILL_MONITOR_TARGET_RADIUS", "16000")
    )
    SKILL_MONITOR_FOLLOWUP_FRAMES = int(
        os.getenv("SKILL_MONITOR_FOLLOWUP_FRAMES", "3")
    )

    MIN_POLICY = 0.00001
    TASK_ID = 15428
    TASK_UUID = "a2dbb49f-8a67-4bd4-9dc5-69e78422e72e"

    TARGET_EMBED_DIM = 32

    data_keys = (
        "observation,reward,advantage,"
        "label0,label1,label2,label3,label4,label5,"
        "prob0,prob1,prob2,prob3,prob4,prob5,"
        "weight0,weight1,weight2,weight3,weight4,weight5,"
        "is_train, lstm_cell, lstm_hidden_state"
    )
    data_shapes = [
        [12944],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [128],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]
    key_types = (
        "tf.float32,tf.float32,tf.float32,"
        "tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,tf.int32,"
        "tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,"
        "tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,"
        "tf.float32,tf.float32,tf.float32,tf.float32"
    )

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]
    slow_time = float(os.getenv("SLOW_TIME", "0").strip())
    ENEMY_TYPE = "common_ai"
    if os.getenv("ENEMY_TYPE") is not None:
        enemy_type = int(os.getenv("ENEMY_TYPE"))
        if enemy_type == 0:
            ENEMY_TYPE = "random"
        elif enemy_type == 1:
            ENEMY_TYPE = "common_ai"
        elif enemy_type == 2:
            ENEMY_TYPE = "network"
    # During network self-play, keep a fixed fraction of games against the
    # scripted opponent so the policy does not forget the common-AI benchmark.
    COMMON_AI_MIX_RATIO = float(os.getenv("COMMON_AI_MIX_RATIO", "0"))
    if not 0.0 <= COMMON_AI_MIX_RATIO <= 1.0:
        raise ValueError("COMMON_AI_MIX_RATIO must be between 0 and 1")
    # Alternate the current/latest policy between the two camps independently
    # for Common-AI and self-play episodes.  This removes a fixed-side bias
    # from the training data while retaining the requested opponent mix.
    BALANCE_TRAINING_CAMPS = os.getenv(
        "BALANCE_TRAINING_CAMPS", "1"
    ).strip().lower() not in ("0", "false", "no")
    EVAL_FREQ = 5
    GAMMA = 0.997
    LAMDA = 0.95
    IS_TRAIN = True
