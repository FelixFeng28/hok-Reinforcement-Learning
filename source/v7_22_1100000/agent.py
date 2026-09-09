import random
import time
import h5py
import numpy as np
import os

from rl_framework.predictor.utils import (
    cvt_tensor_to_infer_input,
    cvt_tensor_to_infer_output,
)
from rl_framework.model_pool import ModelPoolAPIs

from rl_framework.common.logging import log_time
from rl_framework.common.logging import logger as LOG
from hok.hok1v1.agent import AgentBase
from hok.hok1v1.lib.interface import ACTOR_CRYSTAL, ACTOR_TOWER, ACTOR_TOWER_HIGH


_G_CHECK_POINT_PREFIX = "checkpoints_"
_G_RAND_MAX = 10000
_G_MODEL_UPDATE_RATIO = 0.8
_ENV_REWARD_COMPONENTS = (
    "dead",
    "ep_rate",
    "exp",
    "hp_point",
    "kill",
    "last_hit",
    "money",
    "objective_hp",
)


def cvt_infer_list_to_numpy_list(infer_list):
    data_list = [infer.data for infer in infer_list]
    return data_list


class RandomAgent:
    def process(self, feature, legal_action):
        action = [random.randint(0, 2) - 1, random.randint(0, 2) - 1]
        value = [0.0]
        neg_log_pi = [0]
        return action, value, neg_log_pi


class Agent(AgentBase):
    def __init__(
        self,
        model,
        model_pool_addr,
        config,
        keep_latest=False,
        dataset=None,
        single_test=False,
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.single_test = single_test

        if self.config.backend == "pytorch":
            from rl_framework.predictor.predictor.local_torch_predictor import (
                LocalTorchPredictor as LocalPredictor,
            )

            self._predictor = LocalPredictor(self.model)
        elif self.config.backend == "tensorflow":
            from rl_framework.predictor.predictor.local_predictor import (
                LocalCkptPredictor as LocalPredictor,
            )

            self.graph = self.model.build_infer_graph()
            self._predictor = LocalPredictor(self.graph)
        else:
            raise NotImplementedError(
                "SINGLE_TEST: backend not in [tensorflow, pytorch]..."
            )

        if single_test or not model_pool_addr:
            self._model_pool_api = None
        else:
            self._model_pool_api = ModelPoolAPIs(model_pool_addr)

        self.model_version = ""
        self.is_latest_model: bool = False
        self.keep_latest = keep_latest
        self.model_list = []

        self.lstm_unit_size = self.config.LSTM_UNIT_SIZE

        self.lstm_hidden = None
        self.lstm_cell = None

        # self.agent_type = "common_ai"
        self.player_id = 0
        self.hero_camp = 0
        self.last_model_path = None
        self.label_size_list = self.config.LABEL_SIZE_LIST
        self.legal_action_size = self.config.LEGAL_ACTION_SIZE_LIST
        self._last_shaping_stats = None
        self._last_economy_stats = None
        self._last_objective_stats = None
        self._last_position_potential = None
        self._last_position_context = {"low_resource": False}
        self._retreat_recovery_pending_frame = None
        self._retreat_recovery_origin_stats = None
        self._retreat_cycle = None
        self._retreat_trade_history = []
        self._retreat_decision_context = {"must_retreat": False}
        self._retreat_progress_best_home_distance = None
        self._retreat_progress_last_home_distance = None
        self._retreat_progress_last_danger = False
        self._reset_reward_component_stats()

        # self.agent_type = "network"
        if self.keep_latest:
            self.agent_type = "network"
        else:
            self.agent_type = self.config.ENEMY_TYPE

        if dataset is None:
            self.save_h5_sample = False
            self.dataset_name = None
            self.dataset = None
        else:
            self.save_h5_sample = True
            self.dataset_name = dataset
            self.dataset = h5py.File(dataset, "a")

    def set_game_info(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.player_id = player_id

    # reset the agent,agent_type in ["network","common_ai"],if model_path is None,get model from model pool
    def reset(self, agent_type=None, model_path=None):
        # reset lstm input
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self._last_shaping_stats = None
        self._last_economy_stats = None
        self._last_objective_stats = None
        self._last_position_potential = None
        self._last_position_context = {"low_resource": False}
        self._retreat_recovery_pending_frame = None
        self._retreat_recovery_origin_stats = None
        self._retreat_cycle = None
        self._retreat_trade_history = []
        self._retreat_decision_context = {"must_retreat": False}
        self._retreat_progress_best_home_distance = None
        self._retreat_progress_last_home_distance = None
        self._retreat_progress_last_danger = False
        self._reset_reward_component_stats()

        if agent_type is not None:
            if self.keep_latest:
                self.agent_type = "network"
            else:
                self.agent_type = agent_type

        # for test without model pool
        if self.single_test:
            self.is_latest_model = True
            if self.config.backend == "tensorflow":
                LOG.info("SINGLE_TEST: backend=tensorflow")
                self._predictor._sess.run(self.model.init)
            elif self.config.backend == "pytorch":
                LOG.info("SINGLE_TEST: backend=pytorch")
            else:
                raise NotImplementedError(
                    "SINGLE_TEST: backend not in [tensorflow, pytorch]..."
                )
        else:
            if model_path is None:
                while True:
                    try:
                        if self.keep_latest:
                            self._get_latest_model()
                        else:
                            self._get_random_model()
                        self.last_model_path = None
                        return
                    except Exception as e:  # pylint: disable=broad-except
                        LOG.error(e)
                        LOG.error("get_model error, try again...")
                        time.sleep(1)
            else:
                if model_path != self.last_model_path:
                    self._predictor.load_model(model_path)
                    self.last_model_path = model_path
                else:
                    LOG.info(
                        "model {} alreadly load last time, skip now!".format(model_path)
                    )

        if self.dataset is None:
            self.save_h5_sample = False
        else:
            self.save_h5_sample = True
            self.dataset.close()
            self.dataset = h5py.File(self.dataset_name, "a")

    def _update_model_list(self):
        model_key_list = []
        while len(model_key_list) == 0:
            model_key_list = self._model_pool_api.pull_keys()
            if not model_key_list:
                LOG.warning("No model in model_pool, wait for 1 sec...")
                time.sleep(1)
        self.model_list = model_key_list

    def _load_model(self, model_version):
        if model_version == self.model_version:
            return True
        model_path = self._model_pool_api.pull_model_path(model_version)
        model_path = "%s/checkpoint" % (model_path)
        LOG.info("load model: {} in {}".format(model_version, model_path))
        ret = self._predictor.load_model(model_path)
        if ret:
            # if failed, do not update model_version
            self.model_version = model_version
        return ret

    # randomly get a model from model pool with 80% probability for newest model and 20% probability for history models
    def _get_random_model(self):
        if self.agent_type in ["common_ai", "random"]:
            self.is_latest_model = False
            if self.config.backend == "tensorflow":
                self._predictor._sess.run(self.model.init)
                LOG.info("_get_random_model: backend=tensorflow")
            elif self.config.backend == "pytorch":
                LOG.info("_get_random_model: backend=pytorch")
            else:
                raise NotImplementedError(
                    "_get_random_model: backend not in [tensorflow, pytorch]..."
                )

            self.model_version = ""
            return True

        self._update_model_list()
        rand_float = float(random.uniform(0, _G_RAND_MAX)) / float(_G_RAND_MAX)
        if rand_float <= _G_MODEL_UPDATE_RATIO:
            midx = len(self.model_list) - 1
            self.is_latest_model = True
        else:
            midx = int(random.random() * len(self.model_list))
            if midx == len(self.model_list):
                midx = len(self.model_list) - 1
            self.is_latest_model = False
        return self._load_model(self.model_list[midx])

    def _get_latest_model(self):
        self._update_model_list()
        self.is_latest_model = True
        return self._load_model(self.model_list[-1])

    def feature_post_process(self, state_dict):
        return state_dict

    # handle the obs from gamecore, return action result
    @log_time("aiprocess_process")
    def process(self, state_dict, battle=False):
        # call custom feature process (python)
        state_dict = self.feature_post_process(state_dict)

        feature_vec, legal_action = (
            state_dict["observation"],
            state_dict["legal_action"],
        )

        if self.config.backend == "pytorch":
            pred_ret = self._predict_process_torch(feature_vec, legal_action)
        elif self.config.backend == "tensorflow":
            pred_ret = self._predict_process(feature_vec, legal_action)
        else:
            raise NotImplementedError(
                "SINGLE_TEST: backend not in [tensorflow, pytorch]..."
            )
        _, _, action, d_action = pred_ret
        if battle:
            return d_action
        return action, d_action, self._sample_process(state_dict, pred_ret)

    def _update_legal_action(self, original_la, actions):
        target_size = self.config.LABEL_SIZE_LIST[-1]
        top_size = self.config.LABEL_SIZE_LIST[0]
        original_la = np.array(original_la)
        fix_part = original_la[: -target_size * top_size]
        target_la = original_la[-target_size * top_size :]
        target_la = target_la.reshape([top_size, target_size])[actions[0]]
        return np.concatenate([fix_part, target_la], axis=0)

    @staticmethod
    def _distance(loc_a, loc_b):
        return np.sqrt((loc_a.x - loc_b.x) ** 2 + (loc_a.z - loc_b.z) ** 2)

    @staticmethod
    def _get_first_attr(obj, names, default=None):
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    def _get_ep_rate(self, hero_state):
        ep = self._get_first_attr(hero_state, ("ep", "ep_point", "energy", "mp"), None)
        max_ep = self._get_first_attr(
            hero_state, ("max_ep", "maxEp", "max_ep_point", "maxEnergy", "max_mp"), None
        )
        if ep is None or max_ep is None or max_ep <= 0:
            return None
        return ep / max(max_ep, 1)

    def _calc_position_shaping_reward(self, req_pb):
        """Return the change in positional potential since the last state.

        A state reward would pay the policy indefinitely for standing near a
        lane or tower. Using a difference instead pays only for moving toward
        the desired location; moving away gives the matching negative signal.
        """
        if not self.config.HOME_DISTANCE_REWARD_ENABLE:
            return 0.0

        hero_state = None
        enemy_hero_states = []
        for hero in req_pb.hero_list:
            if hero.camp == self.hero_camp:
                hero_state = hero
            elif hero.hp > 0:
                enemy_hero_states.append(hero)

        if hero_state is None or hero_state.hp <= 0:
            return 0.0

        max_hp = max(hero_state.max_hp, 1)
        hp_rate = hero_state.hp / max_hp
        ep_rate = self._get_ep_rate(hero_state)
        low_hp = hp_rate < self.config.RETREAT_HP_RATE
        low_ep = ep_rate is not None and ep_rate < self.config.RETREAT_EP_RATE
        low_resource = low_hp or low_ep
        self._last_position_context = {"low_resource": low_resource}

        own_crystal_state = None
        own_objective_states = []
        enemy_objective_states = []
        for organ in req_pb.organ_list:
            if organ.camp == self.hero_camp and organ.type in (
                ACTOR_TOWER,
                ACTOR_TOWER_HIGH,
                ACTOR_CRYSTAL,
            ):
                if organ.hp > 0:
                    own_objective_states.append(organ)
                if organ.type == ACTOR_CRYSTAL:
                    own_crystal_state = organ
            elif organ.camp != self.hero_camp and organ.type in (
                ACTOR_TOWER,
                ACTOR_TOWER_HIGH,
                ACTOR_CRYSTAL,
            ) and organ.hp > 0:
                enemy_objective_states.append(organ)

        enemy_soldier_states = [
            soldier
            for soldier in req_pb.soldier_list
            if soldier.camp != self.hero_camp and soldier.hp > 0
        ]

        potential = 0.0

        hero_loc = hero_state.location
        home_distance = None
        if own_crystal_state is not None:
            crystal_loc = own_crystal_state.location
            home_distance = self._distance(hero_loc, crystal_loc)

            safe_radius = self.config.HOME_DISTANCE_SAFE_RADIUS
            if not low_resource and home_distance < safe_radius:
                stay_home_ratio = 1.0 - home_distance / safe_radius
                potential += self.config.HOME_DISTANCE_HOME_PENALTY * stay_home_ratio

        if low_resource:
            if home_distance is not None:
                self._last_position_context["low_resource_far_home"] = (
                    home_distance >= self.config.RESOURCE_RECOVERY_HOME_RADIUS
                )
                retreat_radius = max(self.config.RETREAT_DISTANCE_MAX_RADIUS, 1.0)
                retreat_ratio = 1.0 - min(home_distance / retreat_radius, 1.0)
                potential += self.config.RETREAT_HOME_REWARD_WEIGHT * retreat_ratio
                far_home_ratio = min(home_distance / retreat_radius, 1.0)
                potential += (
                    self.config.RETREAT_LOW_RESOURCE_FAR_HOME_PENALTY_WEIGHT
                    * far_home_ratio
                )

            danger_distances = [
                self._distance(hero_loc, enemy_hero.location)
                for enemy_hero in enemy_hero_states
            ]
            danger_distances.extend(
                self._distance(hero_loc, organ.location)
                for organ in enemy_objective_states
            )
            if danger_distances:
                danger_radius = max(self.config.RETREAT_DANGER_MAX_RADIUS, 1.0)
                danger_ratio = 1.0 - min(min(danger_distances) / danger_radius, 1.0)
                potential += self.config.RETREAT_DANGER_PENALTY_WEIGHT * danger_ratio

            return self._position_potential_delta(potential)

        must_retreat = self._retreat_decision_context.get("must_retreat", False)
        if (
            hp_rate < self.config.OBJECTIVE_DISTANCE_HEALTHY_HP_RATE
            or (
                must_retreat
                and self.config.RETREAT_CYCLE_SUPPRESS_OPPORTUNITY_REWARD
            )
        ):
            return self._position_potential_delta(potential)

        if own_objective_states:
            own_objective_distances = [
                self._distance(hero_loc, organ.location) for organ in own_objective_states
            ]
            nearest_own_objective_distance = min(own_objective_distances)
            defense_radius = max(self.config.OWN_OBJECTIVE_DEFENSE_RADIUS, 1.0)
            objective_under_pressure = False
            pressure_units = enemy_soldier_states + enemy_hero_states
            for unit in pressure_units:
                for organ in own_objective_states:
                    if self._distance(unit.location, organ.location) < defense_radius:
                        objective_under_pressure = True
                        break
                if objective_under_pressure:
                    break

            camp_radius = max(self.config.OWN_OBJECTIVE_CAMP_RADIUS, 1.0)
            if (
                not objective_under_pressure
                and nearest_own_objective_distance < camp_radius
            ):
                camp_ratio = 1.0 - nearest_own_objective_distance / camp_radius
                potential += self.config.OWN_OBJECTIVE_CAMP_PENALTY * camp_ratio

        if enemy_soldier_states:
            lane_distances = [
                self._distance(hero_loc, soldier.location)
                for soldier in enemy_soldier_states
            ]
            lane_radius = max(self.config.LANE_DISTANCE_MAX_RADIUS, 1.0)
            lane_ratio = 1.0 - min(min(lane_distances) / lane_radius, 1.0)
            potential += self.config.LANE_DISTANCE_REWARD_WEIGHT * lane_ratio

        if not enemy_objective_states:
            return self._position_potential_delta(potential)

        enemy_distances = []
        for organ in enemy_objective_states:
            enemy_distances.append(self._distance(hero_loc, organ.location))

        nearest_enemy_distance = min(enemy_distances)
        max_radius = max(self.config.OBJECTIVE_DISTANCE_MAX_RADIUS, 1.0)
        objective_ratio = 1.0 - min(nearest_enemy_distance / max_radius, 1.0)
        potential += self.config.OBJECTIVE_DISTANCE_REWARD_WEIGHT * objective_ratio
        return self._position_potential_delta(potential)

    def _position_potential_delta(self, potential):
        last_potential = self._last_position_potential
        self._last_position_potential = potential
        if last_potential is None:
            return 0.0
        return potential - last_potential

    def _calc_retreat_recovery_shaping_reward_details(self, req_pb):
        """Reward concrete retreat progress and completed recovery.

        The older potential signal rewards a positional change but can be too
        small and diffuse to credit a retreat.  This optional event signal
        rewards a meaningful home-distance reduction while resources are low.
        It cannot be farmed by jittering because each distance interval pays
        at most once.  Moving a full interval away while danger persists is
        separately penalized.
        """
        hero_state = None
        enemy_states = []
        own_crystal_state = None
        for hero in req_pb.hero_list:
            if hero.camp == self.hero_camp:
                hero_state = hero
            elif hero.hp > 0:
                enemy_states.append(hero)

        for organ in req_pb.organ_list:
            if organ.camp == self.hero_camp and organ.type == ACTOR_CRYSTAL:
                own_crystal_state = organ
            elif (
                organ.camp != self.hero_camp
                and organ.type in (ACTOR_TOWER, ACTOR_TOWER_HIGH, ACTOR_CRYSTAL)
                and organ.hp > 0
            ):
                enemy_states.append(organ)

        if (
            hero_state is None
            or hero_state.hp <= 0
            or own_crystal_state is None
            or own_crystal_state.hp <= 0
        ):
            self._retreat_recovery_pending_frame = None
            self._retreat_recovery_origin_stats = None
            self._retreat_progress_best_home_distance = None
            self._retreat_progress_last_home_distance = None
            self._retreat_progress_last_danger = False
            return 0.0, 0.0, 0.0, 0.0, 0, 0.0, False, False, False

        hp_rate = hero_state.hp / max(hero_state.max_hp, 1)
        ep_rate = self._get_ep_rate(hero_state)
        low_resource = (
            hp_rate < self.config.RETREAT_HP_RATE
            or (
                ep_rate is not None
                and ep_rate < self.config.RETREAT_EP_RATE
            )
        )
        home_distance = self._distance(
            hero_state.location, own_crystal_state.location
        )
        near_home = home_distance < self.config.RESOURCE_RECOVERY_HOME_RADIUS
        far_home = not near_home
        frame_no = int(req_pb.frame_no)

        if (
            low_resource
            and far_home
            and self._retreat_recovery_pending_frame is None
        ):
            self._retreat_recovery_pending_frame = frame_no
            self._retreat_recovery_origin_stats = {
                "hp_rate": hp_rate,
                "ep_rate": ep_rate,
            }

        recovery_reward = 0.0
        recovery_event = False
        pending_frame = self._retreat_recovery_pending_frame
        origin_stats = self._retreat_recovery_origin_stats
        if (
            pending_frame is not None
            and frame_no - pending_frame
            <= self.config.RETREAT_RECOVERY_EVENT_MAX_FRAMES
            and near_home
            and origin_stats is not None
        ):
            hp_recovered = (
                hp_rate >= self.config.RETREAT_RECOVERY_EVENT_TARGET_HP_RATE
                and hp_rate - origin_stats["hp_rate"]
                >= self.config.RETREAT_RECOVERY_EVENT_MIN_HP_GAIN
            )
            ep_recovered = (
                ep_rate is not None
                and origin_stats["ep_rate"] is not None
                and ep_rate >= self.config.RETREAT_RECOVERY_EVENT_TARGET_EP_RATE
                and ep_rate - origin_stats["ep_rate"]
                >= self.config.RETREAT_RECOVERY_EVENT_MIN_EP_GAIN
            )
            if hp_recovered or ep_recovered:
                recovery_reward = self.config.RETREAT_RECOVERY_EVENT_REWARD
                recovery_event = recovery_reward != 0.0
                self._retreat_recovery_pending_frame = None
                self._retreat_recovery_origin_stats = None

        if (
            self._retreat_recovery_pending_frame is not None
            and frame_no - self._retreat_recovery_pending_frame
            > self.config.RETREAT_RECOVERY_EVENT_MAX_FRAMES
        ):
            self._retreat_recovery_pending_frame = None
            self._retreat_recovery_origin_stats = None

        danger = False
        if low_resource and far_home and enemy_states:
            danger = (
                min(
                    self._distance(hero_state.location, enemy.location)
                    for enemy in enemy_states
                )
                < self.config.RETREAT_DANGER_MAX_RADIUS
            )
        danger_penalty = (
            self.config.RETREAT_LOW_RESOURCE_DANGER_STEP_PENALTY
            if danger
            else 0.0
        )

        progress_reward = 0.0
        progress_events = 0
        regression_penalty = 0.0
        regression_event = False
        if low_resource and far_home:
            best_distance = self._retreat_progress_best_home_distance
            last_distance = self._retreat_progress_last_home_distance
            step_distance = max(self.config.RETREAT_PROGRESS_STEP_DISTANCE, 1.0)

            if best_distance is None:
                self._retreat_progress_best_home_distance = home_distance
            else:
                progressed_distance = best_distance - home_distance
                if progressed_distance >= step_distance:
                    progress_events = int(progressed_distance / step_distance)
                    raw_progress_reward = (
                        progress_events
                        * self.config.RETREAT_PROGRESS_STEP_REWARD
                    )
                    progress_reward = min(
                        raw_progress_reward,
                        max(self.config.RETREAT_PROGRESS_MAX_REWARD, 0.0),
                    )
                    self._retreat_progress_best_home_distance = home_distance

            if (
                last_distance is not None
                and self._retreat_progress_last_danger
                and danger
                and home_distance - last_distance >= step_distance
            ):
                regression_penalty = self.config.RETREAT_DANGER_REGRESSION_PENALTY
                regression_event = regression_penalty != 0.0

            self._retreat_progress_last_home_distance = home_distance
            self._retreat_progress_last_danger = danger
        else:
            self._retreat_progress_best_home_distance = None
            self._retreat_progress_last_home_distance = None
            self._retreat_progress_last_danger = False

        return (
            recovery_reward
            + danger_penalty
            + progress_reward
            + regression_penalty,
            recovery_reward,
            danger_penalty,
            progress_reward,
            progress_events,
            regression_penalty,
            regression_event,
            recovery_event,
            danger,
        )

    def _get_retreat_trade_window(self, frame_no, hero_state):
        """Return recent hero-versus-hero damage deltas for retreat gating."""
        history = self._retreat_trade_history
        if history and frame_no <= history[-1]["frame_no"]:
            history.clear()

        history.append(
            {
                "frame_no": frame_no,
                "dealt": float(hero_state.totalHurtToHero),
                "taken": float(hero_state.totalBeHurtByHero),
            }
        )
        window = max(self.config.RETREAT_CYCLE_TRADE_WINDOW_FRAMES, 1)
        while len(history) > 1 and frame_no - history[0]["frame_no"] > window:
            history.pop(0)

        origin = history[0]
        elapsed = frame_no - origin["frame_no"]
        dealt = max(float(hero_state.totalHurtToHero) - origin["dealt"], 0.0)
        taken = max(float(hero_state.totalBeHurtByHero) - origin["taken"], 0.0)
        return dealt, taken, elapsed

    def _calc_retreat_cycle_shaping_reward_details(self, req_pb):
        """Reward a tactically justified retreat, safety, and resupply cycle.

        With the optional losing-trade gate enabled, low resources alone do
        not force a retreat: a hero may finish an opponent when the recent
        exchange is favorable.  Once a retreat is mandatory, positive
        auxiliary combat/economy shaping can be suppressed and the cycle pays
        only for concrete milestones: escaping danger, reaching the crystal,
        and recovering there.
        """
        details = {
            "started": 0,
            "forced_retreat_samples": 0,
            "fight_allowed_samples": 0,
            "progress_shaping": 0.0,
            "progress_events": 0,
            "escape_shaping": 0.0,
            "escape_events": 0,
            "safe_shaping": 0.0,
            "safe_events": 0,
            "safe_approach_shaping": 0.0,
            "safe_approach_events": 0,
            "home_approach_shaping": 0.0,
            "home_approach_events": 0,
            "home_shaping": 0.0,
            "home_events": 0,
            "home_recovery_shaping": 0.0,
            "home_recovery_events": 0,
            "home_ready_shaping": 0.0,
            "home_ready_events": 0,
            "safe_completion_shaping": 0.0,
            "safe_completion_events": 0,
            "completion_shaping": 0.0,
            "completion_events": 0,
            "rejoin_shaping": 0.0,
            "rejoin_events": 0,
            "danger_damage_shaping": 0.0,
            "danger_damage_events": 0,
            "danger_reward_block_events": 0,
            "danger_hold_shaping": 0.0,
            "danger_hold_events": 0,
            "death_shaping": 0.0,
            "death_events": 0,
            "stall_shaping": 0.0,
            "stall_events": 0,
            "timeout_shaping": 0.0,
            "timeout_events": 0,
        }
        reward_keys = (
            "progress_shaping",
            "escape_shaping",
            "safe_shaping",
            "safe_approach_shaping",
            "home_approach_shaping",
            "home_shaping",
            "home_recovery_shaping",
            "home_ready_shaping",
            "safe_completion_shaping",
            "completion_shaping",
            "rejoin_shaping",
            "danger_damage_shaping",
            "danger_hold_shaping",
            "death_shaping",
            "stall_shaping",
            "timeout_shaping",
        )
        hero_state = None
        enemy_hero_states = []
        enemy_states = []
        own_crystal_state = None
        for hero in req_pb.hero_list:
            if hero.camp == self.hero_camp:
                hero_state = hero
            elif hero.hp > 0:
                enemy_hero_states.append(hero)
                enemy_states.append(hero)

        for organ in req_pb.organ_list:
            if organ.camp == self.hero_camp and organ.type == ACTOR_CRYSTAL:
                own_crystal_state = organ
            elif (
                organ.camp != self.hero_camp
                and organ.type in (ACTOR_TOWER, ACTOR_TOWER_HIGH, ACTOR_CRYSTAL)
                and organ.hp > 0
            ):
                enemy_states.append(organ)

        if hero_state is None or own_crystal_state is None or own_crystal_state.hp <= 0:
            self._retreat_cycle = None
            self._retreat_decision_context = {"must_retreat": False}
            return 0.0, details

        if hero_state.hp <= 0:
            if self._retreat_cycle is not None:
                details["death_shaping"] = self.config.RETREAT_CYCLE_DEATH_PENALTY
                details["death_events"] = int(details["death_shaping"] != 0.0)
            self._retreat_cycle = None
            self._retreat_decision_context = {"must_retreat": False}
            return details["death_shaping"], details

        frame_no = int(req_pb.frame_no)
        hp_rate = hero_state.hp / max(hero_state.max_hp, 1)
        ep_rate = self._get_ep_rate(hero_state)
        critical_hp = hp_rate < self.config.RETREAT_CYCLE_CRITICAL_HP_RATE
        critical_ep = (
            ep_rate is not None
            and ep_rate < self.config.RETREAT_CYCLE_CRITICAL_EP_RATE
        )
        critical_resource = critical_hp or critical_ep
        home_distance = self._distance(
            hero_state.location, own_crystal_state.location
        )
        # A full retreat must reach the crystal itself.  The broader resource
        # monitoring radius (12k by default) marks the lane-side base area and
        # would incorrectly credit a hero that has not yet reached resupply.
        near_home = home_distance < max(
            self.config.RETREAT_CYCLE_HOME_RADIUS, 1.0
        )
        far_home = not near_home
        nearest_enemy_distance = min(
            (
                self._distance(hero_state.location, enemy.location)
                for enemy in enemy_states
            ),
            default=float("inf"),
        )
        nearest_enemy_hero_distance = min(
            (
                self._distance(hero_state.location, enemy.location)
                for enemy in enemy_hero_states
            ),
            default=float("inf"),
        )
        danger = nearest_enemy_distance < self.config.RETREAT_DANGER_MAX_RADIUS

        dealt, taken, trade_elapsed = self._get_retreat_trade_window(
            frame_no, hero_state
        )
        trade_observed = (
            trade_elapsed >= self.config.RETREAT_CYCLE_MIN_TRADE_WINDOW_FRAMES
        )
        losing_trade = trade_observed and (
            taken >= hero_state.max_hp * self.config.RETREAT_CYCLE_LOSING_TRADE_MIN_HP_RATE
            and taken > dealt * self.config.RETREAT_CYCLE_LOSING_TRADE_RATIO
        )
        enemy_hp_rate = min(
            (
                enemy.hp / max(enemy.max_hp, 1)
                for enemy in enemy_hero_states
            ),
            default=None,
        )
        enemy_is_finishable = (
            enemy_hp_rate is not None
            and enemy_hp_rate + self.config.RETREAT_CYCLE_FINISHABLE_HP_MARGIN
            < hp_rate
            and dealt >= taken
        )
        hard_danger = hp_rate < self.config.RETREAT_CYCLE_HARD_HP_RATE

        should_start = critical_resource and far_home and (
            danger or not self.config.RETREAT_CYCLE_REQUIRE_DANGER
        )
        if self.config.RETREAT_CYCLE_REQUIRE_LOSING_TRADE:
            should_start = should_start and (
                hard_danger or (losing_trade and not enemy_is_finishable)
            )

        cycle = self._retreat_cycle
        if cycle is None:
            must_retreat = bool(should_start)
            self._retreat_decision_context = {"must_retreat": must_retreat}
            details["forced_retreat_samples"] = int(must_retreat)
            details["fight_allowed_samples"] = int(
                critical_resource and danger and not must_retreat
            )
            if must_retreat:
                self._retreat_cycle = {
                    "start_frame": frame_no,
                    "needs_hp": critical_hp,
                    "needs_ep": critical_ep,
                    "origin_hp_rate": hp_rate,
                    "origin_ep_rate": ep_rate,
                    "last_hp_rate": hp_rate,
                    "best_home_distance": home_distance,
                    "escape_anchor_home_distance": home_distance,
                    "escape_anchor_enemy_distance": nearest_enemy_hero_distance,
                    "escape_reward_total": 0.0,
                    "escape_reward_events": 0,
                    "last_escape_frame": frame_no,
                    "last_progress_frame": frame_no,
                    "last_frame": frame_no,
                    "safe_hold_frames": 0,
                    "safe_confirmed": False,
                    "safe_approach_anchor_distance": home_distance,
                    "safe_approach_reward_total": 0.0,
                    "safe_approach_reward_events": 0,
                    "home_approach_anchor_distance": None,
                    "home_approach_reward_total": 0.0,
                    "home_approach_reward_events": 0,
                    "home_recovery_anchor_progress": 0.0,
                    "home_recovery_reward_total": 0.0,
                    "home_recovery_reward_events": 0,
                    "home_ready_hold_frames": 0,
                    "home_ready_awarded": False,
                    "stall_events": 0,
                    "danger_hold_events": 0,
                    "phase": "retreat",
                }
                details["started"] = 1
            return 0.0, details

        # A completed recovery is no longer an emergency.  Keep the cycle
        # alive only to observe and reward its one-time rejoin milestone; do
        # not keep suppressing combat/economy shaping while the recovered hero
        # walks back into a useful area of the map.
        must_retreat = cycle["phase"] != "completed"
        self._retreat_decision_context = {"must_retreat": must_retreat}
        details["forced_retreat_samples"] = int(must_retreat)
        details["fight_allowed_samples"] = int(not must_retreat)

        if frame_no - cycle["start_frame"] > self.config.RETREAT_CYCLE_MAX_FRAMES:
            details["timeout_events"] = 1
            details["timeout_shaping"] = self.config.RETREAT_CYCLE_TIMEOUT_PENALTY
            self._retreat_cycle = None
            self._retreat_decision_context = {"must_retreat": False}
            return details["timeout_shaping"], details

        hp_loss = max(cycle["last_hp_rate"] - hp_rate, 0.0)
        frame_delta = max(frame_no - cycle["last_frame"], 0)
        damage_blocks_progress = (
            self.config.RETREAT_CYCLE_BLOCK_PROGRESS_ON_DAMAGE
            and danger
            and hp_loss > 0.0
        )
        if damage_blocks_progress:
            details["danger_reward_block_events"] = 1

        home_progress_blocked = (
            self.config.RETREAT_CYCLE_HOME_PROGRESS_REQUIRE_NO_DAMAGE
            and hp_loss > 0.0
        )
        home_recovery_blocked = (
            self.config.RETREAT_CYCLE_HOME_RECOVERY_REQUIRE_NO_DAMAGE
            and hp_loss > 0.0
        )

        if cycle["phase"] in ("retreat", "safe") and far_home and danger:
            raw_damage_penalty = (
                hp_loss * self.config.RETREAT_CYCLE_DANGER_DAMAGE_PENALTY_WEIGHT
            )
            if raw_damage_penalty < 0.0:
                details["danger_damage_shaping"] = max(
                    raw_damage_penalty,
                    self.config.RETREAT_CYCLE_DANGER_DAMAGE_PENALTY_MAX,
                )
                details["danger_damage_events"] = int(
                    details["danger_damage_shaping"] != 0.0
                )

        if cycle["phase"] in ("retreat", "safe") and far_home:
            if damage_blocks_progress:
                # Consume forward progress made while being damaged so it
                # cannot be paid retroactively on a later safe frame.
                if home_distance < cycle["best_home_distance"]:
                    cycle["best_home_distance"] = home_distance
                    cycle["last_progress_frame"] = frame_no
            else:
                step_distance = max(
                    self.config.RETREAT_CYCLE_PROGRESS_STEP_DISTANCE, 1.0
                )
                progressed_distance = cycle["best_home_distance"] - home_distance
                if progressed_distance >= step_distance:
                    events = int(progressed_distance / step_distance)
                    details["progress_events"] = events
                    details["progress_shaping"] = min(
                        events * self.config.RETREAT_CYCLE_PROGRESS_STEP_REWARD,
                        max(self.config.RETREAT_CYCLE_PROGRESS_MAX_REWARD, 0.0),
                    )
                    cycle["best_home_distance"] = home_distance
                    cycle["last_progress_frame"] = frame_no

        if cycle["phase"] == "retreat" and far_home:
            if damage_blocks_progress:
                # As above, consume the movement without paying it later.
                cycle["escape_anchor_enemy_distance"] = nearest_enemy_hero_distance
                cycle["escape_anchor_home_distance"] = home_distance
                cycle["last_escape_frame"] = frame_no
            else:
                escape_step_distance = max(
                    self.config.RETREAT_CYCLE_ESCAPE_STEP_DISTANCE, 0.0
                )
                min_home_progress = max(
                    self.config.RETREAT_CYCLE_ESCAPE_MIN_HOME_PROGRESS, 0.0
                )
                enemy_distance_progress = (
                    nearest_enemy_hero_distance
                    - cycle["escape_anchor_enemy_distance"]
                )
                home_distance_progress = (
                    cycle["escape_anchor_home_distance"] - home_distance
                )
                if (
                    escape_step_distance > 0.0
                    and np.isfinite(nearest_enemy_hero_distance)
                    and np.isfinite(cycle["escape_anchor_enemy_distance"])
                    and enemy_distance_progress >= escape_step_distance
                    and home_distance_progress >= min_home_progress
                ):
                    events = int(enemy_distance_progress / escape_step_distance)
                    step_reward = self.config.RETREAT_CYCLE_ESCAPE_STEP_REWARD
                    remaining_reward = max(
                        self.config.RETREAT_CYCLE_ESCAPE_MAX_REWARD
                        - cycle["escape_reward_total"],
                        0.0,
                    )
                    rewarded_events = 0
                    if step_reward > 0.0 and remaining_reward > 0.0:
                        # Count only complete, rewardable distance increments.
                        # This makes the configured maximum a true per-cycle cap
                        # and keeps the event metric resistant to reward farming.
                        available_events = int(
                            (remaining_reward + 1e-12) / step_reward
                        )
                        max_events = max(
                            self.config.RETREAT_CYCLE_ESCAPE_MAX_EVENTS, 0
                        )
                        if max_events > 0:
                            available_events = min(
                                available_events,
                                max(
                                    max_events
                                    - cycle["escape_reward_events"],
                                    0,
                                ),
                            )
                        rewarded_events = min(events, available_events)
                        details["escape_events"] = rewarded_events
                        details["escape_shaping"] = rewarded_events * step_reward
                        cycle["escape_reward_events"] += rewarded_events
                        cycle["escape_reward_total"] += details["escape_shaping"]

                    # Continue tracking genuine escape movement after the reward
                    # cap is reached.  Otherwise an escaping hero could be
                    # misclassified as standing still in danger.
                    cycle["escape_anchor_enemy_distance"] = nearest_enemy_hero_distance
                    cycle["escape_anchor_home_distance"] = home_distance
                    cycle["last_escape_frame"] = frame_no

            if (
                danger
                and self.config.RETREAT_CYCLE_DANGER_HOLD_FRAMES > 0
                and self.config.RETREAT_CYCLE_DANGER_HOLD_MAX_EVENTS > 0
                and cycle["danger_hold_events"]
                < self.config.RETREAT_CYCLE_DANGER_HOLD_MAX_EVENTS
                and frame_no - cycle["last_escape_frame"]
                >= self.config.RETREAT_CYCLE_DANGER_HOLD_FRAMES
            ):
                details["danger_hold_shaping"] = (
                    self.config.RETREAT_CYCLE_DANGER_HOLD_PENALTY
                )
                details["danger_hold_events"] = int(
                    details["danger_hold_shaping"] != 0.0
                )
                cycle["danger_hold_events"] += 1
                cycle["last_escape_frame"] = frame_no

        if cycle["phase"] in ("retreat", "safe"):
            min_progress = max(
                self.config.RETREAT_CYCLE_STALL_MIN_PROGRESS_DISTANCE, 0.0
            )
            if home_distance < cycle["best_home_distance"] - min_progress:
                cycle["best_home_distance"] = home_distance
                cycle["last_progress_frame"] = frame_no

            if (
                cycle["phase"] == "retreat"
                and self.config.RETREAT_CYCLE_STALL_FRAMES > 0
                and self.config.RETREAT_CYCLE_STALL_MAX_EVENTS > 0
                and cycle["stall_events"]
                < self.config.RETREAT_CYCLE_STALL_MAX_EVENTS
                and frame_no - cycle["last_progress_frame"]
                >= self.config.RETREAT_CYCLE_STALL_FRAMES
            ):
                details["stall_shaping"] = self.config.RETREAT_CYCLE_STALL_PENALTY
                details["stall_events"] = int(details["stall_shaping"] != 0.0)
                cycle["stall_events"] += 1
                cycle["last_progress_frame"] = frame_no

        safe_radius = max(self.config.RETREAT_CYCLE_SAFE_HOME_RADIUS, 0.0)

        # The old curriculum paid for leaving the enemy and then for walking
        # home only after safety had already been confirmed.  In practice,
        # the policy often died in that gap.  Pay a capped reward for each
        # new, forward-only movement segment toward the reachable safe zone.
        # It is available only in an existing must-retreat cycle and stops as
        # soon as the hero reaches that zone, so it cannot reward ordinary
        # lane movement or back-and-forth farming.
        if (
            cycle["phase"] == "retreat"
            and safe_radius > 0.0
            and home_distance > safe_radius
        ):
            approach_anchor = cycle["safe_approach_anchor_distance"]
            if damage_blocks_progress:
                cycle["safe_approach_anchor_distance"] = min(
                    approach_anchor, home_distance
                )
            else:
                approach_step_distance = max(
                    self.config.RETREAT_CYCLE_SAFE_APPROACH_STEP_DISTANCE, 0.0
                )
                safe_approach_progress = approach_anchor - home_distance
                if (
                    approach_step_distance > 0.0
                    and safe_approach_progress >= approach_step_distance
                ):
                    events = int(safe_approach_progress / approach_step_distance)
                    step_reward = self.config.RETREAT_CYCLE_SAFE_APPROACH_STEP_REWARD
                    remaining_reward = max(
                        self.config.RETREAT_CYCLE_SAFE_APPROACH_MAX_REWARD
                        - cycle["safe_approach_reward_total"],
                        0.0,
                    )
                    if step_reward > 0.0 and remaining_reward > 0.0:
                        available_events = int(
                            (remaining_reward + 1e-12) / step_reward
                        )
                        max_events = max(
                            self.config.RETREAT_CYCLE_SAFE_APPROACH_MAX_EVENTS,
                            0,
                        )
                        if max_events > 0:
                            available_events = min(
                                available_events,
                                max(
                                    max_events
                                    - cycle["safe_approach_reward_events"],
                                    0,
                                ),
                            )
                        rewarded_events = min(events, available_events)
                        details["safe_approach_events"] = rewarded_events
                        details["safe_approach_shaping"] = (
                            rewarded_events * step_reward
                        )
                        cycle["safe_approach_reward_events"] += rewarded_events
                        cycle["safe_approach_reward_total"] += details[
                            "safe_approach_shaping"
                        ]

                    # Advance the anchor even after reaching the reward cap.
                    # Future movement must therefore exceed the best distance
                    # already achieved before it can receive another event.
                    cycle["safe_approach_anchor_distance"] = home_distance

        safe_enemy_distance = (
            nearest_enemy_hero_distance
            if self.config.RETREAT_CYCLE_SAFE_USE_HERO_DISTANCE
            else nearest_enemy_distance
        )
        safe_hold_damage_blocked = (
            self.config.RETREAT_CYCLE_SAFE_REQUIRE_NO_DAMAGE
            and hp_loss > 0.0
        )
        safe_zone = safe_radius > 0 and (
            home_distance <= safe_radius
            and safe_enemy_distance
            >= self.config.RETREAT_CYCLE_SAFE_ENEMY_DISTANCE
            and not safe_hold_damage_blocked
        )

        # Safety entry requires both distance and a damage-free hold.  Once
        # established, retain the Stage-2 walk while the enemy merely moves
        # closer; only actual new damage proves that the retreat failed and
        # should send the hero back to the escape phase.  Revoking safety on
        # distance alone made agents oscillate before reaching the crystal.
        if cycle["phase"] == "safe" and safe_hold_damage_blocked:
            cycle.update(
                {
                    "phase": "retreat",
                    "safe_confirmed": False,
                    "safe_hold_frames": 0,
                    "safe_approach_anchor_distance": home_distance,
                    "home_approach_anchor_distance": None,
                }
            )

        if cycle["phase"] == "retreat" and safe_radius > 0:
            frame_delta = max(frame_no - cycle["last_frame"], 0)
            if safe_zone:
                cycle["safe_hold_frames"] += frame_delta
            else:
                cycle["safe_hold_frames"] = 0
            if cycle["safe_hold_frames"] >= self.config.RETREAT_CYCLE_SAFE_HOLD_FRAMES:
                details["safe_shaping"] = self.config.RETREAT_CYCLE_SAFE_ARRIVAL_REWARD
                details["safe_events"] = int(details["safe_shaping"] != 0.0)
                cycle["safe_confirmed"] = True
                # Begin measuring the Stage-2 walk only after the hero has
                # actually demonstrated safety.  This prevents the earlier
                # escape phase from consuming the approach budget.
                cycle["home_approach_anchor_distance"] = home_distance
                if self.config.RETREAT_CYCLE_SAFE_COMPLETES_CYCLE:
                    self._retreat_cycle = None
                    self._retreat_decision_context = {"must_retreat": False}
                    return sum(details[key] for key in reward_keys), details
                cycle["phase"] = "safe"

        # Reaching the broad base-side safe zone is deliberately not the same
        # as reaching the crystal.  Give a capped, distance-based incentive
        # for continuing from confirmed safety to the actual resupply area.
        # It becomes available only in the safe phase, so a hero cannot farm
        # it while still exposed to the enemy.
        if cycle["phase"] == "safe" and far_home:
            approach_step_distance = max(
                self.config.RETREAT_CYCLE_HOME_APPROACH_STEP_DISTANCE, 0.0
            )
            approach_anchor = cycle["home_approach_anchor_distance"]
            if approach_anchor is None:
                approach_anchor = home_distance
                cycle["home_approach_anchor_distance"] = home_distance
            if home_progress_blocked:
                cycle["home_approach_anchor_distance"] = min(
                    approach_anchor, home_distance
                )
            else:
                approach_progress = approach_anchor - home_distance
                if (
                    approach_step_distance > 0.0
                    and approach_progress >= approach_step_distance
                ):
                    events = int(approach_progress / approach_step_distance)
                    step_reward = self.config.RETREAT_CYCLE_HOME_APPROACH_STEP_REWARD
                    remaining_reward = max(
                        self.config.RETREAT_CYCLE_HOME_APPROACH_MAX_REWARD
                        - cycle["home_approach_reward_total"],
                        0.0,
                    )
                    rewarded_events = 0
                    if step_reward > 0.0 and remaining_reward > 0.0:
                        available_events = int(
                            (remaining_reward + 1e-12) / step_reward
                        )
                        max_events = max(
                            self.config.RETREAT_CYCLE_HOME_APPROACH_MAX_EVENTS,
                            0,
                        )
                        if max_events > 0:
                            available_events = min(
                                available_events,
                                max(
                                    max_events
                                    - cycle["home_approach_reward_events"],
                                    0,
                                ),
                            )
                        rewarded_events = min(events, available_events)
                        details["home_approach_events"] = rewarded_events
                        details["home_approach_shaping"] = (
                            rewarded_events * step_reward
                        )
                        cycle["home_approach_reward_events"] += rewarded_events
                        cycle["home_approach_reward_total"] += details[
                            "home_approach_shaping"
                        ]

                    # Keep measuring genuine progress even after the per-cycle
                    # cap so one earlier cap cannot be reused on later movement.
                    cycle["home_approach_anchor_distance"] = home_distance

        # When a safety milestone is configured, crystal proximity by itself
        # is not enough.  The hero must first remain out of the enemy's threat
        # range for SAFE_HOLD_FRAMES and receive a confirmed safety milestone.
        home_arrival_allowed = safe_radius <= 0 or cycle["safe_confirmed"]
        if (
            cycle["phase"] in ("retreat", "safe")
            and near_home
            and home_arrival_allowed
            and not home_progress_blocked
        ):
            details["home_shaping"] = self.config.RETREAT_CYCLE_HOME_ARRIVAL_REWARD
            details["home_events"] = int(details["home_shaping"] != 0.0)
            cycle.update(
                {
                    "phase": "home",
                    "home_arrival_hp_rate": hp_rate,
                    "home_arrival_ep_rate": ep_rate,
                    "home_recovery_anchor_progress": 0.0,
                    "home_ready_hold_frames": 0,
                }
            )

        # Only crystal-side recovery counts.  In the home-sanctuary profile,
        # a tight own-crystal location plus no new damage is sufficient;
        # enemy-distance changes elsewhere must not turn off resupply.
        home_recovery_safe = not self.config.RETREAT_CYCLE_HOME_RECOVERY_REQUIRE_SAFE
        if self.config.RETREAT_CYCLE_HOME_RECOVERY_REQUIRE_SAFE:
            home_recovery_safe = (
                near_home
                if self.config.RETREAT_CYCLE_HOME_RECOVERY_USE_HOME_SANCTUARY
                else safe_zone
            )
        home_ready_conditions_met = False
        if (
            cycle["phase"] == "home"
            and near_home
            and home_recovery_safe
            and not home_recovery_blocked
        ):
            recovery_gains = []
            if cycle["needs_hp"]:
                recovery_gains.append(
                    max(hp_rate - cycle["home_arrival_hp_rate"], 0.0)
                )
            if (
                cycle["needs_ep"]
                and ep_rate is not None
                and cycle["home_arrival_ep_rate"] is not None
            ):
                recovery_gains.append(
                    max(ep_rate - cycle["home_arrival_ep_rate"], 0.0)
                )

            # Use the least-recovered resource that triggered this retreat.
            # HP-only and EP-only retreats still use their own gain, while a
            # retreat caused by both resources earns recovery reward only as
            # its slower resource recovers.  An average would pay for a
            # partial resupply (for example, HP restored while EP stays low)
            # even though it can never satisfy the completion condition.
            # The anchor only moves forward, preventing recovery reward
            # farming by taking damage and healing it again.
            recovery_progress = (
                min(recovery_gains)
                if recovery_gains
                else 0.0
            )
            recovery_step_gain = max(
                self.config.RETREAT_CYCLE_HOME_RECOVERY_STEP_GAIN, 0.0
            )
            recovery_progress_delta = (
                recovery_progress - cycle["home_recovery_anchor_progress"]
            )
            if (
                recovery_step_gain > 0.0
                and recovery_progress_delta >= recovery_step_gain
            ):
                events = int(recovery_progress_delta / recovery_step_gain)
                step_reward = self.config.RETREAT_CYCLE_HOME_RECOVERY_STEP_REWARD
                remaining_reward = max(
                    self.config.RETREAT_CYCLE_HOME_RECOVERY_MAX_REWARD
                    - cycle["home_recovery_reward_total"],
                    0.0,
                )
                if step_reward > 0.0 and remaining_reward > 0.0:
                    available_events = int(
                        (remaining_reward + 1e-12) / step_reward
                    )
                    rewarded_events = min(events, available_events)
                    details["home_recovery_events"] = rewarded_events
                    details["home_recovery_shaping"] = (
                        rewarded_events * step_reward
                    )
                    cycle["home_recovery_reward_events"] += rewarded_events
                    cycle["home_recovery_reward_total"] += details[
                        "home_recovery_shaping"
                    ]

                # Record the best observed home recovery progress even when
                # the configured reward cap is reached.
                cycle["home_recovery_anchor_progress"] = recovery_progress

            # Reward one stable recovery milestone rather than every healing
            # tick.  It only becomes available at the crystal side, after a
            # real resource gain, and the counter resets immediately on a
            # damage, safety, or location violation.
            ready_hold_frames = max(
                self.config.RETREAT_CYCLE_HOME_READY_HOLD_FRAMES, 0
            )
            ready_min_gain = max(
                self.config.RETREAT_CYCLE_HOME_READY_MIN_RECOVERY_GAIN, 0.0
            )
            home_ready_conditions_met = recovery_progress >= ready_min_gain
            if home_ready_conditions_met:
                cycle["home_ready_hold_frames"] += frame_delta

            if (
                ready_hold_frames > 0
                and not cycle["home_ready_awarded"]
                and cycle["home_ready_hold_frames"] >= ready_hold_frames
            ):
                details["home_ready_shaping"] = (
                    self.config.RETREAT_CYCLE_HOME_READY_REWARD
                )
                details["home_ready_events"] = int(
                    details["home_ready_shaping"] != 0.0
                )
                cycle["home_ready_awarded"] = True

            hp_completed = (
                not cycle["needs_hp"]
                or (
                    hp_rate >= self.config.RETREAT_CYCLE_TARGET_HP_RATE
                    and hp_rate - cycle["origin_hp_rate"]
                    >= self.config.RETREAT_CYCLE_MIN_HP_GAIN
                    and hp_rate - cycle["home_arrival_hp_rate"]
                    >= self.config.RETREAT_CYCLE_HOME_MIN_HP_GAIN
                )
            )
            ep_completed = (
                not cycle["needs_ep"]
                or (
                    ep_rate is not None
                    and cycle["origin_ep_rate"] is not None
                    and cycle["home_arrival_ep_rate"] is not None
                    and ep_rate >= self.config.RETREAT_CYCLE_TARGET_EP_RATE
                    and ep_rate - cycle["origin_ep_rate"]
                    >= self.config.RETREAT_CYCLE_MIN_EP_GAIN
                    and ep_rate - cycle["home_arrival_ep_rate"]
                    >= self.config.RETREAT_CYCLE_HOME_MIN_EP_GAIN
                )
            )
            if hp_completed and ep_completed:
                details["safe_completion_shaping"] = (
                    self.config.RETREAT_CYCLE_SAFE_COMPLETION_BONUS
                )
                details["safe_completion_events"] = int(
                    details["safe_completion_shaping"] != 0.0
                )
                details["completion_shaping"] = (
                    self.config.RETREAT_CYCLE_COMPLETION_REWARD
                )
                details["completion_events"] = int(
                    details["completion_shaping"] != 0.0
                )
                cycle["phase"] = "completed"
                self._retreat_decision_context = {"must_retreat": False}

        if cycle["phase"] == "home" and not home_ready_conditions_met:
            cycle["home_ready_hold_frames"] = 0

        if cycle["phase"] == "completed":
            fully_recovered = (
                hp_rate >= self.config.RETREAT_CYCLE_TARGET_HP_RATE
                and (
                    ep_rate is None
                    or not cycle["needs_ep"]
                    or ep_rate >= self.config.RETREAT_CYCLE_TARGET_EP_RATE
                )
            )
            if (
                fully_recovered
                and home_distance >= self.config.RETREAT_CYCLE_REJOIN_HOME_DISTANCE
            ):
                details["rejoin_shaping"] = self.config.RETREAT_CYCLE_REJOIN_REWARD
                details["rejoin_events"] = int(details["rejoin_shaping"] != 0.0)
                self._retreat_cycle = None

        if self._retreat_cycle is None:
            self._retreat_decision_context = {"must_retreat": False}

        cycle["last_hp_rate"] = hp_rate
        cycle["last_frame"] = frame_no
        reward = sum(details[key] for key in reward_keys)
        return reward, details

    def _calc_progress_shaping_reward_details(self, req_pb):
        """Return trade shaping plus its unclipped value for monitoring."""
        hero_state = None
        for hero in req_pb.hero_list:
            if hero.camp == self.hero_camp:
                hero_state = hero
                break

        if hero_state is None:
            return 0.0, 0.0, False

        enemy_objective_hp = 0
        own_objective_hp = 0
        own_crystal_state = None
        for organ in req_pb.organ_list:
            if organ.camp != self.hero_camp and organ.type in (
                ACTOR_TOWER,
                ACTOR_TOWER_HIGH,
                ACTOR_CRYSTAL,
            ):
                enemy_objective_hp += max(organ.hp, 0)
            elif organ.camp == self.hero_camp and organ.type in (
                ACTOR_TOWER,
                ACTOR_TOWER_HIGH,
                ACTOR_CRYSTAL,
            ):
                own_objective_hp += max(organ.hp, 0)
                if organ.type == ACTOR_CRYSTAL:
                    own_crystal_state = organ

        ep = self._get_first_attr(hero_state, ("ep", "ep_point", "energy", "mp"), None)
        ep_rate = self._get_ep_rate(hero_state)
        current_stats = {
            "frame_no": req_pb.frame_no,
            "hp": hero_state.hp,
            "hp_rate": hero_state.hp / max(hero_state.max_hp, 1),
            "ep": ep,
            "ep_rate": ep_rate,
            "money": hero_state.moneyCnt,
            "total_hurt": hero_state.totalHurt,
            "hero_hurt": hero_state.totalHurtToHero,
            "be_hurt_by_hero": hero_state.totalBeHurtByHero,
            "kill": hero_state.killCnt,
            "dead": hero_state.deadCnt,
            "enemy_objective_hp": enemy_objective_hp,
            "own_objective_hp": own_objective_hp,
        }
        last_stats = self._last_shaping_stats
        self._last_shaping_stats = current_stats

        if last_stats is None or current_stats["frame_no"] <= last_stats["frame_no"]:
            return 0.0, 0.0, False

        hero_hurt_delta = max(
            current_stats["hero_hurt"] - last_stats["hero_hurt"], 0
        )
        be_hurt_by_hero_delta = max(
            current_stats["be_hurt_by_hero"] - last_stats["be_hurt_by_hero"], 0
        )
        net_trade_delta = (
            hero_hurt_delta
            - be_hurt_by_hero_delta * self.config.SHAPING_NET_TRADE_DAMAGE_RATIO
        )

        max_hp = max(hero_state.max_hp, 1)
        hp_rate = hero_state.hp / max_hp
        must_retreat = self._retreat_decision_context.get("must_retreat", False)
        if (
            hp_rate < self.config.OBJECTIVE_DISTANCE_HEALTHY_HP_RATE
            or (
                must_retreat
                and self.config.RETREAT_CYCLE_SUPPRESS_OPPORTUNITY_REWARD
            )
        ):
            # Do not pay for aggressive trades while low, but preserve the
            # negative half of a bad trade so retreat remains preferable.
            net_trade_delta = min(net_trade_delta, 0)

        raw_reward = net_trade_delta * self.config.SHAPING_NET_TRADE_WEIGHT
        reward_clip = self.config.SHAPING_DELTA_REWARD_CLIP
        reward = max(min(raw_reward, reward_clip), -reward_clip)
        return reward, raw_reward, abs(raw_reward) > reward_clip

    def _calc_progress_shaping_reward(self, req_pb):
        """Return the reward component used by training samples."""
        return self._calc_progress_shaping_reward_details(req_pb)[0]

    def _calc_economy_shaping_reward_details(self, req_pb):
        """Reward newly-earned gold relative to the opposing hero.

        The 1v1 SDK exposes a ``reward_money`` vector element, but it is zero
        for the currently deployed GameCore build.  ``moneyCnt`` in the frame
        state is populated, so use positive increments from that authoritative
        observation.  Ignoring negative deltas means an automatic item purchase
        cannot create an artificial training penalty.
        """
        own_hero = None
        enemy_hero = None
        for hero_state in req_pb.hero_list:
            if hero_state.camp == self.hero_camp:
                own_hero = hero_state
            else:
                enemy_hero = hero_state

        if own_hero is None or enemy_hero is None:
            return 0.0, 0.0, 0.0, 0.0

        current_stats = {
            "frame_no": req_pb.frame_no,
            "own_money": float(own_hero.moneyCnt),
            "enemy_money": float(enemy_hero.moneyCnt),
        }
        last_stats = self._last_economy_stats
        self._last_economy_stats = current_stats
        if last_stats is None or current_stats["frame_no"] <= last_stats["frame_no"]:
            return 0.0, 0.0, 0.0, 0.0

        own_gain = max(current_stats["own_money"] - last_stats["own_money"], 0.0)
        enemy_gain = max(
            current_stats["enemy_money"] - last_stats["enemy_money"], 0.0
        )
        raw_reward = (
            own_gain - enemy_gain
        ) * self.config.SHAPING_GOLD_LEAD_WEIGHT
        reward_clip = self.config.SHAPING_GOLD_LEAD_CLIP
        reward = max(min(raw_reward, reward_clip), -reward_clip)
        if (
            reward > 0.0
            and self._retreat_decision_context.get("must_retreat", False)
            and self.config.RETREAT_CYCLE_SUPPRESS_OPPORTUNITY_REWARD
        ):
            reward = 0.0
        return reward, raw_reward, own_gain, enemy_gain

    def _calc_objective_shaping_reward_details(self, req_pb):
        """Reward net tower/crystal HP progress without double-counting state.

        A positive delta means enemy objectives lost more HP than ours since
        the previous decision.  Unlike a proximity bonus, this is paid only
        for actual map progress and is symmetric when defending our own base.
        """
        enemy_objective_hp = 0.0
        own_objective_hp = 0.0
        objective_types = (ACTOR_TOWER, ACTOR_TOWER_HIGH, ACTOR_CRYSTAL)
        for organ in req_pb.organ_list:
            if organ.type not in objective_types:
                continue
            if organ.camp == self.hero_camp:
                own_objective_hp += max(float(organ.hp), 0.0)
            else:
                enemy_objective_hp += max(float(organ.hp), 0.0)

        current_stats = {
            "frame_no": req_pb.frame_no,
            "enemy_objective_hp": enemy_objective_hp,
            "own_objective_hp": own_objective_hp,
        }
        last_stats = self._last_objective_stats
        self._last_objective_stats = current_stats
        if last_stats is None or current_stats["frame_no"] <= last_stats["frame_no"]:
            return 0.0, 0.0, 0.0, 0.0, False

        enemy_damage = max(
            last_stats["enemy_objective_hp"] - current_stats["enemy_objective_hp"],
            0.0,
        )
        own_damage = max(
            last_stats["own_objective_hp"] - current_stats["own_objective_hp"],
            0.0,
        )
        raw_reward = (
            enemy_damage - own_damage
        ) * self.config.SHAPING_OBJECTIVE_PROGRESS_WEIGHT
        reward_clip = self.config.SHAPING_OBJECTIVE_PROGRESS_CLIP
        reward = max(min(raw_reward, reward_clip), -reward_clip)
        return reward, raw_reward, enemy_damage, own_damage, abs(raw_reward) > reward_clip

    def _reset_reward_component_stats(self):
        """Reset episode-local reward diagnostics without changing reward math."""
        self._reward_component_stats = {
            "base_reward": 0.0,
            "position_shaping": 0.0,
            "low_resource_position_shaping": 0.0,
            "low_resource_position_samples": 0,
            "low_resource_far_home_position_shaping": 0.0,
            "low_resource_far_home_position_samples": 0,
            "retreat_recovery_shaping": 0.0,
            "retreat_recovery_events": 0,
            "retreat_danger_shaping": 0.0,
            "retreat_danger_samples": 0,
            "retreat_progress_shaping": 0.0,
            "retreat_progress_events": 0,
            "retreat_regression_shaping": 0.0,
            "retreat_regression_events": 0,
            "retreat_cycle_shaping": 0.0,
            "retreat_cycle_started": 0,
            "retreat_cycle_forced_retreat_samples": 0,
            "retreat_cycle_fight_allowed_samples": 0,
            "retreat_cycle_progress_shaping": 0.0,
            "retreat_cycle_progress_events": 0,
            "retreat_cycle_escape_shaping": 0.0,
            "retreat_cycle_escape_events": 0,
            "retreat_cycle_safe_shaping": 0.0,
            "retreat_cycle_safe_events": 0,
            "retreat_cycle_safe_approach_shaping": 0.0,
            "retreat_cycle_safe_approach_events": 0,
            "retreat_cycle_home_approach_shaping": 0.0,
            "retreat_cycle_home_approach_events": 0,
            "retreat_cycle_home_shaping": 0.0,
            "retreat_cycle_home_events": 0,
            "retreat_cycle_home_recovery_shaping": 0.0,
            "retreat_cycle_home_recovery_events": 0,
            "retreat_cycle_home_ready_shaping": 0.0,
            "retreat_cycle_home_ready_events": 0,
            "retreat_cycle_safe_completion_shaping": 0.0,
            "retreat_cycle_safe_completion_events": 0,
            "retreat_cycle_completion_shaping": 0.0,
            "retreat_cycle_completion_events": 0,
            "retreat_cycle_rejoin_shaping": 0.0,
            "retreat_cycle_rejoin_events": 0,
            "retreat_cycle_danger_damage_shaping": 0.0,
            "retreat_cycle_danger_damage_events": 0,
            "retreat_cycle_danger_reward_block_events": 0,
            "retreat_cycle_danger_hold_shaping": 0.0,
            "retreat_cycle_danger_hold_events": 0,
            "retreat_cycle_death_shaping": 0.0,
            "retreat_cycle_death_events": 0,
            "retreat_cycle_stall_shaping": 0.0,
            "retreat_cycle_stall_events": 0,
            "retreat_cycle_timeout_shaping": 0.0,
            "retreat_cycle_timeout_events": 0,
            "trade_shaping": 0.0,
            "trade_shaping_raw": 0.0,
            "trade_clip_steps": 0,
            "economy_shaping": 0.0,
            "economy_shaping_raw": 0.0,
            "economy_money_gain": 0.0,
            "economy_enemy_money_gain": 0.0,
            "objective_shaping": 0.0,
            "objective_shaping_raw": 0.0,
            "objective_enemy_damage": 0.0,
            "objective_own_damage": 0.0,
            "objective_clip_steps": 0,
            "reward_samples": 0,
        }
        for name in _ENV_REWARD_COMPONENTS:
            self._reward_component_stats["env_{}".format(name)] = 0.0

    def _record_reward_components(
        self,
        base_reward,
        position_shaping,
        trade_shaping,
        trade_shaping_raw,
        trade_clipped,
        economy_shaping,
        economy_shaping_raw,
        economy_money_gain,
        economy_enemy_money_gain,
        objective_shaping,
        objective_shaping_raw,
        objective_enemy_damage,
        objective_own_damage,
        objective_clipped,
        retreat_recovery_reward,
        retreat_recovery_event,
        retreat_danger_penalty,
        retreat_danger,
        retreat_progress_reward,
        retreat_progress_events,
        retreat_regression_penalty,
        retreat_regression_event,
        env_reward_components,
        retreat_cycle_reward,
        retreat_cycle_details,
    ):
        stats = self._reward_component_stats
        stats["base_reward"] += float(base_reward)
        stats["position_shaping"] += float(position_shaping)
        position_context = self._last_position_context
        if position_context.get("low_resource", False):
            stats["low_resource_position_shaping"] += float(position_shaping)
            stats["low_resource_position_samples"] += 1
            if position_context.get("low_resource_far_home", False):
                stats["low_resource_far_home_position_shaping"] += float(
                    position_shaping
                )
                stats["low_resource_far_home_position_samples"] += 1
        stats["retreat_recovery_shaping"] += float(retreat_recovery_reward)
        stats["retreat_recovery_events"] += int(retreat_recovery_event)
        stats["retreat_danger_shaping"] += float(retreat_danger_penalty)
        stats["retreat_danger_samples"] += int(retreat_danger)
        stats["retreat_progress_shaping"] += float(retreat_progress_reward)
        stats["retreat_progress_events"] += int(retreat_progress_events)
        stats["retreat_regression_shaping"] += float(retreat_regression_penalty)
        stats["retreat_regression_events"] += int(retreat_regression_event)
        stats["retreat_cycle_shaping"] += float(retreat_cycle_reward)
        for name, value in retreat_cycle_details.items():
            stats["retreat_cycle_{}".format(name)] += value
        stats["trade_shaping"] += float(trade_shaping)
        stats["trade_shaping_raw"] += float(trade_shaping_raw)
        stats["trade_clip_steps"] += int(trade_clipped)
        stats["economy_shaping"] += float(economy_shaping)
        stats["economy_shaping_raw"] += float(economy_shaping_raw)
        stats["economy_money_gain"] += float(economy_money_gain)
        stats["economy_enemy_money_gain"] += float(economy_enemy_money_gain)
        stats["objective_shaping"] += float(objective_shaping)
        stats["objective_shaping_raw"] += float(objective_shaping_raw)
        stats["objective_enemy_damage"] += float(objective_enemy_damage)
        stats["objective_own_damage"] += float(objective_own_damage)
        stats["objective_clip_steps"] += int(objective_clipped)
        stats["reward_samples"] += 1
        for name, value in env_reward_components.items():
            stats[name] += float(value)

    def get_reward_component_summary(self):
        """Return a copy so actor logging cannot mutate the live counters."""
        return dict(self._reward_component_stats)

    @staticmethod
    def _get_base_reward(reward):
        base_reward, _ = Agent._get_base_reward_components(reward)
        return base_reward

    @staticmethod
    def _get_base_reward_components(reward):
        """Return the environment total and its raw diagnostic vector.

        The 1v1 SDK exposes raw reward as
        ``[dead, ep_rate, exp, hp_point, kill, last_hit, money,
        tower_hp_point, total]``.  The entries are raw SDK signals, not
        addends in the final weighted reward, so do not try to reconstruct
        ``total`` by summing them.
        """
        components = {
            "env_{}".format(name): 0.0 for name in _ENV_REWARD_COMPONENTS
        }
        if reward is None:
            return 0.0, components
        if isinstance(reward, (tuple, list, np.ndarray)):
            values = np.asarray(reward).reshape(-1)
            if values.size == 0:
                return 0.0, components
            base_reward = float(values[-1])
            for index, name in enumerate(_ENV_REWARD_COMPONENTS):
                if index < values.size - 1:
                    components["env_{}".format(name)] = float(values[index])
            return base_reward, components

        base_reward = float(reward)
        return base_reward, components

    def get_shaped_reward(self, state_dict):
        """Build the one reward used for both normal and terminal samples."""
        req_pb = state_dict["req_pb"]
        base_reward, env_reward_components = self._get_base_reward_components(
            state_dict["reward"]
        )
        retreat_cycle_shaping, retreat_cycle_details = (
            self._calc_retreat_cycle_shaping_reward_details(req_pb)
        )
        position_shaping = self._calc_position_shaping_reward(req_pb)
        (
            trade_shaping,
            trade_shaping_raw,
            trade_clipped,
        ) = self._calc_progress_shaping_reward_details(req_pb)
        (
            economy_shaping,
            economy_shaping_raw,
            economy_money_gain,
            economy_enemy_money_gain,
        ) = self._calc_economy_shaping_reward_details(req_pb)
        (
            objective_shaping,
            objective_shaping_raw,
            objective_enemy_damage,
            objective_own_damage,
            objective_clipped,
        ) = self._calc_objective_shaping_reward_details(req_pb)
        (
            retreat_shaping,
            retreat_recovery_reward,
            retreat_danger_penalty,
            retreat_progress_reward,
            retreat_progress_events,
            retreat_regression_penalty,
            retreat_regression_event,
            retreat_recovery_event,
            retreat_danger,
        ) = self._calc_retreat_recovery_shaping_reward_details(req_pb)
        self._record_reward_components(
            base_reward,
            position_shaping,
            trade_shaping,
            trade_shaping_raw,
            trade_clipped,
            economy_shaping,
            economy_shaping_raw,
            economy_money_gain,
            economy_enemy_money_gain,
            objective_shaping,
            objective_shaping_raw,
            objective_enemy_damage,
            objective_own_damage,
            objective_clipped,
            retreat_recovery_reward,
            retreat_recovery_event,
            retreat_danger_penalty,
            retreat_danger,
            retreat_progress_reward,
            retreat_progress_events,
            retreat_regression_penalty,
            retreat_regression_event,
            env_reward_components,
            retreat_cycle_shaping,
            retreat_cycle_details,
        )
        return (
            base_reward
            + position_shaping
            + trade_shaping
            + economy_shaping
            + objective_shaping
            + retreat_shaping
            + retreat_cycle_shaping
        )

    # build samples from state infos
    def _sample_process(self, state_dict, pred_ret):
        # get is_train
        is_train = False
        req_pb = state_dict["req_pb"]
        for hero in req_pb.hero_list:
            if hero.camp == self.hero_camp:
                is_train = True if hero.hp > 0 else False

        frame_no = req_pb.frame_no
        feature_vec, _, sub_action_mask = (
            state_dict["observation"],
            state_dict["reward"],
            state_dict["sub_action_mask"],
        )
        shaped_reward = self.get_shaped_reward(state_dict)
        done = False
        prob, value, action, _ = pred_ret

        legal_action = self._update_legal_action(state_dict["legal_action"], action)
        keys = (
            "frame_no",
            "vec_feature",
            "legal_action",
            "action",
            "reward",
            "value",
            "prob",
            "sub_action",
            "lstm_cell",
            "lstm_hidden",
            "done",
            "is_train",
        )
        values = (
            frame_no,
            feature_vec,
            legal_action,
            action,
            shaped_reward,
            value,
            prob,
            sub_action_mask,
            self.lstm_cell,
            self.lstm_hidden,
            done,
            is_train,
        )
        sample = dict(zip(keys, values))
        self.last_sample = sample

        if self.save_h5_sample:
            self._sample_process_for_saver(sample)
        return sample

    def _get_h5file_keys(self, h5file):
        keys = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                keys.append(name)

        h5file.visititems(visitor)
        return keys

    def _sample_process_for_saver(self, sample_dict):
        keys = ("frame_no", "vec_feature", "legal_action", "action", "reward", "done")
        keys_in_h5 = self._get_h5file_keys(self.dataset)
        if len(keys_in_h5) == 0:
            self.dataset.create_dataset(
                "frame_no",
                data=[[sample_dict["frame_no"]]],
                compression="gzip",
                maxshape=(None, 1),
                chunks=True,
            )
            self.dataset.create_dataset(
                "observation",
                data=[sample_dict["vec_feature"]],
                compression="gzip",
                maxshape=(None, len(sample_dict["vec_feature"])),
                chunks=True,
            )
            self.dataset.create_dataset(
                "legal_action",
                data=[sample_dict["legal_action"]],
                compression="gzip",
                maxshape=(None, len(sample_dict["legal_action"])),
                chunks=True,
            )
            self.dataset.create_dataset(
                "action",
                data=[sample_dict["action"]],
                compression="gzip",
                maxshape=(None, len(sample_dict["action"])),
                chunks=True,
            )
            self.dataset.create_dataset(
                "reward",
                data=[[sample_dict["reward"]]],
                compression="gzip",
                maxshape=(None, 1),
                chunks=True,
            )
            self.dataset.create_dataset(
                "done",
                data=[[sample_dict["done"]]],
                compression="gzip",
                maxshape=(None, 1),
                chunks=True,
            )

        else:
            for key, value in sample_dict.items():
                if key in keys:
                    key_dataset = key
                    if key_dataset == "vec_feature":
                        key_dataset = "observation"
                    self.dataset[key_dataset].resize(
                        (self.dataset[key_dataset].shape[0] + 1), axis=0
                    )
                    if isinstance(value, list):
                        self.dataset[key_dataset][-1] = value
                    else:
                        self.dataset[key_dataset][-1] = [value]

    # given the feature vec and legal_action,return output of the network
    def _predict_process(self, feature, legal_action):
        # put data to input
        input_list = cvt_tensor_to_infer_input(self.model.get_input_tensors())
        input_list[0].set_data(np.array(feature))
        input_list[1].set_data(np.array(legal_action))
        # input_list[2].set_data(label_list)
        input_list[2].set_data(self.lstm_cell)
        input_list[3].set_data(self.lstm_hidden)

        output_list = cvt_tensor_to_infer_output(self.model.get_output_tensors())
        output_list = self._predictor.inference(
            input_list=input_list, output_list=output_list
        )
        # cvt output data
        np_output = cvt_infer_list_to_numpy_list(output_list)

        logits, value, self.lstm_cell, self.lstm_hidden = np_output[:4]

        prob, action, d_action = self._sample_masked_action(logits, legal_action)

        return prob, value, action, d_action  # prob: [[ ]], others: all 1D

    def _predict_process_torch(self, feature, legal_action):
        # put data to input
        input_list = []
        input_list.append(np.array(feature))
        input_list.append(np.array(legal_action))
        input_list.append(self.lstm_cell)
        input_list.append(self.lstm_hidden)

        output_list = self._predictor.inference(input_list)
        np_output = []
        for output in output_list:
            np_output.append(output.numpy())

        logits, value, self.lstm_cell, self.lstm_hidden = np_output[:4]

        prob, action, d_action = self._sample_masked_action(logits, legal_action)

        return prob, value, action, d_action  # prob: [[ ]], others: all 1D

    # get final executable actions
    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []
        """
        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [
            sum(self.label_size_list[: index + 1])
            for index in range(len(self.label_size_list))
        ]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits[0], label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],  # [12, 8]
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]  # [12]
        one_hot_actions = np.reshape(
            one_hot_actions, [self.label_size_list[0], 1]
        )  # [12, 1]
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action  # [12]
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        # target_legal_action = tf.gather(target_legal_action, action_idx, axis=1)
        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        # legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        # prob_list.append(probs)
        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        # Not necessary max clip 1
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        # tmp = tf.exp(tmp - tmp_max)* legal_action + _lsm_const_e
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        # tmp_sum = tf.reduce_sum(tmp, axis=1, keepdims=True)
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        """
        Sample with probability, input probs should be 1D array
        """
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))

    def close(self):
        if self.dataset is not None:
            self.save_h5_sample = True
            self.dataset.close()

    def set_lstm_info(self, lstm_info):
        self.lstm_hidden, self.lstm_cell = lstm_info

    def get_lstm_info(self):
        return (self.lstm_hidden, self.lstm_cell)
