# -*- coding: utf-8 -*-
"""
    KingHonour Data production process
"""
import os
import random
import time
import traceback

import numpy as np
from rl_framework.common.logging import log_time_func, g_log_time
from rl_framework.common.logging import logger as LOG
from hok.common.gamecore_client import SimulatorType
from hok.hok1v1.lib.interface import ACTOR_CRYSTAL, ACTOR_TOWER, ACTOR_TOWER_HIGH


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
OS_ENV = os.environ
IS_DEV = OS_ENV.get("IS_DEV")


def _nonnegative_env_float(name, default):
    """Read a non-negative pacing value without making actor startup fragile."""
    raw_value = os.getenv(name, str(default))
    try:
        value = float(raw_value)
    except ValueError:
        LOG.warning("Invalid {}={!r}; use {}", name, raw_value, default)
        return default
    return max(value, 0.0)


def _nonnegative_env_int(name, default):
    """Read a non-negative integer environment setting safely."""
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        LOG.warning("Invalid {}={!r}; use {}", name, raw_value, default)
        return default
    return max(value, 0)


class GamecoreConcurrencyGate:
    """Filesystem semaphore shared by the actor processes in one container."""

    def __init__(self, actor_id, max_games):
        self.actor_id = actor_id
        self.max_games = max_games
        self.gate_dir = os.getenv(
            "GAMECORE_CONCURRENCY_DIR", "/tmp/aiarena_gamecore_slots"
        )
        self.slot_path = None
        if self.max_games:
            os.makedirs(self.gate_dir, exist_ok=True)

    @property
    def enabled(self):
        return self.max_games > 0

    @staticmethod
    def _owner_is_dead(slot_path):
        owner_path = os.path.join(slot_path, "owner")
        try:
            with open(owner_path) as owner_file:
                first_line = owner_file.readline().strip()
            owner_pid = int(first_line.split("=", 1)[1])
        except (FileNotFoundError, IndexError, ValueError):
            # A process can be interrupted after mkdir but before writing its
            # owner file. Leave a short grace period before reclaiming it.
            try:
                return time.time() - os.path.getmtime(slot_path) > 5
            except FileNotFoundError:
                return False

        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    @staticmethod
    def _remove_slot(slot_path):
        try:
            os.remove(os.path.join(slot_path, "owner"))
        except FileNotFoundError:
            pass
        try:
            os.rmdir(slot_path)
        except FileNotFoundError:
            pass
        except OSError:
            # Another waiting actor may have reclaimed the stale slot first.
            pass

    def acquire(self):
        if not self.enabled:
            return

        while True:
            for slot_id in range(self.max_games):
                slot_path = os.path.join(self.gate_dir, "slot_{}".format(slot_id))
                try:
                    os.mkdir(slot_path)
                except FileExistsError:
                    if self._owner_is_dead(slot_path):
                        self._remove_slot(slot_path)
                    continue

                with open(os.path.join(slot_path, "owner"), "w") as owner_file:
                    owner_file.write("pid={}\nactor_id={}\n".format(os.getpid(), self.actor_id))
                self.slot_path = slot_path
                LOG.info(
                    "acquired GameCore slot {}/{}", slot_id + 1, self.max_games
                )
                return
            time.sleep(0.25)

    def release(self):
        if self.slot_path is None:
            return
        self._remove_slot(self.slot_path)
        self.slot_path = None


class Actor:
    """
    used for sample logic
        run 1 episode
        save sample in sample manager
    """

    # def __init__(self, id, type):
    def __init__(
        self,
        id,
        agents,
        max_episode: int = 0,
        env=None,
        monitor_logger=None,
        camp_iter=None,
        is_train=True,
        enemy_type="network",
        common_ai_mix_ratio=0.0,
        balance_training_camps=True,
    ):
        self.m_config_id = id
        self.m_task_uuid = "TODO TASK_UUID"
        self.env = env
        self._max_episode = max_episode
        self._episode_num = 0
        self.agents = agents
        self.monitor_logger = monitor_logger
        self.camp_iter = camp_iter
        self.is_train = is_train
        self.enemy_type = enemy_type
        self.common_ai_mix_ratio = common_ai_mix_ratio
        self.balance_training_camps = balance_training_camps
        self._training_model_slot_counts = {
            "common_ai": [0, 0],
            "self_play": [0, 0],
        }
        self._training_model_actual_camp_counts = {
            "common_ai": {},
            "self_play": {},
        }
        self._model_slot_to_camp = {}
        self._active_training_episode_type = None
        self._active_training_model_slot = None

    def upload_monitor_data(self, data: dict):
        if self.monitor_logger:
            self.monitor_logger.info(data)

    def set_env(self, environment):
        self.env = environment

    def set_sample_manager(self, sample_manager):
        self.m_sample_manager = sample_manager

    def set_agents(self, agents):
        self.agents = agents

    def _select_training_episode_type(self):
        """Choose the opponent class before assigning the latest-policy camp."""
        if self.enemy_type == "common_ai":
            return "common_ai"
        if (
            self.enemy_type == "network"
            and self.common_ai_mix_ratio > 0
            and random.random() < self.common_ai_mix_ratio
        ):
            return "common_ai"
        return "self_play"

    def _prepare_training_camp_assignment(self):
        """Assign the latest policy to the less-used camp for this game type.

        Common-AI and network self-play are balanced independently: otherwise
        a random Common-AI subset could still leave the learned policy mostly
        on one side of the map.  The actor id resolves ties, so an even number
        of actors starts with an even 50:50 split as well.
        """
        episode_type = self._select_training_episode_type()
        self._active_training_episode_type = episode_type
        self._active_training_model_slot = None
        if not self.balance_training_camps:
            return episode_type

        slot_counts = self._training_model_slot_counts[episode_type]
        actual_counts = self._training_model_actual_camp_counts[episode_type]
        camp0 = self._model_slot_to_camp.get(0)
        camp1 = self._model_slot_to_camp.get(1)
        if camp0 is not None and camp1 is not None:
            camp0_count = actual_counts.get(str(camp0), 0)
            camp1_count = actual_counts.get(str(camp1), 0)
        else:
            camp0_count, camp1_count = slot_counts

        if camp0_count == camp1_count:
            model_slot = (self.m_config_id + (episode_type == "self_play")) % 2
        else:
            model_slot = 0 if camp0_count < camp1_count else 1

        for slot, agent in enumerate(self.agents):
            agent.keep_latest = slot == model_slot

        slot_counts[model_slot] += 1
        self._active_training_model_slot = model_slot
        return episode_type

    def _get_common_ai(self, eval, load_models, training_episode_type=None):
        use_common_ai = [False] * len(self.agents)
        if not eval and training_episode_type is not None:
            if training_episode_type == "common_ai":
                for i, agent in enumerate(self.agents):
                    use_common_ai[i] = not agent.keep_latest
            return use_common_ai

        for i, agent in enumerate(self.agents):
            if not eval and not agent.keep_latest:
                if self.enemy_type == "common_ai":
                    use_common_ai[i] = True
                elif (
                    self.enemy_type == "network"
                    and self.common_ai_mix_ratio > 0
                    and random.random() < self.common_ai_mix_ratio
                ):
                    use_common_ai[i] = True
            elif eval:
                if load_models is None or len(load_models) < 2:
                    if not agent.keep_latest:
                        use_common_ai[i] = True
                elif load_models[i] is None:
                    use_common_ai[i] = True

        return use_common_ai

    def _reload_agents(self, eval=False, load_models=None, use_common_ai=None):
        for i, agent in enumerate(self.agents):
            LOG.debug("reset agent {}".format(i))
            if eval:
                if not load_models:
                    agent.reset("common_ai")
                elif len(load_models) == 1:
                    if agent.keep_latest:
                        agent.reset("network", model_path=load_models[0])
                    else:
                        agent.reset("common_ai")
                else:
                    if load_models[i] is None:
                        agent.reset("common_ai")
                    else:
                        agent.reset("network", model_path=load_models[i])
            else:
                # The scripted side does not use this local predictor.  Do
                # not needlessly fetch a historical network model for it.
                if use_common_ai and use_common_ai[i]:
                    continue
                if len(load_models) == 1 and not agent.keep_latest:
                    agent.reset(self.enemy_type, model_path=load_models[0])
                else:
                    agent.reset(self.enemy_type)

    def _save_last_sample(self, done, eval, sample_manager, state_dict):
        final_rewards = [0.0] * len(self.agents)
        if done:
            for i, agent in enumerate(self.agents):
                # GameCore does not expose a Python state for its scripted
                # common-AI side.  A mixed self-play episode can therefore
                # end with state_dict[i] == None.  That side has no sample to
                # finalize, regardless of which model happened to be marked
                # latest for the actor.
                if agent.is_latest_model and state_dict[i] is not None:
                    final_rewards[i] = agent.get_shaped_reward(state_dict[i])
                    if not eval:
                        sample_manager.save_last_sample(
                            agent_id=i, reward=final_rewards[i]
                        )
        return final_rewards

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

    @staticmethod
    def _init_behavior_stats():
        return {
            "frames": 0,
            "hp_rate_sum": 0.0,
            "ep_rate_sum": 0.0,
            "ep_rate_count": 0,
            "low_resource_frames": 0,
            "low_resource_near_home_frames": 0,
            "low_resource_far_home_frames": 0,
            "hp_recovery_home_frames": 0,
            "ep_recovery_home_frames": 0,
            # Recall is action button 9 in the 1v1 SDK.  These are
            # observation-only diagnostics: do not reward or force Recall
            # until the runtime action mask proves it is actually available.
            "recall_legal_frames": 0,
            "recall_low_resource_legal_frames": 0,
            "recall_safe_legal_frames": 0,
            "recall_selected_frames": 0,
            "recall_selected_legal_frames": 0,
            "last_hp": None,
            "last_ep": None,
            "last_total_hurt": None,
            "last_enemy_objective_hp": None,
            "skill_cast_frames": 0,
            "skill_near_target_frames": 0,
            "skill_no_target_frames": 0,
            "skill_near_hero_frames": 0,
            "skill_near_lane_frames": 0,
            "skill_near_objective_frames": 0,
            "skill_followup_success_frames": 0,
            "skill_pending_followup": 0,
            "skill_pending_resolved": False,
            "healthy_frames": 0,
            "near_own_objective_frames": 0,
            "camp_own_objective_frames": 0,
            "objective_pressure_frames": 0,
            "near_lane_frames": 0,
            "near_enemy_objective_frames": 0,
            "own_objective_distance_sum": 0.0,
            "lane_distance_sum": 0.0,
            "enemy_objective_distance_sum": 0.0,
            "own_objective_distance_count": 0,
            "lane_distance_count": 0,
            "enemy_objective_distance_count": 0,
        }

    def _update_behavior_stats(
        self, stats, agent, req_pb, action=None, action_state=None
    ):
        hero_state = None
        enemy_hero_states = []
        for hero in req_pb.hero_list:
            if hero.runtime_id == agent.player_id:
                hero_state = hero
            elif hero.hp > 0:
                enemy_hero_states.append(hero)

        if hero_state is None or hero_state.hp <= 0:
            return

        stats["frames"] += 1
        max_hp = max(hero_state.max_hp, 1)
        hp_rate = hero_state.hp / max_hp
        ep = self._get_first_attr(hero_state, ("ep", "ep_point", "energy", "mp"), None)
        ep_rate = self._get_ep_rate(hero_state)
        hero_loc = hero_state.location
        objective_types = (ACTOR_TOWER, ACTOR_TOWER_HIGH, ACTOR_CRYSTAL)
        own_crystal_state = None
        for organ in req_pb.organ_list:
            if (
                organ.camp == agent.hero_camp
                and organ.type == ACTOR_CRYSTAL
                and organ.hp > 0
            ):
                own_crystal_state = organ
                break

        stats["hp_rate_sum"] += hp_rate
        if ep_rate is not None:
            stats["ep_rate_sum"] += ep_rate
            stats["ep_rate_count"] += 1

        low_hp = hp_rate < agent.config.RETREAT_HP_RATE
        low_ep = ep_rate is not None and ep_rate < agent.config.RETREAT_EP_RATE
        low_resource = low_hp or low_ep
        recall_legal = False
        if action_state is not None:
            legal_action = action_state.get("legal_action")
            if legal_action is not None and len(legal_action) > 9:
                recall_legal = bool(legal_action[9])

        recall_selected = (
            action is not None and len(action) > 0 and int(action[0]) == 9
        )
        if recall_legal:
            stats["recall_legal_frames"] += 1
        if recall_selected:
            stats["recall_selected_frames"] += 1
            if recall_legal:
                stats["recall_selected_legal_frames"] += 1

        near_home = False
        if own_crystal_state is not None:
            home_distance = self._distance(hero_loc, own_crystal_state.location)
            near_home = home_distance < agent.config.RESOURCE_RECOVERY_HOME_RADIUS
            if low_resource:
                stats["low_resource_frames"] += 1
                if near_home:
                    stats["low_resource_near_home_frames"] += 1
                else:
                    stats["low_resource_far_home_frames"] += 1

            if near_home and stats["last_hp"] is not None and hero_state.hp > stats["last_hp"]:
                stats["hp_recovery_home_frames"] += 1
            if (
                near_home
                and ep is not None
                and stats["last_ep"] is not None
                and ep > stats["last_ep"]
            ):
                stats["ep_recovery_home_frames"] += 1

        if recall_legal and low_resource and not near_home:
            stats["recall_low_resource_legal_frames"] += 1
            enemy_nearby = any(
                self._distance(hero_loc, enemy.location)
                < agent.config.RETREAT_DANGER_MAX_RADIUS
                for enemy in enemy_hero_states
            )
            if not enemy_nearby:
                stats["recall_safe_legal_frames"] += 1

        stats["last_hp"] = hero_state.hp
        stats["last_ep"] = ep

        if hp_rate < agent.config.OBJECTIVE_DISTANCE_HEALTHY_HP_RATE:
            return

        stats["healthy_frames"] += 1
        own_objectives = []
        enemy_objectives = []
        enemy_objective_hp = 0
        for organ in req_pb.organ_list:
            if organ.type not in objective_types or organ.hp <= 0:
                continue
            if organ.camp == agent.hero_camp:
                own_objectives.append(organ)
            else:
                enemy_objectives.append(organ)
                enemy_objective_hp += max(organ.hp, 0)

        enemy_soldiers = [
            soldier
            for soldier in req_pb.soldier_list
            if soldier.camp != agent.hero_camp and soldier.hp > 0
        ]

        if own_objectives:
            own_distance = min(
                self._distance(hero_loc, organ.location) for organ in own_objectives
            )
            stats["own_objective_distance_sum"] += own_distance
            stats["own_objective_distance_count"] += 1
            if own_distance < agent.config.OWN_OBJECTIVE_CAMP_RADIUS:
                stats["near_own_objective_frames"] += 1

                objective_under_pressure = False
                pressure_units = enemy_soldiers + enemy_hero_states
                for unit in pressure_units:
                    for organ in own_objectives:
                        if (
                            self._distance(unit.location, organ.location)
                            < agent.config.OWN_OBJECTIVE_DEFENSE_RADIUS
                        ):
                            objective_under_pressure = True
                            break
                    if objective_under_pressure:
                        break

                if objective_under_pressure:
                    stats["objective_pressure_frames"] += 1
                else:
                    stats["camp_own_objective_frames"] += 1

        if enemy_soldiers:
            lane_distance = min(
                self._distance(hero_loc, soldier.location) for soldier in enemy_soldiers
            )
            stats["lane_distance_sum"] += lane_distance
            stats["lane_distance_count"] += 1
            if lane_distance < agent.config.LANE_DISTANCE_MAX_RADIUS:
                stats["near_lane_frames"] += 1

        if enemy_objectives:
            enemy_objective_distance = min(
                self._distance(hero_loc, organ.location) for organ in enemy_objectives
            )
            stats["enemy_objective_distance_sum"] += enemy_objective_distance
            stats["enemy_objective_distance_count"] += 1
            if enemy_objective_distance < agent.config.OBJECTIVE_DISTANCE_MAX_RADIUS:
                stats["near_enemy_objective_frames"] += 1

        total_hurt_delta = 0
        if stats["last_total_hurt"] is not None:
            total_hurt_delta = max(hero_state.totalHurt - stats["last_total_hurt"], 0)
        objective_hurt_delta = 0
        if stats["last_enemy_objective_hp"] is not None:
            objective_hurt_delta = max(
                stats["last_enemy_objective_hp"] - enemy_objective_hp, 0
            )

        if stats["skill_pending_followup"] > 0:
            if total_hurt_delta > 0 or objective_hurt_delta > 0:
                if not stats["skill_pending_resolved"]:
                    stats["skill_followup_success_frames"] += 1
                stats["skill_pending_resolved"] = True
                stats["skill_pending_followup"] = 0
            else:
                stats["skill_pending_followup"] -= 1
                if stats["skill_pending_followup"] == 0:
                    stats["skill_pending_resolved"] = False

        skill_buttons = (4, 5, 6, 8, 10, 11)
        button = int(action[0]) if action is not None and len(action) > 0 else -1
        if button in skill_buttons:
            stats["skill_cast_frames"] += 1
            target_radius = max(agent.config.SKILL_MONITOR_TARGET_RADIUS, 1.0)
            near_hero = any(
                self._distance(hero_loc, hero.location) < target_radius
                for hero in enemy_hero_states
            )
            near_lane = any(
                self._distance(hero_loc, soldier.location) < target_radius
                for soldier in enemy_soldiers
            )
            near_objective = any(
                self._distance(hero_loc, organ.location) < target_radius
                for organ in enemy_objectives
            )
            if near_hero:
                stats["skill_near_hero_frames"] += 1
            if near_lane:
                stats["skill_near_lane_frames"] += 1
            if near_objective:
                stats["skill_near_objective_frames"] += 1
            if near_hero or near_lane or near_objective:
                stats["skill_near_target_frames"] += 1
            else:
                stats["skill_no_target_frames"] += 1

            stats["skill_pending_followup"] = max(
                agent.config.SKILL_MONITOR_FOLLOWUP_FRAMES, 1
            )
            stats["skill_pending_resolved"] = False

        stats["last_total_hurt"] = hero_state.totalHurt
        stats["last_enemy_objective_hp"] = enemy_objective_hp

    def _run_episode(
        self, camp_config, eval=False, load_models=None, gamecore_eval=None
    ):
        LOG.info("Start a new game")
        for item in g_log_time.items():
            g_log_time[item[0]] = []
        sample_manager = self.m_sample_manager
        done = False
        log_time_func("reset")
        log_time_func("one_episode")
        LOG.debug("reset env")
        LOG.info(camp_config)
        training_episode_type = None
        if not eval:
            training_episode_type = self._prepare_training_camp_assignment()
        use_common_ai = self._get_common_ai(
            eval, load_models, training_episode_type=training_episode_type
        )
        if not eval and self.enemy_type == "network":
            LOG.info(
                "opponent selection: type={}, common_ai={}, mix_ratio={:.2f}, latest_model_slot={}, common_slot_counts={}, selfplay_slot_counts={}",
                training_episode_type,
                use_common_ai,
                self.common_ai_mix_ratio,
                self._active_training_model_slot,
                self._training_model_slot_counts["common_ai"],
                self._training_model_slot_counts["self_play"],
            )

        # ATTENTION: agent.reset() loads models from local file which cost a lot of time.
        #            Before upload your code, please check your code to avoid ANY time-wasting
        #            operations between env.reset() and env.close_game(). Any TIMEOUT in a round
        #            of game will cause undefined errors.

        # reload agent models
        self._reload_agents(eval, load_models, use_common_ai)
        # restart a new game
        # reward :[dead,ep_rate,exp,hp_point,kill,last_hit,money,tower_hp_point,reward_sum]
        # ``eval`` is an SDK processing mode, not merely a GameCore simulator
        # selector.  The official double-network test runs with eval=False;
        # setting it True here changes Interface.Reset() and can make a normal
        # training game fail after both callbacks have connected.  Request a
        # fresh remote simulator explicitly for two network-controlled sides,
        # while preserving training mode and the established RemoteRepeat
        # path for common-AI games.
        remote_network_selfplay = (
            not eval and self.enemy_type == "network" and not any(use_common_ai)
        )
        simulator_type = SimulatorType.Remote if remote_network_selfplay else None
        if remote_network_selfplay:
            LOG.info("network self-play: training mode with simulator=remote")
        _, r, d, state_dict = self.env.reset(
            camp_config,
            use_common_ai=use_common_ai,
            eval=eval,
            simulator_type=simulator_type,
        )
        if state_dict[0] is None:
            game_id = state_dict[1]["game_id"]
        else:
            game_id = state_dict[0]["game_id"]

        # update agents' game information
        for i, agent in enumerate(self.agents):
            player_id = self.env.player_list[i]
            camp = self.env.player_camp.get(player_id)
            agent.set_game_info(camp, player_id)
            self._model_slot_to_camp[i] = camp

        if not eval and self._active_training_model_slot is not None:
            latest_agent = self.agents[self._active_training_model_slot]
            actual_counts = self._training_model_actual_camp_counts[
                training_episode_type
            ]
            latest_camp = str(latest_agent.hero_camp)
            actual_counts[latest_camp] = actual_counts.get(latest_camp, 0) + 1
            LOG.info(
                "training camp assignment: type={}, latest_model_slot={}, latest_model_camp={}, common_actual_camp_counts={}, selfplay_actual_camp_counts={}",
                training_episode_type,
                self._active_training_model_slot,
                latest_agent.hero_camp,
                self._training_model_actual_camp_counts["common_ai"],
                self._training_model_actual_camp_counts["self_play"],
            )

        # reset mem pool and models
        LOG.debug("reset sample_manager")
        sample_manager.reset(agents=self.agents, game_id=game_id)
        rewards = [[], []]
        step = 0
        log_time_func("reset", end=True)
        game_info = {}
        episode_infos = [{"h_act_num": 0} for _ in self.agents]
        behavior_infos = [self._init_behavior_stats() for _ in self.agents]

        while not done:
            log_time_func("one_frame")
            # while True:
            actions = []
            agent_actions = [None] * len(self.agents)
            agent_states = [None] * len(self.agents)
            log_time_func("agent_process")
            for i, agent in enumerate(self.agents):
                if use_common_ai[i]:
                    actions.append(None)
                    rewards[i].append(0.0)
                    continue
                # print("agent{}".format(i),state_dict[i]['observation'])
                action, d_action, sample = agent.process(state_dict[i])
                if eval:
                    action = d_action
                # print("input act: [{}], {}, {}".format(i, action, state_dict[i]["legal_action"][:12]))
                actions.append(action)
                agent_actions[i] = action
                agent_states[i] = state_dict[i]
                if action[0] == 10:
                    episode_infos[i]["h_act_num"] += 1
                rewards[i].append(sample["reward"])

                if agent.is_latest_model and not eval:
                    sample_manager.save_sample(
                        **sample, agent_id=i, game_id=game_id, uuid=self.m_task_uuid
                    )
            log_time_func("agent_process", end=True)

            log_time_func("step")
            # reward :[dead,ep_rate,exp,hp_point,kill,last_hit,money,tower_hp_point,reward_sum]
            _, r, d, state_dict = self.env.step(actions)
            # if np.isnan(r[0][-1]) or np.isnan(r[1][-1]):
            #     exit(0)
            log_time_func("step", end=True)

            req_pbs = self.env.cur_req_pb
            if req_pbs[0] is None:
                req_pb = req_pbs[1]
            else:
                req_pb = req_pbs[0]
            for i, agent in enumerate(self.agents):
                if use_common_ai[i]:
                    continue
                agent_req_pb = req_pbs[i] if req_pbs[i] is not None else req_pb
                self._update_behavior_stats(
                    behavior_infos[i],
                    agent,
                    agent_req_pb,
                    action=agent_actions[i],
                    action_state=agent_states[i],
                )
            LOG.debug(
                "step: {}, frame_no: {}, reward: {}, {}".format(
                    step, req_pb.frame_no, r[0], r[1]
                )
            )
            step += 1
            done = d[0] or d[1]

            final_rewards = self._save_last_sample(
                done, eval, sample_manager, state_dict
            )
            if done:
                for i, reward in enumerate(final_rewards):
                    if self.agents[i].is_latest_model:
                        rewards[i].append(reward)
            log_time_func("one_frame", end=True)

        self.env.close_game()

        game_info["length"] = req_pb.frame_no
        loss_camp = -1
        camp_hp = {}
        objective_hp = {}
        all_camp_list = []
        objective_types = (ACTOR_TOWER, ACTOR_TOWER_HIGH, ACTOR_CRYSTAL)
        for organ in req_pb.organ_list:
            if organ.type in objective_types and organ.hp > 0:
                objective_hp[organ.camp] = objective_hp.get(organ.camp, 0) + max(
                    organ.hp, 0
                )
            if organ.type == ACTOR_CRYSTAL:
                if organ.hp <= 0:
                    loss_camp = organ.camp
                camp_hp[organ.camp] = organ.hp
                all_camp_list.append(organ.camp)
            if organ.type in objective_types:
                LOG.info(
                    "Tower {} in camp {}, hp: {}".format(
                        organ.type, organ.camp, organ.hp
                    )
                )

        for i, agent in enumerate(self.agents):
            if use_common_ai[i]:
                continue
            for hero_state in req_pbs[i].hero_list:
                if agent.player_id == hero_state.runtime_id:
                    episode_infos[i]["money_per_frame"] = (
                        hero_state.moneyCnt / game_info["length"]
                    )
                    episode_infos[i]["kill"] = hero_state.killCnt
                    episode_infos[i]["death"] = hero_state.deadCnt
                    episode_infos[i]["assistCnt"] = hero_state.assistCnt
                    episode_infos[i]["hurt_per_frame"] = (
                        hero_state.totalHurt / game_info["length"]
                    )
                    episode_infos[i]["hurtH_per_frame"] = (
                        hero_state.totalHurtToHero / game_info["length"]
                    )
                    episode_infos[i]["hurtBH_per_frame"] = (
                        hero_state.totalBeHurtByHero / game_info["length"]
                    )
                    episode_infos[i]["hp_rate"] = hero_state.hp / max(
                        hero_state.max_hp, 1
                    )
                    episode_infos[i]["heroes"] = camp_config["heroes"][i]
                    episode_infos[i]["totalHurt"] = hero_state.totalHurt
                    episode_infos[i]["totalHurtToHero"] = hero_state.totalHurtToHero
                    episode_infos[i]["totalBeHurtByHero"] = (
                        hero_state.totalBeHurtByHero
                    )
                    break
            if loss_camp == -1:
                episode_infos[i]["win"] = 0
            else:
                episode_infos[i]["win"] = -1 if agent.hero_camp == loss_camp else 1

            episode_infos[i]["reward"] = np.sum(rewards[i])
            episode_infos[i].update(agent.get_reward_component_summary())
            episode_infos[i]["h_act_rate"] = episode_infos[i]["h_act_num"] / step
            enemy_camps = [camp for camp in all_camp_list if camp != agent.hero_camp]
            enemy_camp = enemy_camps[0] if enemy_camps else None
            own_objective_hp = objective_hp.get(agent.hero_camp, 0)
            enemy_objective_hp = (
                objective_hp.get(enemy_camp, 0) if enemy_camp is not None else 0
            )
            episode_infos[i]["own_objective_hp"] = own_objective_hp
            episode_infos[i]["enemy_objective_hp"] = enemy_objective_hp
            episode_infos[i]["objective_hp_diff"] = (
                own_objective_hp - enemy_objective_hp
            )
            episode_infos[i]["trade_ratio"] = episode_infos[i][
                "totalHurtToHero"
            ] / max(episode_infos[i]["totalBeHurtByHero"], 1)

        if self.is_train and not eval:
            LOG.debug("send sample_manager")
            sample_manager.send_samples()
            LOG.debug("send done.")

        log_time_func("one_episode", end=True)
        # print game information
        self._print_info(
            game_id,
            game_info,
            episode_infos,
            behavior_infos,
            eval,
            common_ai=use_common_ai,
        )

    def _print_info(
        self, game_id, game_info, episode_infos, behavior_infos, eval, common_ai=None
    ):
        if common_ai is None:
            common_ai = [False] * len(self.agents)
        LOG.info("=" * 50)
        LOG.info("game_id : %s" % game_id)
        for item in g_log_time.items():
            if len(item) <= 1 or len(item[1]) == 0 or len(item[0]) == 0:
                continue
            time_mean = np.mean(item[1])
            time_max = np.max(item[1])
            time_sum = np.sum(item[1])
            LOG.info(
                "%s | sum: %s mean:%s max:%s times:%s"
                % (item[0], time_sum, time_mean, time_max, len(item[1]))
            )
            g_log_time[item[0]] = []
        LOG.info("=" * 50)
        for i, agent in enumerate(self.agents):
            if common_ai[i]:
                continue
            LOG.info(
                "Agent is_main:{}, type:{}, camp:{},reward:{:.3f}, win:{}, h_act_rate:{}, heroes:{}".format(
                    agent.keep_latest and eval,
                    agent.agent_type,
                    agent.hero_camp,
                    episode_infos[i]["reward"],
                    episode_infos[i]["win"],
                    episode_infos[i]["h_act_rate"],
                    episode_infos[i]["heroes"],
                )
            )
            LOG.info(
                "Agent is_main:{}, money_per_frame:{:.2f}, kill:{}, death:{}, hurt_pf:{:.2f}".format(
                    agent.keep_latest and eval,
                    episode_infos[i]["money_per_frame"],
                    episode_infos[i]["kill"],
                    episode_infos[i]["death"],
                    episode_infos[i]["hurt_per_frame"],
                )
            )
            LOG.info(
                "Agent is_main:{}, reward_components: base:{:.3f}, position:{:.3f}, trade:{:.3f}, trade_raw:{:.3f}, trade_clip_steps:{}, economy:{:.3f}, economy_raw:{:.3f}, money_gain:{:.1f}, enemy_money_gain:{:.1f}, objective_progress:{:.3f}, objective_raw:{:.3f}, enemy_objective_damage:{:.1f}, own_objective_damage:{:.1f}, objective_clip_steps:{}, samples:{}".format(
                    agent.keep_latest and eval,
                    episode_infos[i]["base_reward"],
                    episode_infos[i]["position_shaping"],
                    episode_infos[i]["trade_shaping"],
                    episode_infos[i]["trade_shaping_raw"],
                    episode_infos[i]["trade_clip_steps"],
                    episode_infos[i]["economy_shaping"],
                    episode_infos[i]["economy_shaping_raw"],
                    episode_infos[i]["economy_money_gain"],
                    episode_infos[i]["economy_enemy_money_gain"],
                    episode_infos[i]["objective_shaping"],
                    episode_infos[i]["objective_shaping_raw"],
                    episode_infos[i]["objective_enemy_damage"],
                    episode_infos[i]["objective_own_damage"],
                    episode_infos[i]["objective_clip_steps"],
                    episode_infos[i]["reward_samples"],
                )
            )
            LOG.info(
                "Agent is_main:{}, retreat_shaping: low_resource_position:{:.4f}, low_resource_samples:{}, far_home_position:{:.4f}, far_home_samples:{}, recovery:{:.4f}, recovery_events:{}, danger:{:.4f}, danger_samples:{}, progress:{:.4f}, progress_events:{}, regression:{:.4f}, regression_events:{}".format(
                    agent.keep_latest and eval,
                    episode_infos[i]["low_resource_position_shaping"],
                    episode_infos[i]["low_resource_position_samples"],
                    episode_infos[i]["low_resource_far_home_position_shaping"],
                    episode_infos[i]["low_resource_far_home_position_samples"],
                    episode_infos[i]["retreat_recovery_shaping"],
                    episode_infos[i]["retreat_recovery_events"],
                    episode_infos[i]["retreat_danger_shaping"],
                    episode_infos[i]["retreat_danger_samples"],
                    episode_infos[i]["retreat_progress_shaping"],
                    episode_infos[i]["retreat_progress_events"],
                    episode_infos[i]["retreat_regression_shaping"],
                    episode_infos[i]["retreat_regression_events"],
                )
            )
            LOG.info(
                "Agent is_main:{}, retreat_cycle: total:{:.4f}, started:{}, progress:{:.4f}, progress_events:{}, safe_approach:{:.4f}, safe_approach_events:{}, home_approach:{:.4f}, home_approach_events:{}, home:{:.4f}, home_events:{}, home_recovery:{:.4f}, home_recovery_events:{}, home_ready:{:.4f}, home_ready_events:{}, safe_completion:{:.4f}, safe_completion_events:{}, completion:{:.4f}, completion_events:{}, rejoin:{:.4f}, rejoin_events:{}, danger_damage:{:.4f}, danger_damage_events:{}, danger_reward_block_events:{}, death:{:.4f}, death_events:{}, timeout_events:{}".format(
                    agent.keep_latest and eval,
                    episode_infos[i]["retreat_cycle_shaping"],
                    episode_infos[i]["retreat_cycle_started"],
                    episode_infos[i]["retreat_cycle_progress_shaping"],
                    episode_infos[i]["retreat_cycle_progress_events"],
                    episode_infos[i]["retreat_cycle_safe_approach_shaping"],
                    episode_infos[i]["retreat_cycle_safe_approach_events"],
                    episode_infos[i]["retreat_cycle_home_approach_shaping"],
                    episode_infos[i]["retreat_cycle_home_approach_events"],
                    episode_infos[i]["retreat_cycle_home_shaping"],
                    episode_infos[i]["retreat_cycle_home_events"],
                    episode_infos[i]["retreat_cycle_home_recovery_shaping"],
                    episode_infos[i]["retreat_cycle_home_recovery_events"],
                    episode_infos[i]["retreat_cycle_home_ready_shaping"],
                    episode_infos[i]["retreat_cycle_home_ready_events"],
                    episode_infos[i]["retreat_cycle_safe_completion_shaping"],
                    episode_infos[i]["retreat_cycle_safe_completion_events"],
                    episode_infos[i]["retreat_cycle_completion_shaping"],
                    episode_infos[i]["retreat_cycle_completion_events"],
                    episode_infos[i]["retreat_cycle_rejoin_shaping"],
                    episode_infos[i]["retreat_cycle_rejoin_events"],
                    episode_infos[i]["retreat_cycle_danger_damage_shaping"],
                    episode_infos[i]["retreat_cycle_danger_damage_events"],
                    episode_infos[i]["retreat_cycle_danger_reward_block_events"],
                    episode_infos[i]["retreat_cycle_death_shaping"],
                    episode_infos[i]["retreat_cycle_death_events"],
                    episode_infos[i]["retreat_cycle_timeout_events"],
                )
            )
            LOG.info(
                "Agent is_main:{}, retreat_decision: forced_samples:{}, fight_allowed_samples:{}, escape:{:.4f}, escape_events:{}, safe:{:.4f}, safe_events:{}, danger_hold:{:.4f}, danger_hold_events:{}, stall:{:.4f}, stall_events:{}, timeout:{:.4f}".format(
                    agent.keep_latest and eval,
                    episode_infos[i]["retreat_cycle_forced_retreat_samples"],
                    episode_infos[i]["retreat_cycle_fight_allowed_samples"],
                    episode_infos[i]["retreat_cycle_escape_shaping"],
                    episode_infos[i]["retreat_cycle_escape_events"],
                    episode_infos[i]["retreat_cycle_safe_shaping"],
                    episode_infos[i]["retreat_cycle_safe_events"],
                    episode_infos[i]["retreat_cycle_danger_hold_shaping"],
                    episode_infos[i]["retreat_cycle_danger_hold_events"],
                    episode_infos[i]["retreat_cycle_stall_shaping"],
                    episode_infos[i]["retreat_cycle_stall_events"],
                    episode_infos[i]["retreat_cycle_timeout_shaping"],
                )
            )
            LOG.info(
                "Agent is_main:{}, env_reward_components: dead:{:.3f}, ep:{:.3f}, exp:{:.3f}, hp:{:.3f}, kill:{:.3f}, last_hit:{:.3f}, sdk_money:{:.3f}, objective:{:.3f}".format(
                    agent.keep_latest and eval,
                    episode_infos[i]["env_dead"],
                    episode_infos[i]["env_ep_rate"],
                    episode_infos[i]["env_exp"],
                    episode_infos[i]["env_hp_point"],
                    episode_infos[i]["env_kill"],
                    episode_infos[i]["env_last_hit"],
                    episode_infos[i]["env_money"],
                    episode_infos[i]["env_objective_hp"],
                )
            )
            LOG.info(
                "Agent is_main:{}, detail: hurtH_pf:{:.2f}, hurtBH_pf:{:.2f}, trade_ratio:{:.2f}, hp_rate:{:.2f}, objective_hp_diff:{}, own_objective_hp:{}, enemy_objective_hp:{}, length:{}".format(
                    agent.keep_latest and eval,
                    episode_infos[i]["hurtH_per_frame"],
                    episode_infos[i]["hurtBH_per_frame"],
                    episode_infos[i]["trade_ratio"],
                    episode_infos[i]["hp_rate"],
                    episode_infos[i]["objective_hp_diff"],
                    episode_infos[i]["own_objective_hp"],
                    episode_infos[i]["enemy_objective_hp"],
                    game_info["length"],
                )
            )
            behavior_info = behavior_infos[i]
            total_frames = max(behavior_info["frames"], 1)
            healthy_frames = max(behavior_info["healthy_frames"], 1)
            own_distance_count = max(behavior_info["own_objective_distance_count"], 1)
            lane_distance_count = max(behavior_info["lane_distance_count"], 1)
            enemy_objective_distance_count = max(
                behavior_info["enemy_objective_distance_count"], 1
            )
            behavior_metrics = {
                "near_own_obj_rate": behavior_info["near_own_objective_frames"]
                / healthy_frames,
                "camp_own_obj_rate": behavior_info["camp_own_objective_frames"]
                / healthy_frames,
                "pressure_obj_rate": behavior_info["objective_pressure_frames"]
                / healthy_frames,
                "near_lane_rate": behavior_info["near_lane_frames"] / healthy_frames,
                "near_enemy_obj_rate": behavior_info["near_enemy_objective_frames"]
                / healthy_frames,
                "avg_own_obj_dist": behavior_info["own_objective_distance_sum"]
                / own_distance_count,
                "avg_lane_dist": behavior_info["lane_distance_sum"]
                / lane_distance_count,
                "avg_enemy_obj_dist": behavior_info["enemy_objective_distance_sum"]
                / enemy_objective_distance_count,
            }
            ep_rate_count = max(behavior_info["ep_rate_count"], 1)
            low_resource_frames = max(behavior_info["low_resource_frames"], 1)
            resource_metrics = {
                "avg_hp_rate": behavior_info["hp_rate_sum"] / total_frames,
                "avg_ep_rate": behavior_info["ep_rate_sum"] / ep_rate_count,
                "ep_observed_rate": behavior_info["ep_rate_count"] / total_frames,
                "low_resource_rate": behavior_info["low_resource_frames"]
                / total_frames,
                "low_resource_near_home_rate": behavior_info[
                    "low_resource_near_home_frames"
                ]
                / low_resource_frames,
                "low_resource_far_home_rate": behavior_info[
                    "low_resource_far_home_frames"
                ]
                / low_resource_frames,
                "hp_recovery_home_rate": behavior_info["hp_recovery_home_frames"]
                / total_frames,
                "ep_recovery_home_rate": behavior_info["ep_recovery_home_frames"]
                / total_frames,
            }
            recall_metrics = {
                "legal_rate": behavior_info["recall_legal_frames"] / total_frames,
                "low_resource_legal_rate": behavior_info[
                    "recall_low_resource_legal_frames"
                ]
                / total_frames,
                "safe_legal_rate": behavior_info["recall_safe_legal_frames"]
                / total_frames,
                "selected_rate": behavior_info["recall_selected_frames"]
                / total_frames,
            }
            skill_cast_frames = max(behavior_info["skill_cast_frames"], 1)
            skill_metrics = {
                "skill_cast_rate": behavior_info["skill_cast_frames"] / total_frames,
                "skill_near_target_rate": behavior_info["skill_near_target_frames"]
                / skill_cast_frames,
                "skill_no_target_rate": behavior_info["skill_no_target_frames"]
                / skill_cast_frames,
                "skill_near_hero_rate": behavior_info["skill_near_hero_frames"]
                / skill_cast_frames,
                "skill_near_lane_rate": behavior_info["skill_near_lane_frames"]
                / skill_cast_frames,
                "skill_near_objective_rate": behavior_info[
                    "skill_near_objective_frames"
                ]
                / skill_cast_frames,
                "skill_followup_success_rate": behavior_info[
                    "skill_followup_success_frames"
                ]
                / skill_cast_frames,
            }
            LOG.info(
                "Agent is_main:{}, behavior: healthy_frames:{}, near_own_obj_rate:{:.3f}, camp_own_obj_rate:{:.3f}, pressure_obj_rate:{:.3f}, near_lane_rate:{:.3f}, near_enemy_obj_rate:{:.3f}, avg_own_obj_dist:{:.1f}, avg_lane_dist:{:.1f}, avg_enemy_obj_dist:{:.1f}".format(
                    agent.keep_latest and eval,
                    behavior_info["healthy_frames"],
                    behavior_metrics["near_own_obj_rate"],
                    behavior_metrics["camp_own_obj_rate"],
                    behavior_metrics["pressure_obj_rate"],
                    behavior_metrics["near_lane_rate"],
                    behavior_metrics["near_enemy_obj_rate"],
                    behavior_metrics["avg_own_obj_dist"],
                    behavior_metrics["avg_lane_dist"],
                    behavior_metrics["avg_enemy_obj_dist"],
                )
            )
            LOG.info(
                "Agent is_main:{}, resource: frames:{}, avg_hp_rate:{:.3f}, avg_ep_rate:{:.3f}, ep_observed_rate:{:.3f}, low_resource_rate:{:.3f}, low_resource_near_home_rate:{:.3f}, low_resource_far_home_rate:{:.3f}, hp_recovery_home_rate:{:.3f}, ep_recovery_home_rate:{:.3f}".format(
                    agent.keep_latest and eval,
                    behavior_info["frames"],
                    resource_metrics["avg_hp_rate"],
                    resource_metrics["avg_ep_rate"],
                    resource_metrics["ep_observed_rate"],
                    resource_metrics["low_resource_rate"],
                    resource_metrics["low_resource_near_home_rate"],
                    resource_metrics["low_resource_far_home_rate"],
                    resource_metrics["hp_recovery_home_rate"],
                    resource_metrics["ep_recovery_home_rate"],
                )
            )
            LOG.info(
                "Agent is_main:{}, recall: legal_frames:{}, legal_rate:{:.4f}, low_resource_legal_frames:{}, low_resource_legal_rate:{:.4f}, safe_legal_frames:{}, safe_legal_rate:{:.4f}, selected_frames:{}, selected_legal_frames:{}, selected_rate:{:.4f}".format(
                    agent.keep_latest and eval,
                    behavior_info["recall_legal_frames"],
                    recall_metrics["legal_rate"],
                    behavior_info["recall_low_resource_legal_frames"],
                    recall_metrics["low_resource_legal_rate"],
                    behavior_info["recall_safe_legal_frames"],
                    recall_metrics["safe_legal_rate"],
                    behavior_info["recall_selected_frames"],
                    behavior_info["recall_selected_legal_frames"],
                    recall_metrics["selected_rate"],
                )
            )
            LOG.info(
                "Agent is_main:{}, skill: casts:{}, skill_cast_rate:{:.3f}, skill_near_target_rate:{:.3f}, skill_no_target_rate:{:.3f}, skill_near_hero_rate:{:.3f}, skill_near_lane_rate:{:.3f}, skill_near_objective_rate:{:.3f}, skill_followup_success_rate:{:.3f}".format(
                    agent.keep_latest and eval,
                    behavior_info["skill_cast_frames"],
                    skill_metrics["skill_cast_rate"],
                    skill_metrics["skill_near_target_rate"],
                    skill_metrics["skill_no_target_rate"],
                    skill_metrics["skill_near_hero_rate"],
                    skill_metrics["skill_near_lane_rate"],
                    skill_metrics["skill_near_objective_rate"],
                    skill_metrics["skill_followup_success_rate"],
                )
            )
            if eval and agent.last_model_path:
                LOG.info(
                    "CHECKPOINT_EVAL_RESULT slot:{}, model:{}, camp:{}, win:{}, kill:{}, death:{}, money_pf:{:.4f}, hurtH_pf:{:.4f}, hurtBH_pf:{:.4f}, objective_hp_diff:{}".format(
                        i,
                        agent.last_model_path,
                        agent.hero_camp,
                        episode_infos[i]["win"],
                        episode_infos[i]["kill"],
                        episode_infos[i]["death"],
                        episode_infos[i]["money_per_frame"],
                        episode_infos[i]["hurtH_per_frame"],
                        episode_infos[i]["hurtBH_per_frame"],
                        episode_infos[i]["objective_hp_diff"],
                    )
                )
            if agent.keep_latest and eval:
                self.upload_monitor_data(
                    {
                        "reward": episode_infos[i]["reward"],
                        "reward_base": episode_infos[i]["base_reward"],
                        "reward_position_shaping": episode_infos[i][
                            "position_shaping"
                        ],
                        "reward_retreat_low_resource": episode_infos[i][
                            "low_resource_position_shaping"
                        ],
                        "reward_retreat_low_resource_samples": episode_infos[i][
                            "low_resource_position_samples"
                        ],
                        "reward_retreat_far_home": episode_infos[i][
                            "low_resource_far_home_position_shaping"
                        ],
                        "reward_retreat_far_home_samples": episode_infos[i][
                            "low_resource_far_home_position_samples"
                        ],
                        "reward_retreat_recovery": episode_infos[i][
                            "retreat_recovery_shaping"
                        ],
                        "reward_retreat_recovery_events": episode_infos[i][
                            "retreat_recovery_events"
                        ],
                        "reward_retreat_danger": episode_infos[i][
                            "retreat_danger_shaping"
                        ],
                        "reward_retreat_danger_samples": episode_infos[i][
                            "retreat_danger_samples"
                        ],
                        "reward_retreat_progress": episode_infos[i][
                            "retreat_progress_shaping"
                        ],
                        "reward_retreat_progress_events": episode_infos[i][
                            "retreat_progress_events"
                        ],
                        "reward_retreat_regression": episode_infos[i][
                            "retreat_regression_shaping"
                        ],
                        "reward_retreat_regression_events": episode_infos[i][
                            "retreat_regression_events"
                        ],
                        "reward_retreat_cycle": episode_infos[i][
                            "retreat_cycle_shaping"
                        ],
                        "reward_retreat_cycle_started": episode_infos[i][
                            "retreat_cycle_started"
                        ],
                        "reward_retreat_cycle_completion": episode_infos[i][
                            "retreat_cycle_completion_shaping"
                        ],
                        "reward_retreat_cycle_completion_events": episode_infos[i][
                            "retreat_cycle_completion_events"
                        ],
                        "reward_retreat_cycle_rejoin": episode_infos[i][
                            "retreat_cycle_rejoin_shaping"
                        ],
                        "reward_retreat_cycle_rejoin_events": episode_infos[i][
                            "retreat_cycle_rejoin_events"
                        ],
                        "reward_retreat_cycle_damage": episode_infos[i][
                            "retreat_cycle_danger_damage_shaping"
                        ],
                        "reward_retreat_cycle_death": episode_infos[i][
                            "retreat_cycle_death_shaping"
                        ],
                        "reward_retreat_cycle_timeout_events": episode_infos[i][
                            "retreat_cycle_timeout_events"
                        ],
                        "reward_trade_shaping": episode_infos[i]["trade_shaping"],
                        "reward_trade_raw": episode_infos[i]["trade_shaping_raw"],
                        "reward_trade_clip_steps": episode_infos[i][
                            "trade_clip_steps"
                        ],
                        "reward_economy_shaping": episode_infos[i][
                            "economy_shaping"
                        ],
                        "reward_economy_raw": episode_infos[i]["economy_shaping_raw"],
                        "reward_money_gain": episode_infos[i]["economy_money_gain"],
                        "reward_enemy_money_gain": episode_infos[i][
                            "economy_enemy_money_gain"
                        ],
                        "reward_objective_shaping": episode_infos[i][
                            "objective_shaping"
                        ],
                        "reward_objective_raw": episode_infos[i][
                            "objective_shaping_raw"
                        ],
                        "reward_enemy_objective_damage": episode_infos[i][
                            "objective_enemy_damage"
                        ],
                        "reward_own_objective_damage": episode_infos[i][
                            "objective_own_damage"
                        ],
                        "reward_objective_clip_steps": episode_infos[i][
                            "objective_clip_steps"
                        ],
                        "reward_component_samples": episode_infos[i][
                            "reward_samples"
                        ],
                        "reward_env_dead": episode_infos[i]["env_dead"],
                        "reward_env_ep_rate": episode_infos[i]["env_ep_rate"],
                        "reward_env_exp": episode_infos[i]["env_exp"],
                        "reward_env_hp_point": episode_infos[i]["env_hp_point"],
                        "reward_env_kill": episode_infos[i]["env_kill"],
                        "reward_env_last_hit": episode_infos[i]["env_last_hit"],
                        "reward_env_sdk_money": episode_infos[i]["env_money"],
                        "reward_env_objective_hp": episode_infos[i][
                            "env_objective_hp"
                        ],
                        "win": episode_infos[i]["win"],
                        "hurt_per_frame": episode_infos[i]["hurt_per_frame"],
                        "money_per_frame": episode_infos[i]["money_per_frame"],
                        "totalHurtToHero": episode_infos[i]["totalHurtToHero"],
                        "kill": episode_infos[i]["kill"],
                        "death": episode_infos[i]["death"],
                        "assistCnt": episode_infos[i]["assistCnt"],
                        "hurtH_per_frame": episode_infos[i]["hurtH_per_frame"],
                        "hurtBH_per_frame": episode_infos[i]["hurtBH_per_frame"],
                        "trade_ratio": episode_infos[i]["trade_ratio"],
                        "hp_rate": episode_infos[i]["hp_rate"],
                        "objective_hp_diff": episode_infos[i]["objective_hp_diff"],
                        "own_objective_hp": episode_infos[i]["own_objective_hp"],
                        "enemy_objective_hp": episode_infos[i]["enemy_objective_hp"],
                        "near_own_obj_rate": behavior_metrics["near_own_obj_rate"],
                        "camp_own_obj_rate": behavior_metrics["camp_own_obj_rate"],
                        "pressure_obj_rate": behavior_metrics["pressure_obj_rate"],
                        "near_lane_rate": behavior_metrics["near_lane_rate"],
                        "near_enemy_obj_rate": behavior_metrics[
                            "near_enemy_obj_rate"
                        ],
                        "avg_own_obj_dist": behavior_metrics["avg_own_obj_dist"],
                        "avg_lane_dist": behavior_metrics["avg_lane_dist"],
                        "avg_enemy_obj_dist": behavior_metrics[
                            "avg_enemy_obj_dist"
                        ],
                        "avg_hp_rate": resource_metrics["avg_hp_rate"],
                        "avg_ep_rate": resource_metrics["avg_ep_rate"],
                        "ep_observed_rate": resource_metrics["ep_observed_rate"],
                        "low_resource_rate": resource_metrics["low_resource_rate"],
                        "low_resource_near_home_rate": resource_metrics[
                            "low_resource_near_home_rate"
                        ],
                        "low_resource_far_home_rate": resource_metrics[
                            "low_resource_far_home_rate"
                        ],
                        "hp_recovery_home_rate": resource_metrics[
                            "hp_recovery_home_rate"
                        ],
                        "ep_recovery_home_rate": resource_metrics[
                            "ep_recovery_home_rate"
                        ],
                        "skill_cast_rate": skill_metrics["skill_cast_rate"],
                        "skill_near_target_rate": skill_metrics[
                            "skill_near_target_rate"
                        ],
                        "skill_no_target_rate": skill_metrics["skill_no_target_rate"],
                        "skill_near_hero_rate": skill_metrics["skill_near_hero_rate"],
                        "skill_near_lane_rate": skill_metrics["skill_near_lane_rate"],
                        "skill_near_objective_rate": skill_metrics[
                            "skill_near_objective_rate"
                        ],
                        "skill_followup_success_rate": skill_metrics[
                            "skill_followup_success_rate"
                        ],
                        "win": episode_infos[i]["win"],
                        "length": game_info["length"],
                    }
                )
        LOG.info("game info length:{}".format(game_info["length"]))

        LOG.info("=" * 50)

    def run(
        self, load_models=None, eval_freq=5, gamecore_eval=None, force_eval=False
    ):

        self._episode_num = 0
        # GameCore is sensitive to several sessions calling newGame together.
        # These controls preserve training semantics while pacing only the
        # session lifecycle.  They are especially important for two-network
        # self-play, where a failed actor used to retry every second and turn
        # one startup burst into a sustained overload.
        episode_gap_seconds = _nonnegative_env_float("EPISODE_GAP_SECONDS", 0)
        retry_backoff_seconds = _nonnegative_env_float(
            "EPISODE_FAILURE_BACKOFF_SECONDS", 1
        )
        retry_backoff_max_seconds = max(
            retry_backoff_seconds,
            _nonnegative_env_float("EPISODE_FAILURE_BACKOFF_MAX_SECONDS", 30),
        )
        gamecore_gate = GamecoreConcurrencyGate(
            self.m_config_id,
            _nonnegative_env_int("MAX_CONCURRENT_GAMECORE_GAMES", 0),
        )
        consecutive_failures = 0

        while True:
            camp_config = next(self.camp_iter)
            try:
                self._episode_num += 1
                # provide a init eval value at the first episode
                eval_all_actors = os.getenv("EVAL_ALL_ACTORS", "0") == "1"
                eval_with_common_ai = force_eval or (
                    (eval_all_actors or self.m_config_id == 0)
                    and self._episode_num % eval_freq == 0
                )

                gamecore_gate.acquire()
                try:
                    self._run_episode(
                        camp_config,
                        eval_with_common_ai,
                        load_models=load_models,
                        gamecore_eval=gamecore_eval,
                    )
                finally:
                    gamecore_gate.release()
                consecutive_failures = 0
            except Exception as e:  # pylint: disable=broad-except
                LOG.error(e)
                traceback.print_exc()
                # Do not leave a simulator running after a Python-side
                # episode failure.  Its callbacks can otherwise reach the
                # next episode after the actor retries on the same ports.
                try:
                    self.env.game_launcher.stop_game(self.env.runtime_id)
                    self.env.game_launcher.wait_game(
                        self.env.runtime_id, max_timeout_second=5
                    )
                    LOG.info("stopped failed GameCore runtime before retry")
                except Exception:  # pylint: disable=broad-except
                    LOG.exception("failed to stop GameCore runtime after episode error")
                consecutive_failures += 1
                backoff_seconds = min(
                    retry_backoff_seconds * (2 ** (consecutive_failures - 1)),
                    retry_backoff_max_seconds,
                )
                LOG.warning(
                    "episode failed; retry in {:.1f}s (consecutive failures={})",
                    backoff_seconds,
                    consecutive_failures,
                )
                time.sleep(backoff_seconds)
            else:
                if episode_gap_seconds > 0:
                    LOG.debug("wait {:.1f}s before next episode", episode_gap_seconds)
                    time.sleep(episode_gap_seconds)

            if 0 < self._max_episode <= self._episode_num:
                break

        for agent in self.agents:
            agent.close()
