import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np

from buffer.per_replay_buffer import MultiAgentExclusivePER
from env.dummyenv import DummyEnv
from env.fmu_env_itms import FMUITMS
from model.mlp_block import MLPModel
from model.maddpg import MADDPG
from utils.utils_env import fill_observation, construct_action_dict, fill_list_with_dict, scale_actions
from utils.utils_i2c import generate_msg_observation
from utils.utils_misc import C_to_K, K_to_C, press_scroll_lock
from utils.utils_reward import RewardCalculator

from buffer.replay_buffer import ReplayBuffer
from buffer.kl_buffer import KLBuffer
from utils.utils_klvalue import get_kl_value, build_kl_sample

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.writer = SummaryWriter(log_dir=config.get("log_dir", "runs/"))

        # 是否启用 I2C 模块
        self.use_i2c = config.get("use_i2c", False)
        self.use_agent_buffer = config.get("use_agent_buffer", False)
        if self.use_i2c:
            self.prior_nets = torch.nn.ModuleList([
                MLPModel(input_dim=config["state_dims"][i]+config["n_agents"], num_outputs=2, num_layers=config["i2c_num_layers"], hidden_dim=config["i2c_hidden_dim"]).to(self.device)
                for i in range(config["n_agents"])
            ])
            self.prior_nets_optimizers = [
                torch.optim.Adam(pn.parameters(), lr=1e-3)
                for pn in self.prior_nets
            ]
            # 为每个prior_net构造一个klbuffer
            self.prior_buffers = []
            self.obs_onehot_dim = len(self.config["obs_dict"])
            for agent_i in range(self.obs_onehot_dim):
                obs_dim = self.config["obs_dict"][agent_i].shape[1]
                buf = KLBuffer(
                    buffer_size=self.config["prior_buffer_size"],
                    obs_dim=obs_dim,
                    obs_onehot_dim=self.obs_onehot_dim,
                    percentile=self.config["prior_buffer_percentile"]
                )
                self.prior_buffers.append(buf)
            self.message_nets = [
                MLPModel(sum(len(self.config["obs_dict"][ii]) for ii in range(self.config["n_agents"]) if ii != agent_i), self.config["message_feature_dim"]).to(self.device)
                for agent_i in range(self.config["n_agents"])
            ]
            self.maddpg = MADDPG(
                device=self.device,
                actor_lr=config["actor_lr"],
                critic_lr=config["critic_lr"],
                hidden_dim=config["hidden_dim"],
                obs_dims=config["enhanced_obs_dims"],  # 增强观测维度=局部obs_dim+msg_feature_dim
                action_dis_dims=config["action_dis_dims"],
                action_con_dims=config["action_con_dims"],
                critic_input_dim=config["critic_input_dim"],
                gamma=config["gamma"],
                tau=config["tau"]
            )
            # 将message_nets的optimizer也更新到maddpg-ddpg中
            for i, agent in enumerate(self.maddpg.agents):
                agent.actor_optimizer = torch.optim.Adam(
                    list(agent.actor.parameters()) +
                    list(self.message_nets[i].parameters()),
                    lr=self.config["actor_lr"]
                )
        else:
            self.maddpg = MADDPG(
                device=self.device,
                actor_lr=config["actor_lr"],
                critic_lr=config["critic_lr"],
                hidden_dim=config["hidden_dim"],
                obs_dims=config["obs_dims"],
                action_dis_dims=config["action_dis_dims"],
                action_con_dims=config["action_con_dims"],
                critic_input_dim=config["critic_input_dim"],
                gamma=config["gamma"],
                tau=config["tau"]
            )
        # 初始化其他组件
        self.replay_buffer = ReplayBuffer(self.config["buffer_size"])
        self.reward_calculator = RewardCalculator(
            config["T_cabin_set"],
            config["T_bat_set"],
            config["T_motor_set"]
        )
        self.env = FMUITMS(fmu_path=config["fmu_path"], step_size=config["step_size"])
        self.step_count = 0
        self.episode_count = 0



    def get_hard_labels(self):
        states, actions, _, _, _ = self.replay_buffer.sample(self.config["prior_buffer_size"])
        is_full_list = [False] * len(self.prior_buffers)
        for i in range(len(states)):
            obs_n = states[i]
            act_n = actions[i]
            for agent_i in range(len(self.prior_buffers)):
                with torch.no_grad():
                    obs_i_rep, comm_id, KL_vals = get_kl_value(
                        agents=self.maddpg.agents,
                        obs_n=obs_n,
                        act_n=act_n,
                        agent_i=agent_i,
                        action_bounds=self.config["action_bounds"],
                        action_sep_num=self.config["action_sep_num"],
                        merged_action_dict=self.config["merged_action_dict"],
                        temperature=self.config["kl_temperature"]
                    )
                obs_inputs, obs_onehot_inputs, KL_values = build_kl_sample(obs_i_rep, comm_id, KL_vals)
                is_full = self.prior_buffers[agent_i].insert(
                    obs_inputs=obs_inputs.cpu().numpy(),  # [B, obs_dim_i]
                    obs_onehot_inputs=obs_onehot_inputs.numpy(),
                    KL_values=KL_values.cpu().numpy()
                )
                is_full_list[agent_i] = is_full
        return is_full_list

    def update_prior(self):
        for agent_i in range(len(self.prior_buffers)):
            total_loss = 0.0
            total_cnt = 0
            for i in range(self.config["prior_train_iter"]):
                buffer = self.prior_buffers[agent_i]
                if len(buffer) < self.config["prior_train_batch_size"]:
                    continue
                obs_inputs, obs_onehot_inputs, labels = buffer.get_samples(self.config["prior_train_batch_size"])
                obs_inputs = torch.tensor(obs_inputs, dtype=torch.float32, device=self.device)
                obs_onehot_inputs = torch.tensor(obs_onehot_inputs, dtype=torch.float32, device=self.device)
                labels = torch.tensor(labels, dtype=torch.long, device=self.device)  # [B]
                prior_net = self.prior_nets[agent_i]
                optimizer = self.prior_nets_optimizers[agent_i]
                logits = prior_net(obs_inputs, obs_onehot_inputs)  # [B, 2]
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_cnt += 1
            agent_i_loss_avg = total_loss / total_cnt


    @torch.no_grad()
    def take_action(self, all_obs, explore=True, training=True):
        if self.use_i2c:
            msg_obs, comm_mask = generate_msg_observation(
                self.prior_nets,
                all_obs,
                self.device,
                for_replay=not training
            )
            enhanced_obs = []
            for i in range(self.config["n_agents"]):
                msg_feat = self.message_nets[i](msg_obs[i])
                enhanced_obs_i = torch.cat(
                    [all_obs[i], msg_feat],
                    dim=-1
                )
                enhanced_obs.append(enhanced_obs_i)
            actions = self.maddpg.take_action(enhanced_obs, explore=explore)
        else:
            actions = self.maddpg.take_action(all_obs, explore=explore)
        return actions


    def run_episode(self):
        obs_raw = self.env.reset()
        episode_reward = [0.0] * self.config["n_agents"]
        done = False
        for step in range(self.config["episode_iter"]):
            obs = fill_observation(self.config["obs_dict"], obs_raw)
            actions = self.take_action(obs, explore=True, training=True)
            actions = construct_action_dict(actions, self.config["action_con_str_dict"], self.config["action_dis_str_dict"])
            actions = scale_actions(actions, self.config["action_bounds"])
            # actions['V_three'] = 1
            # actions['V_four'] = 1

            next_obs_raw, term, trunc = self.env.step(actions)
            done = any((term, trunc))

            rewards = [
                self.reward_calculator.calculate_cabin_reward(obs_raw["cabinVolume.summary.T"], obs_raw["TableDC3.Pe"]),
                self.reward_calculator.calculate_refrigerant_reward(obs_raw["TableDC.Pe"]),
                self.reward_calculator.calculate_coolant_reward(
                    obs_raw["battery.Batt_top[1].T"],
                    obs_raw["machine.heatCapacitor.T"],
                    obs_raw["TableDC1.Pe"],
                    obs_raw["TableDC2.Pe"]
                )
            ]
            _fill_next_obs = fill_observation(self.config["obs_dict"], next_obs_raw)
            _fill_next_obs = torch.zeros_like(torch.FloatTensor(_fill_next_obs)) if done else _fill_next_obs

            self.replay_buffer.add([obs, _fill_next_obs, fill_list_with_dict(self.config["merged_action_dict"], actions), rewards,
                                    [True] * self.config["n_agents"] if done else [False] * self.config["n_agents"]])
            obs_raw = next_obs_raw
            for i, r in enumerate(rewards):
                episode_reward[i] += r

            if self.replay_buffer.size >= self.config["batch_size"]:
                samples, distribution_agents, distribution_indices = self.replay_buffer.sample(self.config["batch_size"])
                batched_list = list(zip(*samples))

                def stack_array(x):
                    rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
                    stacked = []
                    for i, aa in enumerate(rearranged):
                        arr = np.vstack(aa)
                        tensor = torch.FloatTensor(arr).to(self.device)
                        stacked.append(tensor)
                    return stacked

                _obs, _next_obs, _act, _rew, _done = [stack_array(x) for x in batched_list]
                _obs_i = None
                all_td_errors = [[] for _ in range(self.config["n_agents"])]
                agent_stats_list = []
                _msg_obs = None
                if self.use_i2c:
                    _msg_obs, _ = generate_msg_observation(self.prior_nets, _obs, self.device, for_replay=False)
                for agent_i in range(self.config["n_agents"]):
                    if self.use_i2c:
                        _msg_feature_i = self.message_nets[agent_i](_msg_obs[agent_i])
                        _obs_i = torch.cat([_obs[agent_i], _msg_feature_i], dim=-1)
                    rt, critic_loss, actor_loss, agent_stats = self.maddpg.update([_obs_i, _act, _rew, _next_obs, _done], agent_i)
                    all_td_errors[agent_i] = rt
                    agent_stats_list.append(agent_stats)
                self.maddpg.update_all_targets()
                if self.use_i2c:
                    is_full_list = self.get_hard_labels()
                    if all(is_full_list):
                        self.update_prior()
                self.step_count += 1
                self.log_metrics()
        self.episode_count += 1
        return episode_reward


    def evaluate_episode(self):
        pass

    def run(self):
        for episode in tqdm(range(self.config["num_episodes"])):
            reward = self.run_episode()
            print(f"[Episode {episode}] Total Reward: {sum(reward)}")

    def log_metrics(self):
        pass # ex. self.writer.add_scalar(...k, ...v, self.step_count)

    def save_checkpoint(self, path):
        pass

    def load_checkpoint(self, path):
        pass



def derive_dimensions(obs_dict, action_dis_str_dict, action_con_str_dict, message_feature_dim):
    """根据观测和动作字典自动推导维度信息"""
    obs_dims = [len(l) for l in obs_dict]
    action_dis_dims = [len(l) for l in action_dis_str_dict]
    action_con_dims = [len(l) for l in action_con_str_dict]
    # 计算增强观测维度（局部观测 + 所有观测总和）
    enhanced_obs_dims = [sd + message_feature_dim for sd in obs_dims]
    # 计算critic输入维度
    critic_input_dim = sum(obs_dims) + sum(action_dis_dims) + sum(action_con_dims)
    # 其他配置
    n_agents = len(obs_dims)
    i2c_hidden_dim = 256
    return {
        "obs_dims": obs_dims,
        "action_con_dims": action_con_dims,
        "action_dis_dims": action_dis_dims,
        "enhanced_obs_dims": enhanced_obs_dims,
        "critic_input_dim": critic_input_dim,
        "n_agents": n_agents,
        "i2c_hidden_dim": i2c_hidden_dim,
        "state_dims": obs_dims,
        "merged_action_dict": merged_action_dict,
        "action_bounds": action_bounds,
        "T_cabin_set": T_cabin_set,
        "T_bat_set": T_bat_set,
        "T_motor_set": T_motor_set,
        "obs_dict": obs_dict,
        "action_con_str_dict": action_con_str_dict,
        "action_dis_str_dict": action_dis_str_dict,
        "replay_buffer_agents_sample_weight": [0.4, 0.3, 0.3]
    }

# 完整配置
def get_config(base_config, obs_dict, action_dis_str_dict, action_con_str_dict):
    """获取完整配置"""
    config = base_config.copy()
    config.update(derive_dimensions(obs_dict, action_dis_str_dict, action_con_str_dict, config["message_feature_dim"]))
    return config


if __name__ == "__main__":
    import torch
    from utils.utils_misc import C_to_K

    # 基础配置
    base_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_dir": "runs/",
        "num_episodes": 100,
        "episode_iter": 50,
        "buffer_size": 100,
        "hidden_dim": 1024,
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "gamma": 0.95,
        "tau": 1e-2,
        "batch_size": 2,
        "update_interval": 50,
        "minimal_size": 100,
        "fmu_path": "MyITMS.fmu",
        "step_size": 1,
        "use_i2c": True,
        "lambda_temp": 10.0,
        "message_feature_dim": 16,
    }

    # 观测和动作配置
    obs_dict = [
        ["T_cabin_set", "cabinVolume.summary.T", "driverPerformance.controlBus.driverBus._acc_pedal_travel",
         "driverPerformance.controlBus.driverBus._brake_pedal_travel",
         "driverPerformance.controlBus.vehicleStatus.vehicle_velocity"],
        ["superHeatingSensor.outPort", "superCoolingSensor.outPort", "battery.controlBus.batteryBus.battery_SOC[1]"],
        ["T_bat_set", "T_motor_set", "battery.Batt_top[1].T", "machine.heatCapacitor.T"]
    ]

    action_con_str_dict = [
        ["RPM_blower"],
        ["RPM_comp"],
        ["RPM_batt", "RPM_motor"]
    ]

    action_dis_str_dict = [
        [],
        [],
        ['V_three', 'V_four']
    ]

    # 合并动作字典
    merged_action_dict = [
        con + dis
        for con, dis in zip(action_con_str_dict, action_dis_str_dict)
    ]

    # 动作边界配置
    action_bounds = {
        "RPM_blower": [0, 300],
        "RPM_comp": [0, 3000],
        "RPM_batt": [0, 3000],
        "RPM_motor": [0, 3000],
        "V_three": [True, False],
        "V_four": [True, False],
    }
    action_sep_num = {
        "T_epsilon": 6,
        "RPM_blower": 10,
        "RPM_comp": 30,
        "RPM_batt": 30,
        "RPM_motor": 30,
    }
    # 温度设置
    T_cabin_set, T_bat_set, T_motor_set = C_to_K([20, 30, 90])


    # 从配置中推导维度信息
    config = get_config(base_config, obs_dict, action_dis_str_dict, action_con_str_dict)
    trainer = Trainer(config)
    trainer.run()