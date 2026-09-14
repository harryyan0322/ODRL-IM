# yapf: disable
from collections import deque
import copy
import os
from dowel import tabular
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import cloudpickle
from garage import log_performance, obtain_evaluation_episodes, StepType
from garage.np.algos import RLAlgorithm
from garage.torch import as_torch_dict, global_device
from torch_geometric.data import DataLoader

# yapf: enable

torch.set_flush_denormal(True)


def normalized_sum(loss, reg, w):
    return loss / w + reg if w > 1 else loss + w * reg

def normalized_loss(loss, reg):
    # 计算 a 和 b 的均值和标准差
    mean = (loss + reg) / 2
    std = torch.sqrt(((loss - mean) ** 2 + (reg - mean) ** 2) / 2)

    # 归一化 a 和 b
    loss_normalized = (loss - mean) / std
    reg_normalized = (reg - mean) / std

    # 将归一化后的 a 和 b 添加到损失函数中
    loss_new = loss_normalized + reg_normalized 
    return loss_new

def l2_projection(constraint):
    @torch.no_grad()
    def fn(module):
        if hasattr(module, 'weight') and constraint > 0:
            w = module.weight
            norm = torch.norm(w)  # 计算2范数 shape(1,)
            w.mul_(torch.clip(constraint / norm, max=1))  # 如果L2范数超过100了，就要对其W进行投影（放缩）。_mul_() 表示逐元素乘

    return fn


def weight_l2(model):
    l2 = 0.
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2 += torch.norm(param) ** 2
    return l2


class ATAC(RLAlgorithm):
    """ Adversarilly Trained Actor Critic """

    def __init__(
            self,
            env_spec,
            policy,
            qf1,
            qf2,
            replay_buffer,
            sampler,
            device,
            graph,
            *,  # Everything after this is numbers.
            max_episode_length_eval=None,
            gradient_steps_per_itr,
            fixed_alpha=None,
            target_entropy=None,
            initial_log_entropy=0.,
            discount=0.99,
            buffer_batch_size=256,
            min_buffer_size=int(1e4),
            target_update_tau=5e-3,
            policy_lr=5e-6,
            qf_lr=5e-4,
            alpha_lr=None,
            reward_scale=1.0,
            optimizer='Adam',
            steps_per_epoch=1,
            num_evaluation_episodes=10,
            eval_env=None,
            use_deterministic_evaluation=True,
            # ATAC parameters
            beta=1.0,  # the regularization coefficient in front of the Bellman error
            lambd=0.,  # coeff for global pessimism
            init_observations=None,  # for ATAC0 (None or np.ndarray)
            n_warmstart_steps=200000,
            norm_constraint=100,
            q_eval_mode='0.5_0.5',  # 'max' 'w1_w2', 'adaptive'
            q_eval_loss='MSELoss',  # 'MSELoss', 'SmoothL1Loss'
            use_two_qfs=True,
            terminal_value=None,
            Vmin=-float('inf'),  # min value of Q (used in target backup)
            Vmax=float('inf'),  # max value of Q (used in target backup)
            debug=True,
            stats_avg_rate=0.99,  # for logging
            bellman_surrogate='td',  # 'td', None, 'target'
            use_lr_scheduler=False,
            use_std_loss=False,
            data_saving_dir=None,
            algorithm_mode='atac',
            cql_alpha=1.0,
    ):

        #############################################################################################

        assert beta >= 0
        assert norm_constraint >= 0
        # Parsing
        optimizer = eval('torch.optim.' + optimizer)
        # policy_lr = qf_lr if policy_lr is None or policy_lr < 0 else policy_lr # use shared lr if not provided.
        # policy_lr = policy_lr if use_lr_scheduler else qf_lr*1e-3 # use shared lr if not provided.
        ## ATAC parameters
        self.beta = torch.Tensor([beta])  # regularization constant on the Bellman surrogate
        self._lambd = torch.Tensor([lambd])  # global pessimism coefficient
        self._init_observations = torch.Tensor(
            init_observations) if init_observations is not None else init_observations  # if provided, it runs ATAC0
        self._n_warmstart_steps = n_warmstart_steps  # during which, it performs independent C and Bellman minimization
        # q update parameters
        self._norm_constraint = norm_constraint  # l2 norm constraint on the qf's weight; if negative, it gives the weight decay coefficient.
        # 消融实验重置这里的权重误差
        self._q_eval_mode = [float(w) for w in
                             q_eval_mode.split('_')] if '_' in q_eval_mode else q_eval_mode  # residual algorithm
        self._q_eval_loss = eval('torch.nn.' + q_eval_loss)(reduction='none')
        self._use_two_qfs = use_two_qfs
        self.graph = graph  # the graph object
        self._Vmin = Vmin  # lower bound on the target
        self._Vmax = Vmax  # upper bound on the target
        self._terminal_value = terminal_value if terminal_value is not None else lambda r, gamma: 0.

        # Stepsizes
        self._alpha_lr = qf_lr if alpha_lr is None else float(alpha_lr)
        self._bc_policy_lr = qf_lr  # potentially a larger stepsize

        # Logging and algorithm state
        self._debug = debug
        self._n_updates_performed = 0  # Counter of number of grad steps performed
        self._cac_learning = False
        self._stats_avg_rate = stats_avg_rate
        self._bellman_surrogate = bellman_surrogate
        self._avg_bellman_error = 1.  # for logging; so this works with zero warm-start
        self._avg_terminal_td_error = 1

        #############################################################################################
        # Original SAC parameters
        self._qf1 = qf1
        self._qf2 = qf2
        self.replay_buffer = replay_buffer
        self._tau = target_update_tau
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._initial_log_entropy = initial_log_entropy
        self._gradient_steps = gradient_steps_per_itr
        self._optimizer = optimizer
        self._num_evaluation_episodes = num_evaluation_episodes
        self._eval_env = eval_env
        self._use_lr_scheduler = use_lr_scheduler
        self._use_std_loss = use_std_loss
        self._min_buffer_size = min_buffer_size
        self._steps_per_epoch = steps_per_epoch
        self._buffer_batch_size = buffer_batch_size
        self._discount = discount
        self._reward_scale = reward_scale
        # self.max_episode_length = env_spec.max_episode_length
        # self._max_episode_length_eval = env_spec.max_episode_length

        if max_episode_length_eval is not None:
            self._max_episode_length_eval = max_episode_length_eval
        self._use_deterministic_evaluation = use_deterministic_evaluation

        self.policy = policy
        self.env_spec = env_spec
        self._device = device
        self.replay_buffer = replay_buffer
        self._data_saving_dir = data_saving_dir
        self._sampler = sampler
        self._algorithm_mode = str(algorithm_mode).lower()
        if self._algorithm_mode not in ('atac', 'vanilla_ac', 'cql'):
            raise ValueError(
                'algorithm_mode must be atac, vanilla_ac, or cql'
            )
        self._cql_alpha = float(cql_alpha)
        if self._cql_alpha < 0:
            raise ValueError('cql_alpha must be non-negative')

        # use 2 target q networks
        self._target_qf1 = copy.deepcopy(self._qf1)  # 初始target网络和q网络一样
        self._target_qf2 = copy.deepcopy(self._qf2)
        self._policy_optimizer = self._optimizer(self.policy.parameters(),
                                                 lr=self._bc_policy_lr)  # lr for warmstart
        self._qf1_optimizer = self._optimizer(self._qf1.parameters(),
                                              lr=self._qf_lr)
        self._qf2_optimizer = self._optimizer(self._qf2.parameters(),
                                              lr=self._qf_lr)
        if self._use_lr_scheduler:
            # scheduler = lr_scheduler.ReduceLROnPlateau(self._policy_optimizer, mode='min', factor=0.5, patience=2, threshold=0.001)
            scheduler = lr_scheduler.StepLR(self._policy_optimizer, step_size=50, gamma=0)
            self._policy_lr_scheduler = scheduler
        # automatic entropy coefficient tuning
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            if target_entropy is not None:
                self._target_entropy = target_entropy
            else:
                self._target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            self._log_alpha = torch.Tensor([self._initial_log_entropy
                                            ]).requires_grad_()
            self._alpha_optimizer = optimizer([self._log_alpha],
                                              lr=self._alpha_lr)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha]).log()
        self.episode_rewards = deque(maxlen=30)

    def _policy_actions(self, observations, edge_index):
        output = self.policy(observations, edge_index)
        if getattr(self.policy, "output_range", None) == "unit_interval":
            return output.clamp(0.0, 1.0)
        return torch.tanh(output) + 1

    @staticmethod
    def _critic_input(critic, actions, observations):
        if getattr(critic, "expects_state", False):
            return torch.cat((observations, actions), dim=-1)
        return actions

    def _critic_mean(self, critic, actions, observations, edge_index):
        values = critic(
            self._critic_input(critic, actions, observations), edge_index
        )
        return torch.mean(values, dim=(1, 2))

    def optimize_policy(self,
                        samples_data,
                        warmstart=False):
        """Optimize the policy q_functions, and temperature coefficient.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        device = self._device

        obs = samples_data['obs'].to(device)
        next_obs = samples_data['next_obs'].to(device)
        raw_actions = samples_data['action'].to(device)
        if getattr(self.policy, "output_range", None) == "unit_interval":
            actions = raw_actions.clamp(0.0, 1.0)
        else:
            actions = torch.tanh(raw_actions)+1
        # Keep one scalar reward/terminal per transition.  The replay buffer
        # may provide ``(B, 1)`` tensors while the critic reduction returns
        # ``(B,)``; flatten both sides before the Bellman backup so the batch
        # dimension is preserved (and is not accidentally averaged away).
        rewards = (samples_data['reward'] * self._reward_scale).to(device).flatten()
        terminals = samples_data['terminal'].to(device).flatten()
        # A = torch.tensor(samples_data['edge_index'].clone(), dtype=torch.long).to(device)
        A = samples_data['edge_index'].clone().detach().to(dtype=torch.long, device=device)
        if obs.dim() == 3 and obs.shape[0] > 1:
            batch_size, node_count = obs.shape[:2]
            offsets = (
                torch.arange(batch_size, device=device, dtype=A.dtype)
                * node_count
            ).view(batch_size, 1, 1)
            A = (A.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)

        # print("A=",A.shape)
        # exit()

        def compute_bellman_backup(q_pred_next):

            assert rewards.shape == q_pred_next.shape
            return rewards + (1. - terminals) * self._discount * q_pred_next + terminals * self._terminal_value(rewards,
                                                                                                                self._discount)

        def compute_bellman_loss(q_pred, q_pred_next, q_target):

            assert q_pred.shape == q_pred_next.shape == q_target.shape
            target_error = self._q_eval_loss(q_pred,
                                             q_target)  # MSE(Q_1(s,a),r+gamma*Q_target(s',a')) ,epsilon^{td}(f,f-min,pi)
            q_target_pred = compute_bellman_backup(q_pred_next)  # r + (1-d)*gamma*Q_1(s',a') + d*V(s')*0.0
            td_error = self._q_eval_loss(q_pred,
                                         q_target_pred)  # MSE(Q_1(s,a),r+gamma*Q_1(s',a')) ,epsilon^{td}(f,f,pi)
            w1, w2 = self._q_eval_mode  # w1=0.5 w2=0.5
            bellman_loss = w1 * target_error + w2 * td_error  # epsilon^{td}(f,pi)  (7)
            return bellman_loss, target_error, td_error

        ##### Update Critic #####
        ## Compute Bellman error
        with torch.no_grad():

            # print("next_obs.shape=", next_obs.shape)
            # print("obs.shape=", obs.shape)
            # exit()
            # 对动作的修正
            new_next_actions = self._policy_actions(next_obs, A)
            # new_next_actions = torch.softmax(self.policy(next_obs, A), dim=1)
            # new_next_actions = torch.sigmoid(self.policy(next_obs, A))
            # print("new_next_actions_dist", new_next_actions_dist)
            # exit()
            target_q_values = self._critic_mean(
                self._target_qf1, new_next_actions, next_obs, A
            ).flatten()

            if self._use_two_qfs:
                target_q_values = torch.min(target_q_values,
                                            self._critic_mean(
                                                self._target_qf2,
                                                new_next_actions,
                                                next_obs,
                                                A,
                                            ).flatten())

            q_target = compute_bellman_backup(target_q_values.flatten())

        qf1_pred = self._critic_mean(self._qf1, actions, obs, A).flatten()

        qf1_pred_next = self._critic_mean(self._qf1, new_next_actions, next_obs, A).flatten()

        qf1_bellman_losses, qf1_target_errors, qf1_td_errors = compute_bellman_loss(qf1_pred, qf1_pred_next, q_target)
        qf1_bellman_loss = qf1_bellman_losses.mean()
        qf2_bellman_loss = qf2_target_error = qf2_td_error = torch.Tensor([0.])

        if self._use_two_qfs:
            qf2_pred = self._critic_mean(self._qf2, actions, obs, A).flatten()
            qf2_pred_next = self._critic_mean(self._qf2, new_next_actions, next_obs, A).flatten()
            qf2_bellman_losses, qf2_target_errors, qf2_td_errors = compute_bellman_loss(qf2_pred, qf2_pred_next,
                                                                                        q_target)
            qf2_bellman_loss = qf2_bellman_losses.mean()

        # Compute GAN error
        # These samples will be used for the actor update too, so they need to be traced.
        # new_actions_pre_tanh, new_actions = new_actions_dist.rsample_with_pre_tanh_value() #_,a'=pi(s) ;第一项为取tanh前的值
        new_actions = self._policy_actions(obs, A)
        # new_actions = torch.softmax(self.policy(obs, A), dim=1)
        # new_actions = new_actions_dist

        gan_qf1_loss = gan_qf2_loss = 0
        if not warmstart:

            qf1_new_actions = self._critic_mean(self._qf1, new_actions.detach(), obs, A)

            gan_qf1_loss = (qf1_new_actions * (1 + self._lambd) - qf1_pred).mean()

            if self._use_two_qfs:
                qf2_new_actions = self._critic_mean(self._qf2, new_actions.detach(), obs, A)
                gan_qf2_loss = (qf2_new_actions * (1 + self._lambd) - qf2_pred).mean()

        ## Compute full q loss for the selected offline-RL baseline.
        with torch.no_grad():
            beta = self.beta
        cql_qf1_loss = torch.zeros((), device=device)
        cql_qf2_loss = torch.zeros((), device=device)
        if self._algorithm_mode == 'vanilla_ac':
            qf1_loss = qf1_bellman_loss
            qf2_loss = qf2_bellman_loss
        elif self._algorithm_mode == 'cql':
            # CQL for the combinatorial action space: compare the data action,
            # the current policy action, and deterministic random K-hot
            # actions.  All candidates preserve the observed seed budget,
            # which is substantially less biased than the old all-zero /
            # all-one endpoint penalty.  Dividing by the candidate count keeps
            # the log-sum-exp on a stable scale as the candidate set grows.
            def khot_candidates(reference, count=8):
                candidates = [actions.detach(), reference.detach()]
                node_count = reference.shape[1]
                for candidate_index in range(count):
                    random_scores = torch.rand(
                        reference.shape[0], node_count, device=device
                    )
                    requested = torch.round(
                        raw_actions.detach().sum(dim=(1, 2))
                    ).long().clamp(min=1, max=node_count)
                    random_action = torch.zeros_like(reference)
                    for row in range(reference.shape[0]):
                        chosen = torch.topk(
                            random_scores[row], int(requested[row].item())
                        ).indices
                        random_action[row, chosen, 0] = 1.0
                    candidates.append(torch.tanh(random_action) + 1.0)
                return candidates

            cql_candidates = khot_candidates(new_actions)

            def conservative_penalty(q_function, data_pred):
                values = [
                    self._critic_mean(q_function, candidate, obs, A)
                    for candidate in cql_candidates
                ]
                stacked = torch.stack(values, dim=1)
                return (
                    torch.logsumexp(stacked, dim=1)
                    - np.log(float(len(values)))
                    - data_pred.detach()
                ).mean()

            cql_qf1_loss = conservative_penalty(self._qf1, qf1_pred)
            qf1_loss = qf1_bellman_loss + self._cql_alpha * cql_qf1_loss
            if self._use_two_qfs:
                qf2_values = [
                    self._critic_mean(self._qf2, candidate, obs, A)
                    for candidate in cql_candidates
                ]
                cql_qf2_loss = (
                    torch.logsumexp(torch.stack(qf2_values, dim=1), dim=1)
                    - np.log(float(len(qf2_values)))
                    - qf2_pred.detach()
                ).mean()
                qf2_loss = qf2_bellman_loss + self._cql_alpha * cql_qf2_loss
            else:
                qf2_loss = qf2_bellman_loss
        else:
            qf1_loss = normalized_sum(gan_qf1_loss, qf1_bellman_loss, beta)
            qf2_loss = normalized_sum(gan_qf2_loss, qf2_bellman_loss, beta)

        if beta > 0 or not warmstart or self._algorithm_mode != 'atac':
            self._qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self._qf1_optimizer.step()
            self._qf1.apply(l2_projection(self._norm_constraint))

            if self._use_two_qfs:
                self._qf2_optimizer.zero_grad()
                qf2_loss.backward()
                self._qf2_optimizer.step()
                self._qf2.apply(l2_projection(self._norm_constraint))  # 对Q2的每一层权重都进行l2正则化映射

        ##### Update Actor #####

        policy_entropy = (-(new_actions * torch.log(new_actions + 1e-10)
                           + (1.0 - new_actions) * torch.log(1.0 - new_actions + 1e-10))).sum(dim=2).mean()
        alpha_loss = 0

        if self._use_automatic_entropy_tuning:  # it comes first; seems to work also when put after policy update
            alpha_loss = self._log_alpha * (policy_entropy.detach() - self._target_entropy)  # entropy - target
            self._alpha_optimizer.zero_grad()  # 对log_alpha进行优化
            alpha_loss.backward()
            self._alpha_optimizer.step()

        with torch.no_grad():
            alpha = self._log_alpha.exp()

        lower_bound = 0
        policy_std_loss = torch.FloatTensor(np.array(0.)).to(device)
        if self._algorithm_mode in ('vanilla_ac', 'cql') and not warmstart:
            # Recompute after the critic step.  Freezing critic parameters
            # avoids backpropagating actor gradients into a just-updated Q net.
            q_parameters = list(self._qf1.parameters())
            for parameter in q_parameters:
                parameter.requires_grad_(False)
            actor_actions = self._policy_actions(obs, A)
            min_q_new_actions = self._critic_mean(self._qf1, actor_actions, obs, A)
            for parameter in q_parameters:
                parameter.requires_grad_(True)
            policy_entropy = (-(actor_actions * torch.log(actor_actions + 1e-10)
                               + (1.0 - actor_actions) * torch.log(1.0 - actor_actions + 1e-10))).sum(dim=2).mean()
        else:
            # Relative-pessimism Actor objective: differentiate through the
            # Critic with respect to actions, while never accumulating Critic
            # gradients during the Actor optimizer step.
            q_parameters = list(self._qf1.parameters())
            for parameter in q_parameters:
                parameter.requires_grad_(False)
            min_q_new_actions = self._critic_mean(self._qf1, new_actions, obs, A)
            for parameter in q_parameters:
                parameter.requires_grad_(True)
        if warmstart:  # BC warmstart

            policy_log_prob = (actions * torch.log2(actions + 1e-10)).sum(dim=2).mean()

            policy_loss = normalized_sum(policy_log_prob.mean(), -policy_entropy, alpha)
        elif self._algorithm_mode in ('vanilla_ac', 'cql'):
            policy_loss = -min_q_new_actions.mean()
        else:
            # Compute performance difference lower bound
            std_weight = 1e-3
            policy_std_loss = torch.FloatTensor(np.array(0.)).to(device)
            # 实例化损失函数
            mse_loss_fn = nn.MSELoss()
            # 计算均方误
            if self._use_std_loss:
                print("use_std_loss")
                # policy_std_loss = compute_std_loss(ip_data_statistics)
            # policy_loss = normalized_sum(-min_q_new_actions.mean(), -policy_entropy, alpha) + std_weight * policy_std_loss  # (-)
            # 损失函数这里应该再好好考虑一下，如果要考虑之前的一项，那么应该引入一个函数以缩小他们之间的倍率
            # policy_loss = normalized_loss(1e7*mse_loss_fn(new_actions, actions),normalized_sum(-min_q_new_actions.mean(), -policy_entropy, alpha))
            # 尝试的loss
            # policy_loss = mse_loss_fn(new_actions, actions)
            # 修改后的消融loss，前提是warmstart必须为零
            # ``min_q_new_actions`` is evaluated from the non-detached Actor
            # output above; ``qf1_pred`` is the logged-action reference.  The
            # Critic parameters are frozen during the actor step below, so
            # this relative-pessimism term propagates gradients to the Actor.
            policy_loss = (-(min_q_new_actions - qf1_pred).mean()
                           - alpha * (policy_entropy - self._target_entropy))
            # policy_loss = normalized_sum(-min_q_new_actions.mean(), -policy_entropy, alpha)

        log_info = dict(
            policy_loss=policy_loss,
            policy_std_loss=policy_std_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            qf1_bellman_loss=qf1_bellman_loss,
            gan_qf1_loss=gan_qf1_loss,
            qf2_bellman_loss=qf2_bellman_loss,
            gan_qf2_loss=gan_qf2_loss,
            cql_qf1_loss=cql_qf1_loss,
            cql_qf2_loss=cql_qf2_loss,
            beta=beta,
            alpha_loss=alpha_loss,
            policy_entropy=policy_entropy,
            alpha=alpha,
            alpha_lr=self._alpha_lr,
            lower_bound=lower_bound,
            policy_lr=self._policy_lr
        )

        # For logging
        if self._debug:
            with torch.no_grad():
                if self._bellman_surrogate == 'td':
                    qf1_bellman_surrogate = qf1_td_errors.mean()
                    qf2_bellman_surrogate = qf2_td_errors.mean()
                elif self._bellman_surrogate == 'target':
                    qf1_bellman_surrogate = qf1_target_errors.mean()
                    qf2_bellman_surrogate = qf2_target_errors.mean()
                elif self._bellman_surrogate is None:
                    qf1_bellman_surrogate = qf1_bellman_loss
                    qf2_bellman_surrogate = qf2_bellman_loss

                bellman_surrogate = torch.max(qf1_bellman_surrogate, qf2_bellman_surrogate)  # measure the TD error
                self._avg_bellman_error = self._avg_bellman_error * self._stats_avg_rate + bellman_surrogate * (
                        1 - self._stats_avg_rate)

                if terminals.sum() > 0:
                    terminal_td_error = (qf1_td_errors * terminals).sum() / terminals.sum()
                    self._avg_terminal_td_error = self._avg_terminal_td_error * self._stats_avg_rate + terminal_td_error * (
                            1 - self._stats_avg_rate)

                qf1_pred_mean = qf1_pred.mean()
                qf2_pred_mean = qf2_pred.mean() if self._use_two_qfs else 0.
                q_target_mean = q_target.mean()
                target_q_values_mean = target_q_values.mean()
                qf1_new_actions_mean = qf1_new_actions.mean() if not warmstart else 0.
                qf2_new_actions_mean = qf2_new_actions.mean() if not warmstart and self._use_two_qfs else 0.
                action_diff = torch.mean(torch.norm(actions - new_actions))

            debug_log_info = dict(
                avg_bellman_error=self._avg_bellman_error,
                avg_terminal_td_error=self._avg_terminal_td_error,
                qf1_pred_mean=qf1_pred_mean,
                qf2_pred_mean=qf2_pred_mean,
                q_target_mean=q_target_mean,
                target_q_values_mean=target_q_values_mean,
                qf1_new_actions_mean=qf1_new_actions_mean,
                qf2_new_actions_mean=qf2_new_actions_mean,
                action_diff=action_diff,
                qf1_target_error=qf1_target_errors.mean(),
                qf1_td_error=qf1_td_errors.mean(),
                qf2_target_error=qf2_target_errors.mean(),
                qf2_td_error=qf2_td_errors.mean()
            )
            log_info.update(debug_log_info)

        return new_actions, log_info

    # Below is overwritten for general logging with log_info
    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        for ep in trainer.step_epochs():
            for step in range(self._steps_per_epoch):  # 1
                for g_step in range(self._gradient_steps):  # ATAC的每一次更新,每一个epoch更新进行2000次梯度下降
                    log_info = self.train_once()
            if self._use_lr_scheduler:
                self._policy_lr_scheduler.step(log_info['policy_loss'])
            if ep % 10 == 0 or ep == trainer._train_args.n_epochs - 1:
                params = {}
                params['policy'] = self.policy
                params['qf1'] = self._qf1
                params_file_name = os.path.join(self._data_saving_dir, 'epoch_%d.pkl' % ep)
                with open(params_file_name, 'wb') as f:
                    cloudpickle.dump(params, f)
            # if self._num_evaluation_episodes>0:
            #     last_return = self._evaluate_policy(trainer.step_itr)
            self._log_statistics(log_info)
            tabular.record('TotalEnvSteps', trainer.total_env_steps)
            trainer.step_itr += 1

        return np.mean(last_return) if last_return is not None else 0

    def train_once(self, itr=None, paths=None):
        """Complete 1 training iteration of ATAC.

        Args:
            itr (int): Iteration number. This argument is deprecated.
            paths (list[dict]): A list of collected paths.
                This argument is deprecated.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        del itr
        del paths
        if (self.replay_buffer.n_transitions_stored) >= self._min_buffer_size:
            warmstart = self._n_updates_performed < self._n_warmstart_steps
            if not warmstart and not self._cac_learning:
                self._cac_learning = True
                # Reset optimizers since the objective changes
                if self._use_automatic_entropy_tuning:
                    self._log_alpha = torch.Tensor([self._initial_log_entropy]).to(
                        self._log_alpha.device).requires_grad_()
                    self._alpha_optimizer = self._optimizer([self._log_alpha], lr=self._alpha_lr)
                self._policy_optimizer = self._optimizer(self.policy.parameters(), lr=self._policy_lr)

            samples = self.replay_buffer.sample_transitions_option(self._buffer_batch_size)
            samples = as_torch_dict(samples)
            # exit()

            new_actions, log_info = self.optimize_policy(samples, warmstart=warmstart)
            self._update_targets()
            self._n_updates_performed += 1
            log_info['n_updates_performed'] = self._n_updates_performed
            log_info['warmstart'] = warmstart

        return log_info

    # Update also the target policy if needed
    def _update_targets(self):  # 在每轮Q1,Q2,Actor更新后也要更新target网络
        """Update parameters in the target q-functions."""
        if self._use_two_qfs:
            target_qfs = [self._target_qf1, self._target_qf2]
            qfs = [self._qf1, self._qf2]
        else:
            target_qfs = [self._target_qf1]
            qfs = [self._qf1]

        for target_qf, qf in zip(target_qfs, qfs):  # 将[Q-target1,Q-target2]和[Q1,Q2]打包后，依次返回Q-target1,Q1;Q-target2,Q2
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (
                        1.0 - self._tau) + param.data * self._tau)  # 下划线结尾表示原地操作，不会创建一个新的副本。对于每一个参数都是Q-target = (1-tau)*Q-target + tau*Q

    # Set also the target policy if needed
    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)
        if not self._use_automatic_entropy_tuning:
            self._log_alpha = torch.Tensor([self._fixed_alpha
                                            ]).log().to(device)
        else:
            self._log_alpha = torch.Tensor([self._initial_log_entropy
                                            ]).to(device).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                                    lr=self._alpha_lr)
        self.beta = torch.Tensor([self.beta]).to(device)
        self._lambd = torch.Tensor([self._lambd]).to(device)

    # Return also the target policy if needed
    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        if self._use_two_qfs:
            networks = [
                self.policy, self._qf1, self._qf2, self._target_qf1,
                self._target_qf2
            ]
        else:
            networks = [
                self.policy, self._qf1, self._target_qf1
            ]

        return networks

    # Evaluate both the deterministic and the stochastic policies
    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """
        eval_episodes = obtain_evaluation_episodes(  # TODO parallel evaluation
            self.policy,
            self._eval_env,
            self._max_episode_length_eval,
            num_eps=self._num_evaluation_episodes,
            deterministic=self._use_deterministic_evaluation)  # return a episode batch
        last_return = log_performance(epoch,
                                      eval_episodes,
                                      discount=self._discount)  # 在采样的episode batch上进行评价

        return last_return

    def _log_statistics(self, log_info):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf1_loss (torch.Tensor): loss from 1st qf/critic network.
            qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

        """
        for k, v in log_info.items():
            tabular.record('Algorithm/' + k, float(v))
        tabular.record('ReplayBuffer/buffer_size', self.replay_buffer.n_transitions_stored)


# from collections import deque
# import copy
# import os
# from dowel import tabular
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.optim.lr_scheduler as lr_scheduler
# import cloudpickle
# from garage import log_performance, obtain_evaluation_episodes, StepType
# from garage.np.algos import RLAlgorithm
# from garage.torch import as_torch_dict, global_device
# from torch_geometric.data import DataLoader

# # yapf: enable

# torch.set_flush_denormal(True)


# def normalized_sum(loss, reg, w):
#     return loss / w + reg if w > 1 else loss + w * reg


# def l2_projection(constraint):
#     @torch.no_grad()
#     def fn(module):
#         if hasattr(module, 'weight') and constraint > 0:
#             w = module.weight
#             norm = torch.norm(w)  # 计算2范数 shape(1,)
#             w.mul_(torch.clip(constraint / norm, max=1))  # 如果L2范数超过100了，就要对其W进行投影（放缩）。_mul_() 表示逐元素乘

#     return fn


# def weight_l2(model):
#     l2 = 0.
#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             l2 += torch.norm(param) ** 2
#     return l2


# class ATAC(RLAlgorithm):
#     """ Adversarilly Trained Actor Critic """

#     def __init__(
#             self,
#             env_spec,
#             policy,
#             qf1,
#             qf2,
#             replay_buffer,
#             sampler,
#             device,
#             graph,
#             *,  # Everything after this is numbers.
#             max_episode_length_eval=None,
#             gradient_steps_per_itr,
#             fixed_alpha=None,
#             target_entropy=None,
#             initial_log_entropy=0.,
#             discount=0.99,
#             buffer_batch_size=8,
#             min_buffer_size=int(1e4),
#             target_update_tau=5e-3,
#             policy_lr=5e-6,
#             qf_lr=5e-4,
#             reward_scale=1.0,
#             optimizer='Adam',
#             steps_per_epoch=1,
#             num_evaluation_episodes=10,
#             eval_env=None,
#             use_deterministic_evaluation=True,
#             # ATAC parameters
#             beta=1.0,  # the regularization coefficient in front of the Bellman error
#             lambd=0.,  # coeff for global pessimism
#             init_observations=None,  # for ATAC0 (None or np.ndarray)
#             n_warmstart_steps=200000,
#             norm_constraint=100,
#             q_eval_mode='0.5_0.5',  # 'max' 'w1_w2', 'adaptive'
#             q_eval_loss='MSELoss',  # 'MSELoss', 'SmoothL1Loss'
#             use_two_qfs=True,
#             terminal_value=None,
#             Vmin=-float('inf'),  # min value of Q (used in target backup)
#             Vmax=float('inf'),  # max value of Q (used in target backup)
#             debug=True,
#             stats_avg_rate=0.99,  # for logging
#             bellman_surrogate='td',  # 'td', None, 'target'
#             use_lr_scheduler=False,
#             use_std_loss=False,
#             data_saving_dir=None,
#     ):

#         #############################################################################################

#         assert beta >= 0
#         assert norm_constraint >= 0
#         # Parsing
#         optimizer = eval('torch.optim.' + optimizer)
#         # policy_lr = qf_lr if policy_lr is None or policy_lr < 0 else policy_lr # use shared lr if not provided.
#         # policy_lr = policy_lr if use_lr_scheduler else qf_lr*1e-3 # use shared lr if not provided.
#         ## ATAC parameters
#         self.beta = torch.Tensor([beta])  # regularization constant on the Bellman surrogate
#         self._lambd = torch.Tensor([lambd])  # global pessimism coefficient
#         self._init_observations = torch.Tensor(
#             init_observations) if init_observations is not None else init_observations  # if provided, it runs ATAC0
#         self._n_warmstart_steps = n_warmstart_steps  # during which, it performs independent C and Bellman minimization
#         # q update parameters
#         self._norm_constraint = norm_constraint  # l2 norm constraint on the qf's weight; if negative, it gives the weight decay coefficient.
#         self._q_eval_mode = [float(w) for w in
#                              q_eval_mode.split('_')] if '_' in q_eval_mode else q_eval_mode  # residual algorithm
#         self._q_eval_loss = eval('torch.nn.' + q_eval_loss)(reduction='none')
#         self._use_two_qfs = use_two_qfs
#         self.graph = graph  # the graph object
#         self._Vmin = Vmin  # lower bound on the target
#         self._Vmax = Vmax  # upper bound on the target
#         self._terminal_value = terminal_value if terminal_value is not None else lambda r, gamma: 0.

#         # Stepsizes
#         self._alpha_lr = qf_lr
#         self._bc_policy_lr = qf_lr  # potentially a larger stepsize

#         # Logging and algorithm state
#         self._debug = debug
#         self._n_updates_performed = 0  # Counter of number of grad steps performed
#         self._cac_learning = False
#         self._stats_avg_rate = stats_avg_rate
#         self._bellman_surrogate = bellman_surrogate
#         self._avg_bellman_error = 1.  # for logging; so this works with zero warm-start
#         self._avg_terminal_td_error = 1

#         #############################################################################################
#         # Original SAC parameters
#         self._qf1 = qf1
#         self._qf2 = qf2
#         self.replay_buffer = replay_buffer
#         self._tau = target_update_tau
#         self._policy_lr = policy_lr
#         self._qf_lr = qf_lr
#         self._initial_log_entropy = initial_log_entropy
#         self._gradient_steps = gradient_steps_per_itr
#         self._optimizer = optimizer
#         self._num_evaluation_episodes = num_evaluation_episodes
#         self._eval_env = eval_env
#         self._use_lr_scheduler = use_lr_scheduler
#         self._use_std_loss = use_std_loss
#         self._min_buffer_size = min_buffer_size
#         self._steps_per_epoch = steps_per_epoch
#         self._buffer_batch_size = buffer_batch_size
#         self._discount = discount
#         self._reward_scale = reward_scale
#         # self.max_episode_length = env_spec.max_episode_length
#         # self._max_episode_length_eval = env_spec.max_episode_length

#         if max_episode_length_eval is not None:
#             self._max_episode_length_eval = max_episode_length_eval
#         self._use_deterministic_evaluation = use_deterministic_evaluation

#         self.policy = policy
#         self.env_spec = env_spec
#         self._device = device
#         self.replay_buffer = replay_buffer
#         self._data_saving_dir = data_saving_dir
#         self._sampler = sampler

#         # use 2 target q networks
#         self._target_qf1 = copy.deepcopy(self._qf1)  # 初始target网络和q网络一样
#         self._target_qf2 = copy.deepcopy(self._qf2)
#         self._policy_optimizer = self._optimizer(self.policy.parameters(),
#                                                  lr=self._bc_policy_lr)  # lr for warmstart
#         self._qf1_optimizer = self._optimizer(self._qf1.parameters(),
#                                               lr=self._qf_lr)
#         self._qf2_optimizer = self._optimizer(self._qf2.parameters(),
#                                               lr=self._qf_lr)
#         if self._use_lr_scheduler:
#             # scheduler = lr_scheduler.ReduceLROnPlateau(self._policy_optimizer, mode='min', factor=0.5, patience=2, threshold=0.001)
#             scheduler = lr_scheduler.StepLR(self._policy_optimizer, step_size=50, gamma=0)
#             self._policy_lr_scheduler = scheduler
#         # automatic entropy coefficient tuning
#         self._use_automatic_entropy_tuning = fixed_alpha is None
#         self._fixed_alpha = fixed_alpha
#         if self._use_automatic_entropy_tuning:
#             if target_entropy:
#                 self._target_entropy = target_entropy
#             else:
#                 self._target_entropy = -np.prod(
#                     self.env_spec.action_space.shape).item()
#             self._log_alpha = torch.Tensor([self._initial_log_entropy
#                                             ]).requires_grad_()
#             self._alpha_optimizer = optimizer([self._log_alpha],
#                                               lr=self._alpha_lr)
#         else:
#             self._log_alpha = torch.Tensor([self._fixed_alpha]).log()
#         self.episode_rewards = deque(maxlen=30)

#     def optimize_policy(self,
#                         samples_data,
#                         warmstart=False):
#         """Optimize the policy q_functions, and temperature coefficient.

#         Note:
#             samples_data's entries should be torch.Tensor's with the following
#             shapes:
#                 observation: :math:`(N, O^*)`
#                 action: :math:`(N, A^*)`
#                 reward: :math:`(N, 1)`
#                 terminal: :math:`(N, 1)`
#                 next_observation: :math:`(N, O^*)`

#         Returns:
#             torch.Tensor: loss from actor/policy network after optimization.
#             torch.Tensor: loss from 1st q-function after optimization.
#             torch.Tensor: loss from 2nd q-function after optimization.

#         """
#         device = self._device

#         obs = samples_data['obs'].to(device)
#         next_obs = samples_data['next_obs'].to(device)
#         actions = samples_data['action'].to(device)

#         rewards = (samples_data['reward'] * self._reward_scale).to(device)
#         terminals = samples_data['terminal'].to(device)
#         # A = torch.tensor(samples_data['edge_index'].clone(), dtype=torch.long).to(device)
#         A = samples_data['edge_index'].clone().detach().to(dtype=torch.long, device=device)

#         # print("A=",A.shape)
#         # exit()

#         def compute_bellman_backup(q_pred_next):

#             assert rewards.shape == q_pred_next.shape
#             return rewards + (1. - terminals) * self._discount * q_pred_next + terminals * self._terminal_value(rewards,
#                                                                                                                 self._discount)

#         def compute_bellman_loss(q_pred, q_pred_next, q_target):

#             assert q_pred.shape == q_pred_next.shape == q_target.shape
#             target_error = self._q_eval_loss(q_pred,
#                                              q_target)  # MSE(Q_1(s,a),r+gamma*Q_target(s',a')) ,epsilon^{td}(f,f-min,pi)
#             q_target_pred = compute_bellman_backup(q_pred_next)  # r + (1-d)*gamma*Q_1(s',a') + d*V(s')*0.0
#             td_error = self._q_eval_loss(q_pred,
#                                          q_target_pred)  # MSE(Q_1(s,a),r+gamma*Q_1(s',a')) ,epsilon^{td}(f,f,pi)
#             w1, w2 = self._q_eval_mode  # w1=0.5 w2=0.5
#             bellman_loss = w1 * target_error + w2 * td_error  # epsilon^{td}(f,pi)  (7)
#             return bellman_loss, target_error, td_error

#         ##### Update Critic #####
#         ## Compute Bellman error
#         with torch.no_grad():

#             # 这里也用到了
#             new_next_actions = torch.softmax(self.policy(next_obs, A), dim=1)
#             target_q_values = torch.mean(
#                 self._target_qf1(new_next_actions, A), dim=(1, 2)
#             )
#             # target_q_values = torch.mean(
#             #     self._target_qf1(next_obs, A), dim=(1, 2)
#             # )
#             if self._use_two_qfs:
#                 target_q_values = torch.min(target_q_values,
#                                             torch.mean(
#                                                 self._target_qf2(new_next_actions
#                                                                  , A),
#                                                 dim=(1, 2)))
#                 # target_q_values = torch.min(target_q_values,
#                 #                             torch.mean(
#                 #                                 self._target_qf2(next_obs
#                 #                                                  , A),
#                 #                                 dim=(1, 2)))
#             q_target = compute_bellman_backup(target_q_values.flatten())

#         qf1_pred = torch.mean(self._qf1(actions, A), dim=(1, 2)).flatten()
#         # qf1_pred = torch.mean(self._qf1(obs, A), dim=(1, 2)).flatten()
#         qf1_pred_next = torch.mean(self._qf1(new_next_actions, A),
#                                     dim=(1, 2)).flatten()
#         # qf1_pred_next = torch.mean(self._qf1(next_obs, A),dim=(1, 2)).flatten()

#         qf1_bellman_losses, qf1_target_errors, qf1_td_errors = compute_bellman_loss(qf1_pred, qf1_pred_next, q_target)
#         qf1_bellman_loss = qf1_bellman_losses.mean()
#         qf2_bellman_loss = qf2_target_error = qf2_td_error = torch.Tensor([0.])

#         if self._use_two_qfs:
#             qf2_pred = torch.mean(self._qf2(actions, A), dim=(1, 2)).flatten()
#             qf2_pred_next = torch.mean(self._qf2(new_next_actions, A),
#                                        dim=(1, 2)).flatten()

#             # qf2_pred = torch.mean(self._qf2(obs, A), dim=(1, 2)).flatten()
#             # qf2_pred_next = torch.mean(self._qf2(next_obs, A),
#             #                            dim=(1, 2)).flatten()
#             qf2_bellman_losses, qf2_target_errors, qf2_td_errors = compute_bellman_loss(qf2_pred, qf2_pred_next,
#                                                                                         q_target)
#             qf2_bellman_loss = qf2_bellman_losses.mean()

#         # Compute GAN error
#         # These samples will be used for the actor update too, so they need to be traced.
#         # new_actions_pre_tanh, new_actions = new_actions_dist.rsample_with_pre_tanh_value() #_,a'=pi(s) ;第一项为取tanh前的值
#         # 注意这里用到了obs
#         new_actions = torch.softmax(self.policy(obs, A), dim=1)
#         # new_actions = torch.sigmoid(self.policy(obs, A))

#         # new_actions = new_actions_dist

#         gan_qf1_loss = gan_qf2_loss = 0
#         if not warmstart:

#             qf1_new_actions = torch.mean(self._qf1(new_actions.detach(), A),
#                                          dim=(1, 2))
#             # qf1_new_actions = torch.mean(self._qf1(obs, A),dim=(1, 2))

#             gan_qf1_loss = (qf1_new_actions * (1 + self._lambd) - qf1_pred).mean()

#             if self._use_two_qfs:
#                 qf2_new_actions = torch.mean(self._qf2(new_actions.detach(), A),
#                                              dim=(1, 2))
#                 # qf2_new_actions = torch.mean(self._qf2(obs, A),
#                 #                              dim=(1, 2))
#                 gan_qf2_loss = (qf2_new_actions * (1 + self._lambd) - qf2_pred).mean()

#         ## Compute full q loss
#         # We normalized the objective to prevent exploding gradients

#         with torch.no_grad():
#             beta = self.beta
#         qf1_loss = normalized_sum(gan_qf1_loss, qf1_bellman_loss, beta)
#         qf2_loss = normalized_sum(gan_qf2_loss, qf2_bellman_loss, beta)

#         if beta > 0 or not warmstart:  # 如果beta=0且warmstart阶段，不更新Q1,Q2；否则更新Q1,Q2
#             self._qf1_optimizer.zero_grad()
#             qf1_loss.backward()
#             self._qf1_optimizer.step()
#             self._qf1.apply(l2_projection(self._norm_constraint))

#             if self._use_two_qfs:
#                 self._qf2_optimizer.zero_grad()
#                 qf2_loss.backward()
#                 self._qf2_optimizer.step()
#                 self._qf2.apply(l2_projection(self._norm_constraint))  # 对Q2的每一层权重都进行l2正则化映射

#         ##### Update Actor #####

#         policy_entropy = (new_actions * torch.log2(new_actions + 1e-10)).sum(dim=2).mean()

#         alpha_loss = 0

#         if self._use_automatic_entropy_tuning:  # it comes first; seems to work also when put after policy update
#             alpha_loss = self._log_alpha * (policy_entropy.detach() - self._target_entropy)  # entropy - target
#             self._alpha_optimizer.zero_grad()  # 对log_alpha进行优化
#             alpha_loss.backward()
#             self._alpha_optimizer.step()

#         with torch.no_grad():
#             alpha = self._log_alpha.exp()

#         lower_bound = 0
#         if warmstart:  # BC warmstart

#             policy_log_prob = (actions * torch.log2(actions + 1e-10)).sum(dim=2).mean()

#             policy_loss = normalized_sum(-policy_log_prob.mean(), -policy_entropy, alpha)
#         else:
#             # Compute performance difference lower bound
#             min_q_new_actions = torch.mean(self._qf1(new_actions, A),
#                                            dim=(1, 2))  # Q_1(s,a')

#             policy_std_loss = torch.FloatTensor(np.array(0.)).to(device)

#             std_weight = 1e-3
#             if self._use_std_loss:
#                 print("use_std_loss")
#                 # policy_std_loss = compute_std_loss(ip_data_statistics)
#             #
#             policy_loss = normalized_sum(-min_q_new_actions.mean(), -policy_entropy,
#                                          alpha) + std_weight * policy_std_loss  # (-)

#         self._policy_optimizer.zero_grad()
#         policy_loss.backward()
#         self._policy_optimizer.step()

#         log_info = dict(
#             policy_loss=policy_loss,
#             policy_std_loss=policy_std_loss,
#             qf1_loss=qf1_loss,
#             qf2_loss=qf2_loss,
#             qf1_bellman_loss=qf1_bellman_loss,
#             gan_qf1_loss=gan_qf1_loss,
#             qf2_bellman_loss=qf2_bellman_loss,
#             gan_qf2_loss=gan_qf2_loss,
#             beta=beta,
#             alpha_loss=alpha_loss,
#             policy_entropy=policy_entropy,
#             alpha=alpha,
#             lower_bound=lower_bound,
#             policy_lr=self._policy_lr
#         )

#         # For logging
#         if self._debug:
#             with torch.no_grad():
#                 if self._bellman_surrogate == 'td':
#                     qf1_bellman_surrogate = qf1_td_errors.mean()
#                     qf2_bellman_surrogate = qf2_td_errors.mean()
#                 elif self._bellman_surrogate == 'target':
#                     qf1_bellman_surrogate = qf1_target_errors.mean()
#                     qf2_bellman_surrogate = qf2_target_errors.mean()
#                 elif self._bellman_surrogate is None:
#                     qf1_bellman_surrogate = qf1_bellman_loss
#                     qf2_bellman_surrogate = qf2_bellman_loss

#                 bellman_surrogate = torch.max(qf1_bellman_surrogate, qf2_bellman_surrogate)  # measure the TD error
#                 self._avg_bellman_error = self._avg_bellman_error * self._stats_avg_rate + bellman_surrogate * (
#                         1 - self._stats_avg_rate)

#                 if terminals.sum() > 0:
#                     terminal_td_error = (qf1_td_errors * terminals).sum() / terminals.sum()
#                     self._avg_terminal_td_error = self._avg_terminal_td_error * self._stats_avg_rate + terminal_td_error * (
#                             1 - self._stats_avg_rate)

#                 qf1_pred_mean = qf1_pred.mean()
#                 qf2_pred_mean = qf2_pred.mean() if self._use_two_qfs else 0.
#                 q_target_mean = q_target.mean()
#                 target_q_values_mean = target_q_values.mean()
#                 qf1_new_actions_mean = qf1_new_actions.mean() if not warmstart else 0.
#                 qf2_new_actions_mean = qf2_new_actions.mean() if not warmstart and self._use_two_qfs else 0.
#                 action_diff = torch.mean(torch.norm(actions - new_actions))

#             debug_log_info = dict(
#                 avg_bellman_error=self._avg_bellman_error,
#                 avg_terminal_td_error=self._avg_terminal_td_error,
#                 qf1_pred_mean=qf1_pred_mean,
#                 qf2_pred_mean=qf2_pred_mean,
#                 q_target_mean=q_target_mean,
#                 target_q_values_mean=target_q_values_mean,
#                 qf1_new_actions_mean=qf1_new_actions_mean,
#                 qf2_new_actions_mean=qf2_new_actions_mean,
#                 action_diff=action_diff,
#                 qf1_target_error=qf1_target_errors.mean(),
#                 qf1_td_error=qf1_td_errors.mean(),
#                 qf2_target_error=qf2_target_errors.mean(),
#                 qf2_td_error=qf2_td_errors.mean()
#             )
#             log_info.update(debug_log_info)

#         return new_actions, log_info

#     # Below is overwritten for general logging with log_info
#     def train(self, trainer):
#         """Obtain samplers and start actual training for each epoch.

#         Args:
#             trainer (Trainer): Gives the algorithm the access to
#                 :method:`~Trainer.step_epochs()`, which provides services
#                 such as snapshotting and sampler control.

#         Returns:
#             float: The average return in last epoch cycle.

#         """
#         if not self._eval_env:
#             self._eval_env = trainer.get_env_copy()
#         last_return = None
#         for ep in trainer.step_epochs():
#             for step in range(self._steps_per_epoch):  # 1
#                 for g_step in range(self._gradient_steps):  # ATAC的每一次更新,每一个epoch更新进行2000次梯度下降
#                     log_info = self.train_once()
#             if self._use_lr_scheduler:
#                 self._policy_lr_scheduler.step(log_info['policy_loss'])
#             if ep % 10 == 0:
#                 params = {}
#                 params['policy'] = self.policy
#                 params_file_name = os.path.join(self._data_saving_dir, 'epoch_%d.pkl' % ep)
#                 with open(params_file_name, 'wb') as f:
#                     cloudpickle.dump(params, f)
#             # if self._num_evaluation_episodes>0:
#             #     last_return = self._evaluate_policy(trainer.step_itr)
#             self._log_statistics(log_info)
#             tabular.record('TotalEnvSteps', trainer.total_env_steps)
#             trainer.step_itr += 1

#         return np.mean(last_return) if last_return is not None else 0

#     def train_once(self, itr=None, paths=None):
#         """Complete 1 training iteration of ATAC.

#         Args:
#             itr (int): Iteration number. This argument is deprecated.
#             paths (list[dict]): A list of collected paths.
#                 This argument is deprecated.

#         Returns:
#             torch.Tensor: loss from actor/policy network after optimization.
#             torch.Tensor: loss from 1st q-function after optimization.
#             torch.Tensor: loss from 2nd q-function after optimization.

#         """
#         del itr
#         del paths
#         if (self.replay_buffer.n_transitions_stored) >= self._min_buffer_size:
#             warmstart = self._n_updates_performed < self._n_warmstart_steps
#             if not warmstart and not self._cac_learning:
#                 self._cac_learning = True
#                 # Reset optimizers since the objective changes
#                 if self._use_automatic_entropy_tuning:
#                     self._log_alpha = torch.Tensor([self._initial_log_entropy]).to(
#                         self._log_alpha.device).requires_grad_()
#                     self._alpha_optimizer = self._optimizer([self._log_alpha], lr=self._alpha_lr)
#                 self._policy_optimizer = self._optimizer(self.policy.parameters(), lr=self._policy_lr)

#             samples = self.replay_buffer.sample_transitions_option(self._buffer_batch_size)
#             samples = as_torch_dict(samples)
#             # exit()

#             new_actions, log_info = self.optimize_policy(samples, warmstart=warmstart)
#             self._update_targets()
#             self._n_updates_performed += 1
#             log_info['n_updates_performed'] = self._n_updates_performed
#             log_info['warmstart'] = warmstart

#         return log_info

#     # Update also the target policy if needed
#     def _update_targets(self):  # 在每轮Q1,Q2,Actor更新后也要更新target网络
#         """Update parameters in the target q-functions."""
#         if self._use_two_qfs:
#             target_qfs = [self._target_qf1, self._target_qf2]
#             qfs = [self._qf1, self._qf2]
#         else:
#             target_qfs = [self._target_qf1]
#             qfs = [self._qf1]

#         for target_qf, qf in zip(target_qfs, qfs):  # 将[Q-target1,Q-target2]和[Q1,Q2]打包后，依次返回Q-target1,Q1;Q-target2,Q2
#             for t_param, param in zip(target_qf.parameters(), qf.parameters()):
#                 t_param.data.copy_(t_param.data * (
#                         1.0 - self._tau) + param.data * self._tau)  # 下划线结尾表示原地操作，不会创建一个新的副本。对于每一个参数都是Q-target = (1-tau)*Q-target + tau*Q

#     # Set also the target policy if needed
#     def to(self, device=None):
#         """Put all the networks within the model on device.

#         Args:
#             device (str): ID of GPU or CPU.

#         """
#         if device is None:
#             device = global_device()
#         for net in self.networks:
#             net.to(device)
#         if not self._use_automatic_entropy_tuning:
#             self._log_alpha = torch.Tensor([self._fixed_alpha
#                                             ]).log().to(device)
#         else:
#             self._log_alpha = torch.Tensor([self._initial_log_entropy
#                                             ]).to(device).requires_grad_()
#             self._alpha_optimizer = self._optimizer([self._log_alpha],
#                                                     lr=self._alpha_lr)
#         self.beta = torch.Tensor([self.beta]).to(device)
#         self._lambd = torch.Tensor([self._lambd]).to(device)

#     # Return also the target policy if needed
#     @property
#     def networks(self):
#         """Return all the networks within the model.

#         Returns:
#             list: A list of networks.

#         """
#         if self._use_two_qfs:
#             networks = [
#                 self.policy, self._qf1, self._qf2, self._target_qf1,self._target_qf2
#             ]
#         else:
#             networks = [
#                 self.policy, self._qf1, self._target_qf1
#             ]

#         return networks

#     # Evaluate both the deterministic and the stochastic policies
#     def _evaluate_policy(self, epoch):
#         """Evaluate the performance of the policy via deterministic sampling.

#             Statistics such as (average) discounted return and success rate are
#             recorded.

#         Args:
#             epoch (int): The current training epoch.

#         Returns:
#             float: The average return across self._num_evaluation_episodes
#                 episodes

#         """
#         eval_episodes = obtain_evaluation_episodes(  # TODO parallel evaluation
#             self.policy,
#             self._eval_env,
#             self._max_episode_length_eval,
#             num_eps=self._num_evaluation_episodes,
#             deterministic=self._use_deterministic_evaluation)  # return a episode batch
#         last_return = log_performance(epoch,
#                                       eval_episodes,
#                                       discount=self._discount)  # 在采样的episode batch上进行评价

#         return last_return

#     def _log_statistics(self, log_info):
#         """Record training statistics to dowel such as losses and returns.

#         Args:
#             policy_loss (torch.Tensor): loss from actor/policy network.
#             qf1_loss (torch.Tensor): loss from 1st qf/critic network.
#             qf2_loss (torch.Tensor): loss from 2nd qf/critic network.

#         """
#         for k, v in log_info.items():
#             tabular.record('Algorithm/' + k, float(v))
#         tabular.record('ReplayBuffer/buffer_size', self.replay_buffer.n_transitions_stored)
