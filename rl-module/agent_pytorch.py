'''
  MIT License
  Copyright (c) Chen-Yu Yen 2020

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
'''
import torch.nn as nn
import torch
from torch.optim import Adam
import itertools

# import tensorflow as tf
import numpy as np
import os
import time

EXPLORE = 4000
STDDEV = 0.1
NSTEP = 0.3

# TODO: Inference of the networks should add model.eval()


from utils import OU_Noise, ReplayBuffer, G_Noise, Prioritized_ReplayBuffer

def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)


# build the actor network
class ActorNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, h1_shape, h2_shape, action_scale=1.0):
        super(ActorNetwork, self).__init__()
        self.action_scale = action_scale
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1_shape = h1_shape
        self.h2_shape = h2_shape

        self.fc1 = nn.Linear(self.s_dim, self.h1_shape)
        self.fc2 = nn.Linear(self.h1_shape, self.h2_shape)
        self.fc3 = nn.Linear(self.h2_shape, self.a_dim)

        # batch normalization
        self.bn1 = nn.BatchNorm1d(self.h1_shape, affine=False) # equals to tf scale=False
        self.bn2 = nn.BatchNorm1d(self.h2_shape, affine=False)

        # ReLU
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2) # 0.2 is the tf default value
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2)

        # self.leakyrelu1 = nn.ReLU()
        # self.leakyrelu2 = nn.ReLU()

    # TODO: did not assign names to the layers, e.g. fc1, fc2, fc3
    def forward(self, s):
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        h1 = self.fc1(s)
        h1 = self.bn1(h1)
        h1 = self.leakyrelu1(h1)

        h2 = self.fc2(h1)
        h2 = self.bn2(h2)
        h2 = self.leakyrelu2(h2)

        output = self.fc3(h2)
        scale_output = torch.tanh(output) * self.action_scale
        
        return scale_output


class CriticNetwork(nn.Module):
    def __init__(self, s_dim, a_dim, h1_shape, h2_shape, action_scale=1.0):
        super(CriticNetwork, self).__init__()
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.h1_shape = h1_shape
        self.h2_shape = h2_shape
        self.action_scale = action_scale

        self.fc1 = nn.Linear(self.s_dim, self.h1_shape)
        self.fc2 = nn.Linear(self.h1_shape + self.a_dim, self.h2_shape)
        self.fc3 = nn.Linear(self.h2_shape, 1)

        # ReLU
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2) # 0.2 is the tf default value
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2)

        # self.leakyrelu1 = nn.ReLU()
        # self.leakyrelu2 = nn.ReLU()
    
    def forward(self, s, action):
        h1 = self.fc1(s)
        h1 = self.leakyrelu1(h1)

        h2 = self.fc2(torch.cat([h1, action], -1))
        h2 = self.leakyrelu2(h2)

        output = self.fc3(h2)

        return output


class Agent():
    def __init__(self, s_dim, a_dim, h1_shape, h2_shape, gamma=0.995, batch_size=8, lr_a=1e-4, lr_c=1e-3, tau=1e-3, mem_size=1e5, action_scale=1.0, action_range=(-1.0, 1.0),
                noise_type=3, noise_exp=50000, summary=None,stddev=0.1, PER=False, alpha=0.6, CDQ=True, LOSS_TYPE='HUBERT', device='cpu',
                train_dir=None, mix_env=None, num_actors=None):
        
        self.mix_env = mix_env
        
        self.PER = PER
        self.CDQ = CDQ # True by default
        self.LOSS_TYPE = LOSS_TYPE
        # TODO: update
        # self.LOSS_TYPE = 'MSE'
        # self.lr_a = lr_a
        # self.lr_c = lr_c

        self.lr_a = lr_a / 5
        self.lr_c = lr_c / 5

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.gamma = gamma

        # TODO: to update
        self.device = device    

        self.tau = tau
        self.train_dir = f"{train_dir}/trained_model" #'./rl-module/pytorch_train_dir/trained_model'

        self.step_epochs = 0
        # TODO, recheck the usage of the global steps
        self.global_step = 0
        # TODO 
        # TODO
        # self.step_epochs = tf.Variable(0, trainable=False, name='epoch')
        # self.global_step = tf.train.get_or_create_global_step(graph=None)

        # TODO: continue
        # TODO: create the input
        # self.s0 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s0')
        # self.s1 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s1')
        # self.is_training = tf.placeholder(tf.bool, name='Actor_is_training')
        # self.action = tf.placeholder(tf.float32, shape=[None, a_dim], name='action')

        # TODO:
        # add sac_buffer
        # add the replay_buffer (keep the same)
        # contain a list of models (actor only has one model, the learner has multiple models)
        # each learner has a list of models
        # Learner
        # Make the decision about which buffer/environment to use
        # Then add that environment's sac buffer to the 
        # in original loss: just sample from the replay buffer
        # 

        self.noise_type = noise_type
        self.noise_exp = noise_exp
        self.action_range = action_range
        self.h1_shape=h1_shape
        self.h2_shape=h2_shape
        self.stddev=stddev
        if not self.mix_env:
            self.rp_buffer_list = []
            if not self.PER:
                for i in range(num_actors):
                    self.rp_buffer_list.append(ReplayBuffer(int(mem_size/num_actors), s_dim, a_dim, batch_size=batch_size))
            else:
                for i in range(num_actors):
                    self.rp_buffer_list.append(Prioritized_ReplayBuffer(int(mem_size/num_actors), s_dim, a_dim, batch_size=batch_size, alpha=alpha))
            for i in range(num_actors):
                self.dynamics_model_list.append(
                    
                )
        else:
            if not self.PER:
                self.rp_buffer = ReplayBuffer(int(mem_size), s_dim, a_dim, batch_size=batch_size)
            else:
                self.rp_buffer = Prioritized_ReplayBuffer(int(mem_size), s_dim, a_dim, batch_size=batch_size, alpha=alpha)

        if noise_type == 1:
            self.actor_noise = OU_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), dt=1, exp=self.noise_exp)
        elif noise_type == 2:
            ## Gaussian with gradually decay
            self.actor_noise = G_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), explore =self.noise_exp)
        elif noise_type == 3:
            ## Gaussian without gradually decay
            self.actor_noise = G_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), explore = None,theta=0.1)
        elif noise_type == 4:
            ## Gaussian without gradually decay
            self.actor_noise = G_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), explore = EXPLORE, theta=0.1, mode="step", step=NSTEP)
        elif noise_type == 5:
            self.actor_noise = None
        else:
            self.actor_noise = OU_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), dt=0.5)

        # Main Actor/Critic Network
        self.actor = ActorNetwork(self.s_dim, self.a_dim, action_scale=action_scale, h1_shape=self.h1_shape, h2_shape=self.h2_shape).to(self.device)
        self.critic = CriticNetwork(self.s_dim, self.a_dim, action_scale=action_scale, h1_shape=self.h1_shape, h2_shape=self.h2_shape).to(self.device)
        self.critic2 = CriticNetwork(self.s_dim, self.a_dim, action_scale=action_scale, h1_shape=self.h1_shape, h2_shape=self.h2_shape).to(self.device)

        # Target Actor/Critic network
        self.target_actor = ActorNetwork(self.s_dim, self.a_dim, action_scale=action_scale, h1_shape=self.h1_shape,h2_shape=self.h2_shape).to(self.device)
        self.target_critic = CriticNetwork(self.s_dim, self.a_dim, action_scale=action_scale ,h1_shape=self.h1_shape,h2_shape=self.h2_shape).to(self.device)
        self.target_critic2 = CriticNetwork(self.s_dim, self.a_dim, action_scale=action_scale, h1_shape=self.h1_shape,h2_shape=self.h2_shape).to(self.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_a)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_c)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=self.lr_c)
        self.critic_optim_all = Adam(itertools.chain(*[self.critic.parameters(), self.critic2.parameters()]), lr=self.lr_c)

        self.terminal = None
        self.reward = None
        self.y = None
        self.y2 = None
        self.importance = None
        self.td_error = None

        # todo: remove these placeholder
        # self.td_error = self.critic_out - self.y

        # todo: pytorch logger
        # self.summary_writer = summary
        
        self.a_loss = None

    def train_actor(self, s0, is_training=True):
        # todo: omit the global step for now
        actor_out = self.actor(s0) # when eval, the network is in eval mode
        critic_actor_out = self.critic(s0, actor_out)
        a_loss = - torch.mean(critic_actor_out)
        self.actor_optim.zero_grad()
        a_loss.backward()
        self.actor_optim.step()
        self.a_loss = a_loss.item()

    def train_critic(self, s0, action, reward, s1, terminal, is_training=True, importance=False):
        use_huber = True
        if use_huber:
            # def f1(y, pred, weights=1.0):
            #     error = torch.square(y - pred)
            #     weighted_error = torch.mean(error * weights)
            #     return weighted_error
            
            loss_function = {
                'HUBER': nn.HuberLoss(reduction='mean'), # the pytorch default reduction is 'mean'; the tf default reduction is 'sum'
                'MSE': nn.MSELoss(reduction='mean') # f1
            }
            
            if self.CDQ:
                target_actor_out = self.target_actor(s1)
                eps = torch.randn_like(target_actor_out) * 0.1
                eps = torch.clamp(eps, -0.2, 0.2)
                # encourage exploration
                # eps = torch.randn_like(target_actor_out) * 0.2
                # eps = torch.clamp(eps, -0.25, 0.25)

                t_a = target_actor_out + eps
                t_a = torch.clamp(t_a, -1.0, 1.0)

                target_critic_actor_out = self.target_critic(s1, t_a)
                target_critic_actor_out2 = self.target_critic2(s1, t_a)
                y = reward + self.gamma * (1 - terminal) * target_critic_actor_out
                y2 = reward + self.gamma * (1 - terminal) * target_critic_actor_out2
                q_min_target = torch.min(y, y2)

                critic_out = self.critic(s0, action)
                critic_out2 = self.critic2(s0, action)
                if self.PER:
                    c_loss = loss_function[self.LOSS_TYPE](q_min_target, critic_out, weights=importance)
                    c_loss2 = loss_function[self.LOSS_TYPE](q_min_target, critic_out2, weights=importance)
                else:
                    c_loss = loss_function[self.LOSS_TYPE](q_min_target, critic_out)
                    c_loss2 = loss_function[self.LOSS_TYPE](q_min_target, critic_out2)
            else:
                NotImplementedError(f"CDQ is False, not implemented yet.")
        else:
            NotImplementedError(f"use_huber is False, not implemented yet.")
        
        self.critic_optim.zero_grad()
        c_loss.backward(retain_graph=True)
        self.critic_optim.step()
        self.critic2_optim.zero_grad()
        c_loss2.backward()
        self.critic2_optim.step()

        # total_c_loss = c_loss + c_loss2
        # self.critic_optim_all.zero_grad()
        # total_c_loss.backward()
        # self.critic_optim_all.step()

    # todo: add pytorch logger
    # def create_tf_summary(self):
    #     if self.CDQ:
    #         tf.summary.scalar('Loss/critic_loss:', self.c_loss)
    #         tf.summary.scalar('Loss/critic_loss_2:', self.c_loss2)
    #     else:
    #         tf.summary.scalar('Loss/critic_loss:', self.critic_loss)

    #     tf.summary.scalar('Loss/actor_loss:', self.a_loss)

    #     self.summary_op = tf.summary.merge_all()

    def init_target(self):
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        self.hard_update(self.target_critic2, self.critic2)

    # NO use
    # def assign_sess(self, sess):
    #     self.sess = sess
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def target_update(self):
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)
        self.soft_update(self.target_critic2, self.critic2, self.tau)

    # TODO: no use?
    # def actor_clone_update(self):
    #     self.sess.run(self.actor_clone_update_op)

    def get_action(self, s, use_noise=True):
        with torch.no_grad(): # is_training=False
            s0 = torch.FloatTensor(s).to(self.device)
            # print(f"s0 shape: {s0.shape}")
            action = self.actor(s0)
            if use_noise:
                action = action.cpu().numpy()
                noise = self.actor_noise(action)
                action += noise
                action = np.clip(action, self.action_range[0], self.action_range[1])
            return action

    # TODO: NOT USE?
    def get_q(self, s, a):
        s0 = torch.FloatTensor(s).to(self.device)
        action = torch.FloatTensor(a).to(self.device)
        critic_out = self.critic(s0, action)
        return critic_out

    def get_q_actor(self, s):
        s0 = torch.FloatTensor(s).to(self.device)
        actor_out = self.actor(s0)
        critic_out = self.critic(s0, actor_out)
        return critic_out

    def store_experience(self, s0, a, r, s1, terminal, idx=None):
        # if different env uses different environments
        if not self.mix_env:
            self.rp_buffer_list[idx].store(s0, a, r, s1, terminal)
        else:
            self.rp_buffer.store(s0, a, r, s1, terminal)

    def store_many_experience(self, s0, a, r, s1, terminal, length, idx=None):
        if not self.mix_env:
            if self.PER:
                for i in range(length):
                    self.rp_buffer_list[idx].store(s0[i], a[i], r[i], s1[i], terminal[i])
            else:
                self.rp_buffer_list[idx].store_many(s0, a, r, s1, terminal, length)
        else:
            if self.PER:
                for i in range(length):
                    self.rp_buffer.store(s0[i], a[i], r[i], s1[i], terminal[i])
            else:
                self.rp_buffer.store_many(s0, a, r, s1, terminal, length)

    def sample_experince(self, idx=None):
        if not self.mix_env:
            return self.rp_buffer_list[idx].sample()
        else:
            return self.rp_buffer.sample()

    def train_step_td(self):
        return None

    def train_step(self, idx=None):
        if self.PER: # Orca sets PER to False by default
            NotImplementedError(f"PER True is not implemented for {self.__class__.__name__}.")
            # _, td_errors = self.sess.run([self.critic_train_op, self.td_error], feed_dict=fd)\# if self.PER:
            # if self.PER:
            #     new_priorities = np.abs(np.squeeze(td_errors)) + 1e-6
            #     self.rp_buffer.update_priorities(idxes, new_priorities)
        else:
            if not self.mix_env:
                (
                    s0_batch,
                    action_batch,
                    reward_batch,
                    s1_batch,
                    terminal_batch,
                ) = self.rp_buffer_list[idx].sample() 
            else:
                (
                    s0_batch,
                    action_batch,
                    reward_batch,
                    s1_batch,
                    terminal_batch,
                ) = self.rp_buffer.sample() 
        
        # print(f"s0_batch.shape: {s0_batch.shape}")
        s0_batch = torch.FloatTensor(s0_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        s1_batch = torch.FloatTensor(s1_batch).to(self.device)
        terminal_batch = torch.FloatTensor(terminal_batch).to(self.device)

        # ---------------------- optimize critic ----------------------
        self.train_critic(s0_batch, 
            action_batch, 
            reward_batch, 
            s1_batch, 
            terminal_batch, 
            is_training=True)

        # ---------------------- optimize actor ----------------------
        self.train_actor(s0_batch, is_training=True)

        # TODO: torch distributed logger
        # summary, step = self.sess.run([self.summary_op, self.global_step], feed_dict=fd)
        # self.summary_writer.add_summary(summary, global_step=step)

    # TODO: torch distributed logger
    # def log_tf(self, val, tag=None, step_counter=0):
    #     summary = tf.Summary()
    #     summary.value.add(tag= tag, simple_value=val)
    #     self.summary_writer.add_summary(summary, step_counter)

    def save_model(self, step=None):
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        ckpt_path = f"{self.train_dir}/model.pth"
        # print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "target_actor_state_dict": self.target_actor.state_dict(),
                "target_critic_state_dict": self.target_critic.state_dict(),
                "target_critic2_state_dict": self.target_critic2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optim.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "critic2_optimizer_state_dict": self.critic2_optim.state_dict(),
            },
            ckpt_path,
        )

    def load_model(self, ckpt_path, evaluate=False):
        # print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
            self.target_actor.load_state_dict(checkpoint["target_actor_state_dict"])
            self.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
            self.target_critic2.load_state_dict(checkpoint["target_critic2_state_dict"])
            self.actor_optim.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.critic2_optim.load_state_dict(checkpoint["critic2_optimizer_state_dict"])
        if evaluate:
            self.actor.eval()
            self.critic.eval()
            self.critic2.eval()
            self.target_actor.eval()
            self.target_critic.eval()
            self.target_critic2.eval()
        else:
            self.actor.train()
            self.critic.train()
            self.critic2.train()
            self.target_actor.train()
            self.target_critic.train()
            self.target_critic2.train()

    # TODO: update function's name -> update_step_epochs
    def updat_step_epochs(self, epoch):
        self.step_epochs = epoch

    def get_step_epochs(self):
        return self.step_epochs
