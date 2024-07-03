import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from config import experi_dir
import random
import config

from math import sqrt


np.random.seed(1337)
random.seed(1337)


def setup_seed(seed):
    T.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(1337)

mach_mask = None

class PPOMemory:
    def __init__(self, batch_size):

        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i: i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class SharedNetwork(nn.Module):
    def __init__(self, input_size, alpha, dim_k=128, chkpt_dir=os.path.join(experi_dir(), 'param')):
        super(SharedNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'shared_network_ppo.pt')

        self.feature_emb = nn.Sequential(
            nn.Linear(input_size, dim_k),
            T.nn.ReLU(),
            nn.Linear(dim_k, dim_k),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        out = self.feature_emb(state)
        return out

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class SelfAttention(nn.Module):
    def __init__(self, input_size, embed_size=128, num_heads=config.NUM_Head, mask=mach_mask):
        super(SelfAttention, self).__init__()
        self.mask = mask
        self.emb_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (self.head_dim * num_heads == embed_size), "Embedding size needs to be divisible by heads"

        """2 Layer Attention"""
        self.q_linear = nn.Sequential(
            nn.Linear(input_size, embed_size),
            T.nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )
        self.k_linear = nn.Sequential(
            nn.Linear(input_size, embed_size),
            T.nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )
        self.v_linear = nn.Sequential(
            nn.Linear(input_size, embed_size),
            T.nn.ReLU(),
            nn.Linear(embed_size, embed_size),
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 2 * embed_size),
            nn.ReLU(),
            nn.Linear(2 * embed_size, embed_size)
        )

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.scale_fkt = 1 / sqrt(embed_size)

    def forward(self, state_emb):
        n = state_emb.shape[0]  # number of samples
        dim = state_emb.shape[1]  # dimension of q, k, v

        queries = self.q_linear(state_emb)
        keys = self.k_linear(state_emb)
        values = self.v_linear(state_emb)

        queries = queries.reshape(n, dim, self.num_heads, self.head_dim)
        keys = keys.reshape(n, dim, self.num_heads, self.head_dim)
        values = values.reshape(n, dim, self.num_heads, self.head_dim)

        # n: number of samples
        # q,k: dim of q and k, which is same here
        # h: number of heads
        # d: head dim, equal to emb_size//num_head
        energy = T.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if self.mask is not None:
            energy = energy.masked_fill(self.mask, float("-1e20"))

        attention_alpha = energy * self.scale_fkt
        attention_a = T.softmax(attention_alpha, dim=3)

        # attention times values
        attention = T.einsum("nhql,nlhd->nqhd", [attention_a, values]).reshape(n, dim, self.num_heads * self.head_dim)
        queries = queries.reshape(n, dim, self.num_heads * self.head_dim)

        # add norm
        x = self.norm1(attention + queries)
        forward = self.feed_forward(x)
        out = self.norm2(x + forward)
        return out


class ActorNetwork(nn.Module):

    def __init__(self, input_size, dim_k=128, dim_v=128, num_heads=config.NUM_Head, chkpt_dir=os.path.join(experi_dir(), 'param')):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo.pt')

        # self.emb = SharedNetwork(input_size, dim_k)

        self.attention = SelfAttention(input_size, dim_k, num_heads)
        self.attention_2 = SelfAttention(dim_k, dim_k, num_heads)
        # self.attention_3 = SelfAttention(dim_k, num_heads)

        self.fc_out = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            nn.ReLU(),
            nn.Linear(dim_v, 1)
        )

        # self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)



    def forward(self, state):
        # x = self.emb(state)
        x = self.attention(state)
        x = self.attention_2(x)
        # x = self.attention_3(x)
        prio = self.fc_out(x)
        dist = T.squeeze(prio)  # squeeze from 3-dim to 1-dim
        # if the shape is (1, 1, 1), matrix will be squeezed to 0 dim
        if dist.dim() == 0:
            dist = T.unsqueeze(dist, 0)
        # mask
        mask = T.zeros(state.shape[1]).to(self.device)
        for i in range(config.NUM_Machs):
            mask[-1-i] = 1
        mask = mask.to(T.bool)
        dist = dist.masked_fill(mask, float("-1e20"))
        dist = T.softmax(dist, dim=-1)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def learning_rate_reduce(self):
        self.alpha = self.alpha / config.learning_rate_decay


class CriticNetwork(nn.Module):
    def __init__(self, input_size, dim_k=128, dim_v=128, num_heads=config.NUM_Head, chkpt_dir=os.path.join(experi_dir(), 'param')):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo.pt')

        # self.emb = SharedNetwork(input_size, dim_k)

        self.attention = SelfAttention(input_size, dim_k, num_heads)
        self.attention_2 = SelfAttention(dim_k, dim_k, num_heads)
        # self.attention_3 = SelfAttention(dim_k, num_heads)

        self.fc_out = nn.Sequential(
            nn.Linear(dim_v, dim_v),
            T.nn.ReLU(),
            nn.Linear(dim_v, 1),
        )

        self.scale_fkt = 1 / sqrt(dim_k)

        # self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):

        # x = self.emb(state)

        x = self.attention(state)
        x = self.attention_2(x)
        # x = self.attention_3(x)

        # x = self.mlp(state)

        value = T.sum(x, dim=1)  # value is a vector now
        value = self.fc_out(value)
        value = T.unsqueeze(value, 0)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

    def learning_rate_reduce(self):
        self.alpha = self.alpha / config.learning_rate_decay


class Agent:
    def __init__(self, input_size, gamma=0.99, alpha=0.00001, gae_lambda=0.98, policy_clip=0.2,
                 batch_size=5, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.lr = alpha

        self.actor = ActorNetwork(input_size).to('cuda:0' if T.cuda.is_available() else 'cpu')
        self.critic = CriticNetwork(input_size).to('cuda:0' if T.cuda.is_available() else 'cpu')
        # self.emb = SharedNetwork(input_size, alpha).to('cuda:0' if T.cuda.is_available() else 'cpu')
        self.memory = PPOMemory(batch_size)

        self.optimizer_actor = T.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = T.optim.Adam(self.critic.parameters(), lr=self.lr)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def learning_rate_change(self):
        self.lr = self.lr / config.learning_rate_decay
        for p in self.optimizer_actor.param_groups:
            p['lr'] = self.lr
        for p in self.optimizer_critic.param_groups:
            p['lr'] = self.lr

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        # self.emb.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        # self.emb.load_checkpoint()

    def choose_action(self, observation):
        state_ary = np.array([observation])
        state = T.tensor(state_ary, dtype=T.float).to(self.actor.device)
        # state = self.emb(state)

        dist = self.actor(state)

        value = self.critic(state)
        # in the validation brain, the action are not sampled
        # action = dist.sample()
        # On the contrary, the action with the highest prob will be directly selected
        action = T.argmax(dist.probs)
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 0.98
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])

                    discount *= self.gamma * self.gae_lambda

                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                action_batch = T.tensor(action_arr[batch]).to(self.actor.device)
                old_prob_batch = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                state_batch = []
                for state in state_arr[batch]:
                    state_batch.append(T.tensor(state, dtype=T.float, requires_grad=True).to(self.actor.device))
                state_pad = pad_sequence(state_batch, batch_first=True)

                dist_batch = self.actor(state_pad)

                new_probs_batch = dist_batch.log_prob(action_batch)

                prob_ratio = new_probs_batch.exp() / old_prob_batch.exp()

                value_batch = self.critic(state_pad)
                value_batch = T.squeeze(value_batch)
                # print("value batch", value_batch.is_leaf)

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - value_batch) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + critic_loss
                # self.actor.optimizer.zero_grad()
                self.optimizer_actor.zero_grad()
                # self.critic.optimizer.zero_grad()
                self.optimizer_critic.zero_grad()

                total_loss.backward()

                self.optimizer_actor.step()

                self.optimizer_critic.step()

        self.memory.clear_memory()






