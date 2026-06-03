import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
import matplotlib.pyplot as plt
import time

# --- Hyperparameters ---
ENV_NAME = "CarRacing-v3"
EPISODES = 200
MAX_STEPS = 1000
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 64

# DDPG
DDPG_ACTOR_LR = 1e-4
DDPG_CRITIC_LR = 1e-3
DDPG_BUFFER_SIZE = 10000
DDPG_NOISE_STD = 0.1

# PPO
PPO_LR = 2e-4
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 6
PPO_GAE_LAMBDA = 0.95
PPO_ENT_COEF = 0.02
PPO_VF_COEF = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Preprocessing ---
def preprocess(obs):
    gray = np.mean(obs, axis=2) / 255.0
    return gray.astype(np.float32)  # (96, 96)

# --- Shared CNN Base ---
class CNNBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        dummy = torch.zeros(1, 1, 96, 96)
        self.out_dim = self.cnn(dummy).shape[1]

    def forward(self, x):
        return self.cnn(x)

# ============================================================
# DDPG
# ============================================================
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)

class DDPGActor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.cnn = CNNBase()
        self.net = nn.Sequential(
            nn.Linear(self.cnn.out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(self.cnn(x))

class DDPGCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.cnn = CNNBase()
        self.net = nn.Sequential(
            nn.Linear(self.cnn.out_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([self.cnn(state), action], dim=1))

class DDPGAgent:
    def __init__(self, action_dim):
        self.actor = DDPGActor(action_dim).to(device)
        self.actor_target = DDPGActor(action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = DDPGCritic(action_dim).to(device)
        self.critic_target = DDPGCritic(action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=DDPG_ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=DDPG_CRITIC_LR)
        self.buffer = ReplayBuffer(DDPG_BUFFER_SIZE)
        self.actor_losses = []
        self.critic_losses = []

    def select_action(self, state, noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        action = self.actor(state_tensor).detach().cpu().numpy()[0]
        if noise:
            action += np.random.normal(0, DDPG_NOISE_STD, size=action.shape)
        return np.clip(action, -1, 1)

    def train(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + GAMMA * (1 - dones) * self.critic_target(next_states, next_actions)
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())

def run_ddpg():
    print("\n========== Running DDPG ==========")
    env = gym.make(ENV_NAME)
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(action_dim)
    episode_rewards = []
    start = time.time()

    for episode in range(EPISODES):
        obs, _ = env.reset()
        state = preprocess(obs)
        total_reward = 0
        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess(next_obs)
            done = terminated or truncated
            agent.buffer.add(state, action, reward, next_state, float(done))
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                break
        episode_rewards.append(total_reward)
        print(f"DDPG Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f}")

    env.close()
    elapsed = time.time() - start
    print(f"DDPG finished in {elapsed:.1f}s | Avg reward: {np.mean(episode_rewards):.2f} | Best: {max(episode_rewards):.2f}")
    return episode_rewards, agent.actor_losses, agent.critic_losses

# ============================================================
# PPO
# ============================================================
class PPOActorCritic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.cnn = CNNBase()
        self.shared = nn.Sequential(
            nn.Linear(self.cnn.out_dim, 512),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(512, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.shared(self.cnn(x))
        mean = torch.tanh(self.actor_mean(x))
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(x)
        return mean, std, value

    def get_action(self, state):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), log_prob, value.squeeze()

    def evaluate(self, states, actions):
        mean, std, value = self.forward(states)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value.squeeze(), entropy

def compute_gae(rewards, values, dones, next_value):
    advantages = []
    gae = 0
    values = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * PPO_GAE_LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages, losses):
    states = torch.FloatTensor(np.array(states)).unsqueeze(1).to(device)
    actions = torch.FloatTensor(np.array(actions)).to(device)
    old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(device)
    returns = torch.FloatTensor(np.array(returns)).to(device)
    advantages = torch.FloatTensor(np.array(advantages)).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), BATCH_SIZE):
            idx = indices[start:start+BATCH_SIZE]
            log_probs, values, entropy = model.evaluate(states[idx], actions[idx])
            ratio = (log_probs - old_log_probs[idx]).exp()
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * advantages[idx]
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns[idx])
            entropy_loss = -entropy.mean()
            loss = actor_loss + PPO_VF_COEF * critic_loss + PPO_ENT_COEF * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())

def run_ppo():
    print("\n========== Running PPO ==========")
    env = gym.make(ENV_NAME)
    action_dim = env.action_space.shape[0]
    model = PPOActorCritic(action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=PPO_LR)
    episode_rewards = []
    losses = []
    start = time.time()

    for episode in range(EPISODES):
        obs, _ = env.reset()
        state = preprocess(obs)
        total_reward = 0
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        for _ in range(MAX_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                action, log_prob, value = model.get_action(state_tensor)
            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            next_state = preprocess(next_obs)
            done = terminated or truncated
            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(float(done))
            log_probs.append(log_prob.cpu().item())
            values.append(value.cpu().item())
            state = next_state
            total_reward += reward
            if done:
                break

        with torch.no_grad():
            _, _, next_value = model.get_action(torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device))
        advantages, returns = compute_gae(rewards, values, dones, next_value.cpu().item())
        ppo_update(model, optimizer, states, actions, log_probs, returns, advantages, losses)
        episode_rewards.append(total_reward)
        print(f"PPO Episode {episode+1}/{EPISODES} | Reward: {total_reward:.2f}")

    env.close()
    elapsed = time.time() - start
    print(f"PPO finished in {elapsed:.1f}s | Avg reward: {np.mean(episode_rewards):.2f} | Best: {max(episode_rewards):.2f}")
    return episode_rewards, losses

# ============================================================
# Plotting
# ============================================================
def plot_results(ddpg_rewards, ddpg_actor_losses, ddpg_critic_losses, ppo_rewards, ppo_losses):
    episodes = range(1, EPISODES + 1)
    ma = lambda x, n=5: np.convolve(x, np.ones(n)/n, mode='valid')

    # Comparison plot
    plt.figure(figsize=(12, 5))
    plt.plot(episodes, ddpg_rewards, alpha=0.3, color='blue')
    plt.plot(episodes, ppo_rewards, alpha=0.3, color='orange')
    plt.plot(range(5, EPISODES+1), ma(ddpg_rewards), label='DDPG (5-ep MA)', color='blue', linewidth=2)
    plt.plot(range(5, EPISODES+1), ma(ppo_rewards), label='PPO (5-ep MA)', color='orange', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DDPG vs PPO on CarRacing-v3")
    plt.legend()
    plt.savefig("/home/ubuntu/persistent/benchmark_comparison.png")
    plt.close()

    # DDPG individual plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(ddpg_actor_losses, alpha=0.7, label='Actor Loss')
    plt.title("DDPG Actor Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(ddpg_critic_losses, alpha=0.7, color='red', label='Critic Loss')
    plt.title("DDPG Critic Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/ubuntu/persistent/benchmark_ddpg_losses.png")
    plt.close()

    # PPO loss plot
    plt.figure(figsize=(10, 4))
    plt.plot(ppo_losses, alpha=0.7, color='orange', label='PPO Loss')
    plt.title("PPO Loss over Training Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("/home/ubuntu/persistent/benchmark_ppo_loss.png")
    plt.close()

    print("\nAll plots saved!")
    print(f"  benchmark_comparison.png")
    print(f"  benchmark_ddpg_losses.png")
    print(f"  benchmark_ppo_loss.png")

# ============================================================
# Summary
# ============================================================
def print_summary(ddpg_rewards, ppo_rewards):
    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"{'Algorithm':<10} {'Avg Reward':<15} {'Best Reward':<15} {'Worst Reward':<15}")
    print("-" * 55)
    print(f"{'DDPG':<10} {np.mean(ddpg_rewards):<15.2f} {max(ddpg_rewards):<15.2f} {min(ddpg_rewards):<15.2f}")
    print(f"{'PPO':<10} {np.mean(ppo_rewards):<15.2f} {max(ppo_rewards):<15.2f} {min(ppo_rewards):<15.2f}")
    print("========================================")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    ddpg_rewards, ddpg_actor_losses, ddpg_critic_losses = run_ddpg()
    ppo_rewards, ppo_losses = run_ppo()
    plot_results(ddpg_rewards, ddpg_actor_losses, ddpg_critic_losses, ppo_rewards, ppo_losses)
    print_summary(ddpg_rewards, ppo_rewards)
