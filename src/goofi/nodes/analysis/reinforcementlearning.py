import numpy as np

from goofi.data import Data, DataType
from goofi.node import Node
from goofi.params import BoolParam, FloatParam, IntParam, StringParam


class ReinforcementLearning(Node):
    """
    Real-time Reinforcement Learning node using Proximal Policy Optimization (PPO).

    This node trains a continuous action policy in real-time using PPO, a state-of-the-art
    policy gradient method. Each call to process() represents one timestep: the node receives
    observations and the reward from the previous action, then outputs a new action vector.

    The agent learns to maximize cumulative rewards through interaction, with a configurable
    MLP policy network and action space dimensionality.

    Inputs:
    - observations: m-dimensional vector of floats representing the current state/observation
    - reward: single float value representing the reward for the previous action

    Outputs:
    - actions: n-dimensional vector of continuous action values (floats in range [-1, 1])
    - value: estimated value of the current state (useful for monitoring learning progress)
    """

    @staticmethod
    def config_input_slots():
        return {
            "observations": DataType.ARRAY,
            "reward": DataType.ARRAY,
        }

    @staticmethod
    def config_output_slots():
        return {
            "actions": DataType.ARRAY,
            "value": DataType.ARRAY,
        }

    @staticmethod
    def config_params():
        return {
            "architecture": {
                "observation_dim": IntParam(4, 1, 256, doc="Dimension of observation space"),
                "action_dim": IntParam(2, 1, 64, doc="Dimension of action space (output vector size)"),
                "hidden_layers": StringParam(
                    "64,64",
                    doc="Comma-separated list of hidden layer sizes (e.g., '128,64,32')",
                ),
            },
            "training": {
                "learning_rate": FloatParam(3e-4, 1e-6, 1e-2, doc="Learning rate for optimizer"),
                "gamma": FloatParam(0.99, 0.8, 0.999, doc="Discount factor for future rewards"),
                "gae_lambda": FloatParam(0.95, 0.8, 0.99, doc="Lambda for Generalized Advantage Estimation"),
                "clip_epsilon": FloatParam(0.2, 0.05, 0.5, doc="PPO clipping parameter"),
                "value_loss_coef": FloatParam(0.5, 0.1, 1.0, doc="Coefficient for value loss"),
                "entropy_coef": FloatParam(0.01, 0.0, 0.1, doc="Coefficient for entropy bonus"),
                "max_grad_norm": FloatParam(0.5, 0.1, 10.0, doc="Maximum gradient norm for clipping"),
                "buffer_size": IntParam(128, 16, 2048, doc="Number of timesteps before policy update"),
                "epochs_per_update": IntParam(4, 1, 20, doc="Number of epochs to train on each buffer"),
                "minibatch_size": IntParam(32, 4, 256, doc="Minibatch size for training"),
                "enable_training": BoolParam(True, doc="Enable/disable training (inference only when False)"),
            },
            "control": {
                "reset_agent": BoolParam(False, trigger=True, doc="Reset the agent and training"),
                "device": StringParam("auto", options=["auto", "cpu", "cuda"], doc="Device to run model on"),
                "action_smoothing": FloatParam(
                    0.0, 0.0, 0.99, doc="Action smoothing factor (0=no smoothing, higher=slower changes)"
                ),
            },
        }

    def setup(self):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn

        # Set device
        self._set_device()

        # Initialize agent
        self._initialize_agent()

        # Training buffer
        self.buffer = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
        }

        # State tracking
        self.prev_observation = None
        self.prev_action = None
        self.prev_log_prob = None
        self.prev_value = None
        self.timestep = 0
        self.smoothed_action = None

        # Training statistics
        self.total_reward = 0
        self.episode_rewards = []

    def _set_device(self):
        """Set the computation device."""
        device_param = self.params.control.device.value
        if device_param == "auto":
            self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        else:
            self.device = device_param
        print(f"ReinforcementLearning node using device: {self.device}")

    def _parse_hidden_layers(self):
        """Parse the comma-separated hidden layer sizes."""
        hidden_str = self.params.architecture.hidden_layers.value.strip()
        if not hidden_str:
            return []
        try:
            layers = [int(x.strip()) for x in hidden_str.split(",") if x.strip()]
            return layers
        except ValueError:
            print(f"Warning: Could not parse hidden layers '{hidden_str}', using default [64, 64]")
            return [64, 64]

    def _initialize_agent(self):
        """Initialize the PPO agent with policy and value networks."""
        obs_dim = self.params.architecture.observation_dim.value
        action_dim = self.params.architecture.action_dim.value
        hidden_layers = self._parse_hidden_layers()

        # Build policy network (actor)
        policy_layers = []
        prev_size = obs_dim
        for hidden_size in hidden_layers:
            policy_layers.extend(
                [
                    self.nn.Linear(prev_size, hidden_size),
                    self.nn.Tanh(),
                ]
            )
            prev_size = hidden_size

        # Output layer for mean and log_std of action distribution
        policy_layers.append(self.nn.Linear(prev_size, action_dim))
        self.policy_net = self.nn.Sequential(*policy_layers).to(self.device)

        # Learnable log standard deviation (state-independent)
        self.log_std = self.nn.Parameter(self.torch.zeros(action_dim, device=self.device))

        # Build value network (critic)
        value_layers = []
        prev_size = obs_dim
        for hidden_size in hidden_layers:
            value_layers.extend(
                [
                    self.nn.Linear(prev_size, hidden_size),
                    self.nn.Tanh(),
                ]
            )
            prev_size = hidden_size
        value_layers.append(self.nn.Linear(prev_size, 1))
        self.value_net = self.nn.Sequential(*value_layers).to(self.device)

        # Optimizer
        lr = self.params.training.learning_rate.value
        params = list(self.policy_net.parameters()) + list(self.value_net.parameters()) + [self.log_std]
        self.optimizer = self.torch.optim.Adam(params, lr=lr)

        print(f"Initialized PPO agent: obs_dim={obs_dim}, action_dim={action_dim}, hidden={hidden_layers}")

    def _get_action_and_value(self, observation):
        """Get action and value estimate for a given observation."""
        obs_tensor = self.torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            # Get action mean from policy network
            action_mean = self.policy_net(obs_tensor)
            action_std = self.torch.exp(self.log_std)

            # Sample action from Gaussian distribution
            dist = self.torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            # Get value estimate
            value = self.value_net(obs_tensor)

            # Clip actions to [-1, 1]
            action = self.torch.tanh(action)

        return (
            action.cpu().numpy().flatten(),
            value.cpu().numpy().flatten()[0],
            log_prob.cpu().numpy().flatten()[0],
        )

    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        gamma = self.params.training.gamma.value
        gae_lambda = self.params.training.gae_lambda.value

        advantages = []
        gae = 0

        # Process in reverse order
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return np.array(advantages)

    def _update_policy(self):
        """Update the policy using PPO algorithm."""
        if len(self.buffer["observations"]) < self.params.training.minibatch_size.value:
            return

        # Convert buffer to tensors
        obs = self.torch.FloatTensor(np.array(self.buffer["observations"])).to(self.device)
        actions = self.torch.FloatTensor(np.array(self.buffer["actions"])).to(self.device)
        old_log_probs = self.torch.FloatTensor(np.array(self.buffer["log_probs"])).to(self.device)
        rewards = np.array(self.buffer["rewards"])
        values = np.array(self.buffer["values"])
        dones = np.array(self.buffer["dones"])

        # Compute advantages
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values

        advantages = self.torch.FloatTensor(advantages).to(self.device)
        returns = self.torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        buffer_size = len(obs)
        minibatch_size = min(self.params.training.minibatch_size.value, buffer_size)
        epochs = self.params.training.epochs_per_update.value

        for epoch in range(epochs):
            # Shuffle indices
            indices = self.torch.randperm(buffer_size)

            for start in range(0, buffer_size, minibatch_size):
                end = start + minibatch_size
                if end > buffer_size:
                    continue

                batch_indices = indices[start:end]
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get current policy outputs
                action_mean = self.policy_net(batch_obs)
                action_std = self.torch.exp(self.log_std)
                dist = self.torch.distributions.Normal(action_mean, action_std)

                # Compute log probabilities of the taken actions
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # Compute value estimates
                values_pred = self.value_net(batch_obs).squeeze()

                # PPO clipped loss
                ratio = self.torch.exp(log_probs - batch_old_log_probs)
                clip_epsilon = self.params.training.clip_epsilon.value

                surr1 = ratio * batch_advantages
                surr2 = self.torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                policy_loss = -self.torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = self.nn.functional.mse_loss(values_pred, batch_returns)

                # Total loss
                entropy_coef = self.params.training.entropy_coef.value
                value_loss_coef = self.params.training.value_loss_coef.value
                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()) + [self.log_std],
                    self.params.training.max_grad_norm.value,
                )
                self.optimizer.step()

        # Clear buffer after update
        self.buffer = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
        }

    def process(self, observations: Data, reward: Data):
        # Handle reset
        if self.params.control.reset_agent.value:
            self.setup()
            self.params.control.reset_agent.value = False
            print("Agent reset")

        # Check for None inputs
        if observations is None:
            return None

        # Get observation data
        obs_data = observations.data.flatten()

        # Validate observation dimension
        expected_obs_dim = self.params.architecture.observation_dim.value
        if len(obs_data) != expected_obs_dim:
            print(f"Warning: Expected {expected_obs_dim} observations, got {len(obs_data)}. Padding/truncating.")
            if len(obs_data) < expected_obs_dim:
                obs_data = np.pad(obs_data, (0, expected_obs_dim - len(obs_data)), mode="constant")
            else:
                obs_data = obs_data[:expected_obs_dim]

        # Store previous transition in buffer (if we have previous data)
        if self.prev_observation is not None and reward is not None:
            reward_value = reward.data.flatten()[0] if reward.data.size > 0 else 0.0

            self.buffer["observations"].append(self.prev_observation)
            self.buffer["actions"].append(self.prev_action)
            self.buffer["rewards"].append(reward_value)
            self.buffer["values"].append(self.prev_value)
            self.buffer["log_probs"].append(self.prev_log_prob)
            self.buffer["dones"].append(0)  # Assuming continuous interaction

            self.total_reward += reward_value

            # Update policy when buffer is full
            if (
                len(self.buffer["observations"]) >= self.params.training.buffer_size.value
                and self.params.training.enable_training.value
            ):
                self._update_policy()
                print(
                    f"Policy updated at timestep {self.timestep}, avg reward: {self.total_reward / self.params.training.buffer_size.value:.3f}"
                )
                self.total_reward = 0

        # Get action for current observation
        action, value, log_prob = self._get_action_and_value(obs_data)

        # Apply action smoothing
        smoothing = self.params.control.action_smoothing.value
        if smoothing > 0 and self.smoothed_action is not None:
            # Exponential moving average: smoothed = α * smoothed + (1-α) * new
            action = smoothing * self.smoothed_action + (1 - smoothing) * action
        self.smoothed_action = action

        # Store current state for next iteration
        self.prev_observation = obs_data
        self.prev_action = action
        self.prev_value = value
        self.prev_log_prob = log_prob
        self.timestep += 1

        # Return action and value
        return {
            "actions": (action, {"timestep": self.timestep, "training": self.params.training.enable_training.value}),
            "value": (np.array([value]), {"timestep": self.timestep}),
        }
