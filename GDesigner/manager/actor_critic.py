# GDesigner/manager/actor_critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class Actor(nn.Module):
    """
    The Actor network (Policy Network).

    It observes the current state of the system and decides which expert agent (API)
    to call next.
    """

    def __init__(self, state_dim: int, num_agents: int):
        """
        Initializes the Actor network.

        Args:
            state_dim (int): The dimension of the global state representation.
            num_agents (int): The number of available expert agents (the action space size).
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, num_agents)
        )
        print(f"[SKM-Net Manager] Actor initialized: StateDim={state_dim}, NumAgents={num_agents}")

    def forward(self, state: torch.Tensor):
        """
        Takes the current state and returns a probability distribution over actions.

        Args:
            state (torch.Tensor): The global state representation. Shape: (batch_size, state_dim)

        Returns:
            torch.distributions.Categorical: A distribution over the agents to call.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        action_logits = self.policy_net(state)
        action_dist = F.softmax(action_logits, dim=-1)

        # Using Categorical distribution for easy sampling and log_prob calculation
        dist = torch.distributions.Categorical(action_dist)
        return dist


class Critic(nn.Module):
    """
    The Centralized Critic network (Value Network).

    It evaluates the long-term value of the current situation (state, messages, action)
    and provides the advantage signal for training the Actor and evolving the graph.
    """

    def __init__(self, state_dim: int, message_dim: int, num_agents: int):
        """
        Initializes the Critic network.

        Args:
            state_dim (int): Dimension of the global state representation.
            message_dim (int): Dimension of the latent messages 'm'.
            num_agents (int): Number of expert agents.
        """
        super().__init__()
        # The input to the critic is complex: state + all messages + action
        # We can concatenate them after processing.
        # Let's assume a simple aggregation for now: state + mean_of_messages + action_one_hot
        input_dim = state_dim + message_dim + num_agents

        self.value_net = nn.Sequential(
            nn.Linear(input_dim, (input_dim + 1) // 2),
            nn.ReLU(),
            nn.Linear((input_dim + 1) // 2, 1)  # Outputs a single scalar Q-value
        )
        self.num_agents = num_agents
        self.message_dim = message_dim
        print(
            f"[SKM-Net Manager] Critic initialized: InputDim={input_dim} (State={state_dim}, Msg={message_dim}, Action={num_agents})")

    def forward(self, state: torch.Tensor, messages: Dict[int, torch.Tensor], action: torch.Tensor):
        """
        Calculates the Q-value for the current state, messages, and action.

        Args:
            state (torch.Tensor): Global state. Shape: (batch_size, state_dim)
            messages (Dict[int, torch.Tensor]): A dictionary mapping agent_id to its message.
                                                We'll aggregate these.
            action (torch.Tensor): The action taken by the Actor (index of the agent called).
                                   Shape: (batch_size,)

        Returns:
            torch.Tensor: The predicted Q-value. Shape: (batch_size, 1)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        # Aggregate messages - using a simple mean for now.
        # A more sophisticated approach could use attention.
        message_list = list(messages.values())
        if not message_list:
            # Handle the case where there are no messages yet
            aggregated_message = torch.zeros(batch_size, self.message_dim, device=state.device)
        else:
            aggregated_message = torch.stack(message_list).mean(dim=0)

        # Convert action to one-hot encoding
        action_one_hot = F.one_hot(action, num_classes=self.num_agents).float()

        # Concatenate all inputs
        critic_input = torch.cat([state, aggregated_message, action_one_hot], dim=-1)

        q_value = self.value_net(critic_input)
        return q_value