# GDesigner/manager/manager.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .gateway import VIBGateway
from .actor_critic import Actor, Critic


class Manager(nn.Module):
    """
    The central trainable Manager (Orchestrator) for the SKM-Net framework.

    It integrates all core components and manages the entire process of
    expert agent orchestration. It supports multiple graph generation and
    information propagation modes, which can be switched via the `graph_mode` parameter.
    """

    def __init__(self,
                 state_dim: int,
                 message_dim: int,
                 num_agents: int,
                 agent_input_dims: Dict[int, int],
                 graph_mode: str = 'shapley_evolution',
                 credit_alpha: float = 0.1):
        """
        Initializes the Manager network.

        Args:
            state_dim (int): Dimension of the global state representation.
            message_dim (int): Dimension of the latent messages 'm'.
            num_agents (int): Number of expert agents.
            agent_input_dims (Dict[int, int]): A map from agent_id to its raw output embedding dim.
            graph_mode (str): The mode for graph structure.
                               Supported: 'shapley_evolution' (Direction 3),
                                          'implicit_attention' (Direction 5).
            credit_alpha (float): The learning rate for the credit graph evolution.
        """
        super().__init__()

        # --- Core Component Initialization ---
        print(f"[SKM-Net Manager] Initializing in '{graph_mode}' mode.")
        self.actor = Actor(state_dim, num_agents)
        self.critic = Critic(state_dim, message_dim, num_agents)
        self.gateways = nn.ModuleDict({
            str(agent_id): VIBGateway(input_dim, message_dim)
            for agent_id, input_dim in agent_input_dims.items()
        })

        self.num_agents = num_agents
        self.graph_mode = graph_mode
        self.state_dim = state_dim
        self.message_dim = message_dim

        # --- Mode-Specific Initialization ---
        if self.graph_mode == 'shapley_evolution':
            # Direction 3: Initialize the Dynamic Credit Graph
            # This matrix is updated manually, not via backpropagation, so requires_grad=False.
            self.credit_matrix = torch.zeros(num_agents, num_agents, requires_grad=False)
            self.credit_alpha = credit_alpha
            print(
                f"[SKM-Net Manager] Initialized Credit Matrix of shape {self.credit_matrix.shape} for graph evolution.")

        elif self.graph_mode == 'implicit_attention':
            # Direction 5: Initialize a multi-head attention layer for information propagation
            # We use a simple single-head attention for clarity here.
            self.attention_layer = nn.MultiheadAttention(embed_dim=state_dim, num_heads=4, batch_first=True)
            # A layer to project messages to the same dimension as the state for attention
            self.message_projector = nn.Linear(message_dim, state_dim)
            print(f"[SKM-Net Manager] Initialized Attention Layer for implicit graph.")

        else:
            raise ValueError(f"Unsupported graph_mode: {graph_mode}")

    def _propagate_shapley_graph(self, messages: Dict[int, torch.Tensor], agent_states: torch.Tensor):
        """
        Propagates information based on the explicitly evolving Credit Graph.
        """
        # A simple propagation: each agent's state is updated by a weighted sum of all messages,
        # where weights are determined by the credit scores.

        # Ensure all agents have a message tensor (even if it's zeros)
        batch_size = agent_states.shape[0]
        all_messages = torch.zeros(batch_size, self.num_agents, self.message_dim, device=agent_states.device)
        for agent_id, msg in messages.items():
            all_messages[:, agent_id, :] = msg

        # Get weights from the credit matrix (N, N) and apply sigmoid
        # The weights determine how much agent j listens to agent i.
        weights = torch.sigmoid(self.credit_matrix.to(agent_states.device)).unsqueeze(0)  # Shape: (1, N, N)

        # Weighted sum of messages: b x N_j x N_i * b x N_i x D_m -> b x N_j x D_m
        aggregated_info = torch.bmm(weights, all_messages)

        # A simple update rule: new_state = old_state + MLP(aggregated_info)
        # For simplicity, we'll just return the aggregated info to be used by a final aggregator
        print("[SKM-Net Debug] Propagating info via Shapley Graph.")
        return aggregated_info

    def _propagate_attention(self, messages: Dict[int, torch.Tensor], agent_states: torch.Tensor):
        """
        Propagates information using a Transformer-style attention mechanism.
        """
        # The agent states act as Queries, and the projected messages act as Keys and Values.

        # Ensure all agents are present
        batch_size = agent_states.shape[0]

        # Project messages to the same dimension as states
        # Even if some agents didn't speak, their state can still act as a query.
        projected_messages = torch.zeros_like(agent_states)
        for agent_id, msg in messages.items():
            projected_messages[:, agent_id, :] = self.message_projector(msg)

        # The agent's current state is the Query.
        # The messages from other agents are the Keys and Values.
        # This allows each agent to "look" at all messages and decide which are relevant.
        updated_states, attention_weights = self.attention_layer(
            query=agent_states,
            key=projected_messages,
            value=projected_messages
        )
        print("[SKM-Net Debug] Propagating info via Attention.")
        return updated_states

    def forward(self, state: torch.Tensor, messages: Dict[int, torch.Tensor]):
        """
        A simplified forward pass for information propagation.
        The full step-by-step logic will be in the training loop.
        """
        # In the full implementation, this will be more complex, involving the actor call etc.
        # Here we just demonstrate the switchable propagation.

        # Create dummy agent states for propagation demonstration
        # In a real scenario, these would be managed RNN states for each agent.
        agent_states = state.unsqueeze(1).repeat(1, self.num_agents, 1)  # A simplified shared state

        if self.graph_mode == 'shapley_evolution':
            updated_agent_info = self._propagate_shapley_graph(messages, agent_states)
        elif self.graph_mode == 'implicit_attention':
            updated_agent_info = self._propagate_attention(messages, agent_states)

        # The final decision would be made by another module based on this updated info
        return updated_agent_info

    def compute_advantage(self, state: torch.Tensor, messages: Dict[int, torch.Tensor], action: torch.Tensor):
        """
        Computes the Shapley-COMA advantage for the agent that was just called.
        This method is independent of the graph_mode.
        """
        agent_id = action.item()

        # 1. Factual Q-value
        q_factual = self.critic(state, messages, action)

        # 2. Counterfactual Q-value
        messages_cf = messages.copy()
        # Replace the message with a sample from the prior (uninformative message)
        uninformative_message = torch.randn(1, self.message_dim, device=state.device)
        messages_cf[agent_id] = uninformative_message
        q_counterfactual = self.critic(state, messages_cf, action)

        advantage = q_factual - q_counterfactual
        print(
            f"[SKM-Net Debug] Advantage for agent {agent_id}: {advantage.item():.4f} (Q_fact: {q_factual.item():.4f} - Q_cf: {q_counterfactual.item():.4f})")
        return advantage, q_factual

    def update_credit_graph(self, source_agent_id: int, advantage: torch.Tensor):
        """
        Evolves the credit graph based on the Shapley-based advantage signal.
        This method should only be called if graph_mode is 'shapley_evolution'.
        """
        if self.graph_mode != 'shapley_evolution':
            # print("[SKM-Net Debug] Skipping credit graph update (not in shapley_evolution mode).")
            return

        with torch.no_grad():
            advantage_value = advantage.item()
            # Update all outgoing edges from the source agent
            self.credit_matrix[source_agent_id, :] = \
                (1 - self.credit_alpha) * self.credit_matrix[source_agent_id, :] + \
                self.credit_alpha * advantage_value
            print(
                f"[SKM-Net Debug] Credit graph updated for source agent {source_agent_id} with advantage {advantage_value:.4f}.")