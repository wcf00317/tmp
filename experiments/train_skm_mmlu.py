# experiments/train_skm_mmlu.py (fixed args/config handling)

import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import yaml
import os
import random
import numpy as np
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer  # Correctly import the necessary library
from typing import List
import pandas as pd
# Ensure GDesigner modules can be correctly imported
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Imports based on the latest repository structure ---
from GDesigner.agents.agent_registry import AgentRegistry
from datasets.mmlu_dataset import MMLUDataset  # Located in the top-level datasets directory
from GDesigner.manager.manager import Manager  # Our new Manager

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Local Utility Function (as per original repo style) ---
def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train the SKM-Net framework using a configuration file.")
    parser.add_argument('--config', type=str, default='./config.yaml', help="Path to the training configuration file (.yaml). Default: ./config.yaml")

    # Allow overriding important config values via CLI (these will override values in the YAML if provided)
    parser.add_argument('--state_dim', type=int, default=None)
    parser.add_argument('--message_dim', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_reasoning_steps', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr_actor', type=float, default=None)
    parser.add_argument('--lr_critic', type=float, default=None)
    parser.add_argument('--lr_gateway', type=float, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--embedding_model', type=str, default=None)
    parser.add_argument('--graph_mode', type=str, default=None, help='Optional graph mode override (string)')
    parser.add_argument('--credit_alpha', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    # --- Load YAML config (if exists) ---
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        logging.warning(f"Config file {args.config} not found. Using empty/default config.")
        config = {}

    # Extract training params from config, or create an empty dict
    params = config.get('training_params', {}) or {}

    # Provide sensible defaults for missing training params
    default_params = {
        'state_dim': 128,
        'message_dim': 128,
        'batch_size': 1,
        'max_reasoning_steps': 3,
        'gamma': 0.99,
        'beta': 1.0,
        'epochs': 10,
        'lr_actor': 1e-4,
        'lr_critic': 1e-4,
        'lr_gateway': 1e-4,
    }
    for k, v in default_params.items():
        params.setdefault(k, v)

    # Override params with any CLI args provided
    override_param_keys = ['state_dim', 'message_dim', 'batch_size', 'max_reasoning_steps', 'gamma', 'beta', 'epochs', 'lr_actor', 'lr_critic', 'lr_gateway']
    for key in override_param_keys:
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val

    # Top-level config overrides (dataset_path, embedding model, save path, graph_mode, credit_alpha)
    dataset_path = args.dataset_path or config.get('dataset_path') or './data'
    embedding_model_name = args.embedding_model or config.get('embedding_model') or 'sentence-transformers/all-MiniLM-L6-v2'
    save_path = args.save_path or config.get('save_path') or './checkpoints/manager.pth'
    graph_mode = args.graph_mode if args.graph_mode is not None else config.get('graph_mode', 'shapley_evolution')
    credit_alpha = args.credit_alpha if args.credit_alpha is not None else config.get('credit_alpha', 0.5)

    # Optional seed
    if args.seed is not None:
        set_seed(args.seed)

    # --- 1. Initialization ---
    logging.info("Initializing SKM-Net Framework...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load agent list from config (be forgiving if missing)
    agent_list = list(config.get('agents', {}).keys())
    if len(agent_list) == 0:
        logging.error("No agents found in config['agents']. Please provide an 'agents' mapping in your config file.")
        return

    agent_registry = AgentRegistry()
    num_agents = len(agent_list)
    logging.info(f"Initialized {num_agents} fixed expert agents: {agent_list}")

    # --- Correctly initialize the embedding model ---
    embedding_model = SentenceTransformer(embedding_model_name, device=str(device))
    # SentenceTransformer returns embedding dim by method
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    logging.info(f"Loaded embedding model '{embedding_model_name}' with dimension {embedding_dim}.")

    agent_input_dims = {i: embedding_dim for i in range(num_agents)}

    # Manager initialization using unified params/config values
    manager = Manager(
        state_dim=params['state_dim'],
        message_dim=params['message_dim'],
        num_agents=num_agents,
        agent_input_dims=agent_input_dims,
        graph_mode=graph_mode,
        credit_alpha=credit_alpha
    ).to(device)

    # Initialize State Updater
    state_updater = torch.nn.GRUCell(params['message_dim'], params['state_dim']).to(device)

    # Initialize Optimizer for all trainable components
    optimizer = optim.Adam([
        {'params': manager.actor.parameters(), 'lr': params['lr_actor']},
        {'params': manager.critic.parameters(), 'lr': params['lr_critic']},
        {'params': manager.gateways.parameters(), 'lr': params['lr_gateway']},
        {'params': state_updater.parameters(), 'lr': params['lr_critic']}
    ])

    logging.info(f"Loading MMLU dataset from base path: {dataset_path}")
    dataset_dev = MMLUDataset(split='dev')
    dataset_val = MMLUDataset(split='val')
    dataset = torch.utils.data.ConcatDataset([dataset_dev, dataset_val])
    logging.info(f"Successfully loaded dataset with {len(dataset)} samples.")

    def mmlu_collate_fn(batch: List[pd.Series]):
        """
        Collates a list of pandas Series from MMLUDataset into a batch of lists.
        """
        questions = [item['question'] for item in batch]
        # 将每个样本的ABCD选项组合成一个列表
        choices = [[item['A'], item['B'], item['C'], item['D']] for item in batch]
        answers = [item['correct_answer'] for item in batch]

        # 返回三个列表，分别对应 question, choices, 和 answer
        return questions, choices, answers

    # 2. 在 DataLoader 中使用这个自定义函数
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        collate_fn=mmlu_collate_fn  # 关键改动在这里
    )

    # --- 2. Training Loop ---
    for epoch in range(params['epochs']):
        total_epoch_loss = 0.0
        total_epoch_reward = 0.0

        for i, (question, choices, answer) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{params['epochs']}")):

            # Initialize state for the new question
            # Keep original behavior: use configured batch_size as the state first dimension
            state = torch.zeros(params['batch_size'], params['state_dim']).to(device)
            messages, log_probs, q_values, advantages, ib_losses, rewards = {}, [], [], [], [], []

            # Multi-step reasoning process for a single question
            for step in range(params['max_reasoning_steps']):
                action_dist = manager.actor(state)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                # Assume single-agent selection as in original code (uses first item of batch)
                agent_id = action.item() if hasattr(action, 'item') else int(action)
                print(agent_id)
                agent_name = agent_list[agent_id]
                print(agent_name)
                agent_config = config.get('agents', {}).get(agent_name, {})

                # 从代理配置中提取 llm_name 和 domain (prompt_set)
                llm_name = agent_config.get('llm', {}).get('model_name', "")
                domain = agent_config.get('prompt_set', "")

                # 使用正确的参数获取代理实例
                agent_to_call = agent_registry.get(agent_name, domain=domain, llm_name=llm_name)
                # print(agent_to_call)

                raw_output = agent_to_call.run(question=question[0], **{"choices": choices[0]})

                with torch.no_grad():
                    # Ensure raw_output is a string for the encoder
                    raw_output_str = str(raw_output) if raw_output is not None else ""
                    raw_output_embedding = embedding_model.encode(raw_output_str, convert_to_tensor=True)
                    raw_output_embedding = raw_output_embedding.float().to(device)

                message, ib_loss = manager.gateways[str(agent_id)](raw_output_embedding)
                messages[agent_id] = message

                advantage, q_factual = manager.compute_advantage(state, messages, action)
                manager.update_credit_graph(agent_id, advantage)

                log_probs.append(log_prob)
                q_values.append(q_factual)
                advantages.append(advantage)
                ib_losses.append(ib_loss)
                rewards.append(0)  # Intermediate steps have no reward

                # Detach to prevent gradients from flowing through multiple unrolls of the state
                state = state_updater(message.detach(), state.detach())

            # Calculate final reward based on the final answer
            final_reward = 1.0 if answer[0] in str(raw_output) else -1.0
            rewards[-1] = final_reward
            total_epoch_reward += final_reward

            # Calculate returns for the Critic (Monte Carlo returns)
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + params['gamma'] * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32).to(device).unsqueeze(1)

            # Calculate losses
            actor_loss = -(torch.stack(log_probs) * torch.stack(advantages).detach()).mean()
            critic_loss = F.smooth_l1_loss(torch.stack(q_values), returns)
            total_ib_loss = torch.stack(ib_losses).mean()

            total_loss = actor_loss + critic_loss + params['beta'] * total_ib_loss
            total_epoch_loss += total_loss.item()

            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        avg_loss = total_epoch_loss / max(1, len(dataloader))
        avg_reward = total_epoch_reward / max(1, len(dataloader))
        logging.info(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}, Average Reward: {avg_reward:.4f}")

        # Save the model periodically
        if (epoch + 1) % 5 == 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'manager_state_dict': manager.state_dict(),
                'state_updater_state_dict': state_updater.state_dict(),
            }, save_path)
            logging.info(f"Model and state updater saved to {save_path}")


if __name__ == '__main__':
    main()
