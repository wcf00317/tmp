# experiments/run_skm_mmlu.py

import torch
import argparse
import yaml
import os
from tqdm import tqdm
import logging
import numpy as np
import random
import re
from sentence_transformers import SentenceTransformer

# 确保GDesigner模块可以被正确导入
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GDesigner.agents.agent_registry import AgentRegistry
from datasets.mmlu_dataset import MMLUDataset
from GDesigner.manager.manager import Manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation for SKM-Net using a configuration file.")
    # --- 核心改进：只需要一个config文件 ---
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the configuration file (.yaml) used for training.")
    args = parser.parse_args()

    # --- 1. 从YAML文件加载所有参数 ---
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found at {args.config}")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    params = config['training_params']

    set_seed(params['seed'])

    logging.info(f"--- Starting Evaluation with Configuration: {args.config} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    agent_list = list(config['agents'].keys())
    agent_registry = AgentRegistry(config)
    num_agents = len(agent_list)

    embedding_model_name = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    logging.info(f"Loaded embedding model '{embedding_model_name}' with dimension {embedding_dim}.")

    agent_input_dims = {i: embedding_dim for i in range(num_agents)}

    manager = Manager(
        state_dim=params['state_dim'],
        message_dim=params['message_dim'],
        num_agents=num_agents,
        agent_input_dims=agent_input_dims,
        graph_mode=params['graph_mode'],
        # credit_alpha is not needed for evaluation
    ).to(device)

    state_updater = torch.nn.GRUCell(params['message_dim'], params['state_dim']).to(device)

    # --- 核心改进：自动从config读取checkpoint路径 ---
    checkpoint_path = params['save_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file specified in config ('{checkpoint_path}') not found. Please train the model first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    manager.load_state_dict(checkpoint['manager_state_dict'])
    state_updater.load_state_dict(checkpoint['state_updater_state_dict'])

    manager.eval()
    state_updater.eval()
    logging.info(f"Successfully loaded trained model and state updater from {checkpoint_path}")

    # 初始化数据集
    dataset = MMLUDataset(config['dataset_path'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=False)

    correct_predictions, total_predictions = 0, 0

    # --- 2. 评估循环 (逻辑不变) ---
    with torch.no_grad():
        for i, (question, choices, answer) in enumerate(tqdm(dataloader, desc="Evaluating")):

            state = torch.zeros(params['batch_size'], params['state_dim']).to(device)
            final_agent_output = ""

            for step in range(params['max_reasoning_steps']):
                action_dist = manager.actor(state)
                action = action_dist.probs.argmax(dim=-1)

                agent_id = action.item()
                agent_to_call = agent_registry.get_agent(agent_list[agent_id])
                raw_output = agent_to_call.run(question=question[0], **{"choices": choices[0]})
                final_agent_output = raw_output

                raw_output_str = str(raw_output) if raw_output is not None else ""
                raw_output_embedding = embedding_model.encode(raw_output_str, convert_to_tensor=True,
                                                              device=device).float()

                message, _ = manager.gateways[str(agent_id)](raw_output_embedding)

                state = state_updater(message, state)

            predicted_char = "Z"
            if isinstance(final_agent_output, str):
                match = re.search(r'final answer is.*([A-D])', final_agent_output, re.IGNORECASE)
                if match:
                    predicted_char = match.group(1).upper()

            if predicted_char == answer[0]:
                correct_predictions += 1
            total_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    logging.info(f"Trained model and state updater from {checkpoint_path}.")
    logging.info(f"Evaluation finished. Total questions: {total_predictions}, Correct: {correct_predictions}")
    logging.info(f"Final Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()