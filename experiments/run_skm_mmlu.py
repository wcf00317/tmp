# experiments/run_skm_mmlu.py

import torch
import argparse
import yaml
import os
import random
import numpy as np
from tqdm import tqdm
import logging
import re
import asyncio
import json
import hashlib
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer

# Ensure GDesigner modules can be correctly imported
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GDesigner.agents.agent_registry import AgentRegistry
from datasets.mmlu_dataset import MMLUDataset
from GDesigner.manager.manager import Manager

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Caching Implementation (same as in training script) ---
CACHE_FILE = 'api_response_cache.json'
API_CACHE = {}


def load_cache():
    """Loads the API response cache from a file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("Cache file exists but there was a loading error!")
                return {}
    return {}


def save_cache():
    """Saves the API response cache to a file."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(API_CACHE, f, indent=4)


def get_cache_key(text: str) -> str:
    """Creates a unique MD5 hash for a given text string."""
    return hashlib.md5(text.encode()).hexdigest()


# Load cache at the start of the script
API_CACHE = load_cache()


async def cached_async_execute(agent_to_call, input_dict, spatial_info, temporal_info):
    """
    A wrapper for _async_execute that caches results based on the task string.
    """
    task_string = input_dict.get("task", "")
    cache_key = get_cache_key(task_string)

    if cache_key in API_CACHE:
        return API_CACHE[cache_key]
    else:
        response = await agent_to_call._async_execute(
            input=input_dict,
            spatial_info=spatial_info,
            temporal_info=temporal_info
        )
        API_CACHE[cache_key] = response
        save_cache()  # Save after each new API call
        return response


# --- Utility Function ---
def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the SKM-Net framework using a configuration file.")
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help="Path to the training configuration file (.yaml). Default: ./config.yaml")

    # Allow overriding important config values via CLI
    parser.add_argument('--save_path', type=str, default=None,
                        help="Path to the saved model checkpoint (.pth). Overrides config.")
    parser.add_argument('--dataset_path', type=str, default=None, help="Path to the MMLU dataset. Overrides config.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for evaluation. Default is 1.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed. Overrides config.")

    args = parser.parse_args()

    # --- Load YAML config ---
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        logging.warning(f"Config file {args.config} not found. Using empty/default config.")
        config = {}

    params = config.get('training_params', {})

    # --- Determine parameters, prioritizing CLI args > config file > defaults ---
    save_path = args.save_path or config.get('save_path') or './checkpoints/manager.pth'
    dataset_path = args.dataset_path or config.get('dataset_path') or './data'
    embedding_model_name = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    graph_mode = config.get('graph_mode',
                            'shapley_evolution')  # graph_mode is not critical for eval but needed for Manager init

    if args.seed is not None:
        set_seed(args.seed)

    # --- 1. Initialization ---
    logging.info(f"--- Starting Evaluation with Configuration: {args.config} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    agent_list = list(config.get('agents', {}).keys())
    if not agent_list:
        raise ValueError("No agents found in config['agents']. Please provide an 'agents' mapping in your config file.")

    agent_registry = AgentRegistry()
    num_agents = len(agent_list)

    embedding_model = SentenceTransformer(embedding_model_name, device=str(device))
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    logging.info(f"Loaded embedding model '{embedding_model_name}' with dimension {embedding_dim}.")

    agent_input_dims = {str(i): embedding_dim for i in range(num_agents)}

    manager = Manager(
        state_dim=params['state_dim'],
        message_dim=params['message_dim'],
        num_agents=num_agents,
        agent_input_dims=agent_input_dims,
        graph_mode=graph_mode,
        # credit_alpha is not needed for evaluation
    ).to(device)

    state_updater = torch.nn.GRUCell(params['message_dim'], params['state_dim']).to(device)

    # --- 2. Load Model from Checkpoint ---
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Checkpoint file not found at '{save_path}'. Please provide the correct path.")

    checkpoint = torch.load(save_path, map_location=device)
    manager.load_state_dict(checkpoint['manager_state_dict'])
    state_updater.load_state_dict(checkpoint['state_updater_state_dict'])

    manager.eval()
    state_updater.eval()
    logging.info(f"Successfully loaded trained model and state updater from {save_path}")

    # --- 3. Dataset Initialization ---
    logging.info(f"Loading MMLU dataset for evaluation (test split)...")
    dataset = MMLUDataset(split='test')  # Using the 'test' split for final evaluation

    def mmlu_collate_fn(batch: List[pd.Series]):
        questions = [item['question'] for item in batch]
        choices = [[item['A'], item['B'], item['C'], item['D']] for item in batch]
        answers = [item['correct_answer'] for item in batch]
        return questions, choices, answers

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=mmlu_collate_fn
    )

    correct_predictions, total_predictions = 0, 0

    # --- 4. Evaluation Loop ---
    with torch.no_grad():
        for questions, choices_list, answers in tqdm(dataloader, desc="Evaluating"):

            for j in range(len(questions)):
                question = questions[j]
                choices = choices_list[j]
                correct_answer = answers[j]

                state = torch.zeros(1, params['state_dim']).to(device)
                final_agent_output = ""

                for step in range(params.get('max_reasoning_steps', 3)):
                    action_dist = manager.actor(state)
                    action = action_dist.probs.argmax(dim=-1)  # Use argmax for deterministic evaluation

                    agent_id = action.item()
                    agent_name = agent_list[agent_id]
                    agent_config = config.get('agents', {}).get(agent_name, {})

                    llm_name = agent_config.get('llm', {}).get('model_name', "")
                    domain = agent_config.get('prompt_set', "")

                    agent_to_call = agent_registry.get(agent_name, domain=domain, llm_name=llm_name)

                    task_string = (
                        f"{question}\n\n"
                        f"Choices:\n"
                        f"A. {choices[0]}\n"

                        f"B. {choices[1]}\n"
                        f"C. {choices[2]}\n"
                        f"D. {choices[3]}"
                    )
                    input_dict = {"task": task_string}

                    # Run agent asynchronously
                    raw_output = asyncio.run(cached_async_execute(
                        agent_to_call,
                        input_dict=input_dict,
                        spatial_info={},
                        temporal_info={}
                    ))
                    final_agent_output = raw_output

                    # Encode output and update state
                    raw_output_str = str(raw_output) if raw_output is not None else ""
                    raw_output_embedding = embedding_model.encode(raw_output_str, convert_to_tensor=True)
                    raw_output_embedding = raw_output_embedding.float().to(device).unsqueeze(0)

                    message, _ = manager.gateways[str(agent_id)](raw_output_embedding)
                    state = state_updater(message, state)

                # --- 5. Parse final answer and calculate accuracy ---
                predicted_char = "Z"  # Default to a wrong answer
                if isinstance(final_agent_output, str):
                    # Use a robust regex to find the final answer choice
                    match = re.search(r'([A-D])', str(final_agent_output))
                    if match:
                        predicted_char = match.group(1).upper()

                if predicted_char == correct_answer:
                    correct_predictions += 1
                total_predictions += 1

    # --- 6. Final Results ---
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    logging.info(f"Evaluation finished.")
    logging.info(f"Total Questions: {total_predictions}, Correct Predictions: {correct_predictions}")
    logging.info(f"Final Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()