# experiments/run_skm_mmlu.py

import torch
import argparse
import yaml
import os
import random
import numpy as np
# TQDM has been removed as requested
# from tqdm import tqdm
import logging
import re
import asyncio
import json
import hashlib
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer
import time

# Ensure GDesigner modules can be correctly imported
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from GDesigner.agents.agent_registry import AgentRegistry
from datasets.mmlu_dataset import MMLUDataset
from GDesigner.manager.manager import Manager
# Assuming accuracy.py exists and has the Accuracy class
from experiments.accuracy import Accuracy

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Caching Implementation ---
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
    # Add historical context to cache key to avoid conflicts in multi-step reasoning
    history = input_dict.get("history", "")
    full_task_string = task_string + "\n---\nHistory:\n" + history
    cache_key = get_cache_key(full_task_string)

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
                        help="Path to the training configuration file (.yaml).")
    parser.add_argument('--save_path', type=str, default=None,
                        help="Path to the saved model checkpoint (.pth). Overrides config.")
    parser.add_argument('--dataset_path', type=str, default=None, help="Path to the MMLU dataset. Overrides config.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for evaluation. Default is 1.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed. Overrides config.")
    # --- Control debug output ---
    parser.add_argument('--debug_count', type=int, default=5,
                        help="Number of initial samples to print for debugging. Set to 0 to disable.")

    args = parser.parse_args()

    # --- Load YAML config ---
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        logging.warning(f"Config file {args.config} not found. Using empty/default config.")
        config = {}

    params = config.get('training_params', {})

    # --- Determine parameters ---
    save_path = args.save_path or config.get('save_path') or './checkpoints/manager.pth'
    dataset_path = args.dataset_path or config.get('dataset_path') or './data'
    embedding_model_name = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    graph_mode = config.get('graph_mode', 'shapley_evolution')

    if args.seed is not None:
        set_seed(args.seed)

    # --- 1. Initialization ---
    logging.info(f"--- Starting Evaluation with Configuration: {args.config} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    agent_list = list(config.get('agents', {}).keys())
    if not agent_list:
        raise ValueError("No agents found in config['agents'].")

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
    ).to(device)

    state_updater = torch.nn.GRUCell(params['message_dim'], params['state_dim']).to(device)

    # --- 2. Load Model from Checkpoint ---
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Checkpoint file not found at '{save_path}'.")

    checkpoint = torch.load(save_path, map_location=device)
    manager.load_state_dict(checkpoint['manager_state_dict'])
    state_updater.load_state_dict(checkpoint['state_updater_state_dict'])
    manager.eval()
    state_updater.eval()
    logging.info(f"Successfully loaded model from {save_path}")

    # --- 3. Dataset Initialization ---
    logging.info(f"Loading MMLU dataset for evaluation (test split)...")
    dataset = MMLUDataset(split='test')

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

    accuracy = Accuracy()
    total_samples = len(dataloader)

    # --- 4. Evaluation Loop ---
    with torch.no_grad():
        # The main loop, TQDM is removed.
        for i, (questions, choices_list, answers) in enumerate(dataloader):
            # Print progress manually
            logging.info(f"Processing question {i + 1} of {total_samples}...")

            for j in range(len(questions)):
                question = questions[j]
                choices = choices_list[j]
                correct_answer = answers[j]

                # --- Enhanced Debugging Start ---
                is_debug_sample = (i * args.batch_size + j) < args.debug_count
                if is_debug_sample:
                    print("\n" + "=" * 80)
                    print(f"DEBUGGING SAMPLE #{i * args.batch_size + j + 1}")
                    print(f"QUESTION: {question}")
                    print(f"CHOICES: A) {choices[0]}, B) {choices[1]}, C) {choices[2]}, D) {choices[3]}")
                    print(f"CORRECT ANSWER: {correct_answer}")
                    print("-" * 80)
                # --- Enhanced Debugging End ---

                state = torch.zeros(1, params['state_dim']).to(device)
                reasoning_history = ""  # To track the chain of thought

                for step in range(params.get('max_reasoning_steps', 3)):
                    action_dist = manager.actor(state)
                    action = action_dist.probs.argmax(dim=-1)
                    agent_id = action.item()
                    agent_name = agent_list[agent_id]

                    agent_config = config.get('agents', {}).get(agent_name, {})
                    llm_name = agent_config.get('llm', {}).get('model_name', "")
                    domain = agent_config.get('prompt_set', "")
                    agent_to_call = agent_registry.get(agent_name, domain=domain, llm_name=llm_name)

                    task_string = (
                        f"Based on the following question and choices, please perform your task.\n\n"
                        f"Question: {question}\n\n"
                        f"Choices:\n"
                        f"A. {choices[0]}\n"
                        f"B. {choices[1]}\n"
                        f"C. {choices[2]}\n"
                        f"D. {choices[3]}"
                    )

                    # The input for the agent includes the history of previous steps
                    input_dict = {
                        "task": task_string,
                        "history": reasoning_history
                    }

                    raw_output = asyncio.run(cached_async_execute(
                        agent_to_call,
                        input_dict=input_dict,
                        spatial_info={},
                        temporal_info={}
                    ))

                    # Update the history for the next step
                    reasoning_history += f"Step {step + 1} ({agent_name}):\n{raw_output}\n\n"

                    # --- Enhanced Debugging Start ---
                    if is_debug_sample:
                        print(f"--- Step {step + 1} ---")
                        print(f"Manager selected Agent: '{agent_name}' (ID: {agent_id})")
                        print(f"Agent's Raw Output:\n---\n{raw_output}\n---")
                    # --- Enhanced Debugging End ---

                    raw_output_str = str(raw_output) if raw_output is not None else ""
                    raw_output_embedding = embedding_model.encode(raw_output_str, convert_to_tensor=True)
                    raw_output_embedding = raw_output_embedding.float().to(device).unsqueeze(0)

                    message, _ = manager.gateways[str(agent_id)](raw_output_embedding)
                    state = state_updater(message, state)

                # --- 5. Postprocess final reasoning chain and update accuracy ---
                predicted_answer = dataset.postprocess_answer(reasoning_history)
                accuracy.update(predicted_answer, correct_answer)

                # --- Enhanced Debugging Start ---
                if is_debug_sample:
                    print("-" * 80)
                    print("FINAL REASONING CHAIN (used for post-processing):")
                    print(reasoning_history)
                    print(f"==> Post-processed Predicted Answer: '{predicted_answer}'")
                    print(f"==> Correct Answer: '{correct_answer}'")
                    print("=" * 80 + "\n")
                # --- Enhanced Debugging End ---

    # --- 6. Final Results ---
    logging.info("Evaluation finished.")
    accuracy.print()


if __name__ == '__main__':
    # It's a good practice to clear the cache for a fresh evaluation run
    if os.path.exists(CACHE_FILE):
        logging.warning(f"Cache file '{CACHE_FILE}' found. For a fresh evaluation, consider deleting it.")
    main()