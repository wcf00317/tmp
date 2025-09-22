# experiments/test_skm_gsm8k.py
import asyncio
import torch
import argparse
import yaml
import os
import random
import numpy as np
import re
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer
from typing import List
import pandas as pd
import json
import hashlib
from torch.utils.data import Dataset, DataLoader

# --- GDesigner Module Imports ---
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.manager.manager import Manager

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Caching Implementation (same as training script) ---
CACHE_FILE = 'api_response_cache_gsm8k.json'
API_CACHE = {}


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_cache():
    with open(CACHE_FILE, 'w') as f:
        json.dump(API_CACHE, f, indent=4)


def get_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


API_CACHE = load_cache()


async def cached_async_execute(agent_to_call, input_dict, spatial_info, temporal_info):
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
        save_cache()
        return response


# --- Custom Dataset for local gsm8k.jsonl file ---
class JSONLDataset(Dataset):
    """
    A PyTorch Dataset to load data from a .jsonl file where each line
    is a JSON object with 'question' and 'answer' keys.
    """

    def __init__(self, file_path: str):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        # The answer is extracted from the format '...#### 123'
        answer = item['answer'].split('####')[-1].strip()
        return question, answer


# --- Utility Functions ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_last_number(text: str) -> float | None:
    """Extracts the last number from a string, handling commas."""
    # Remove commas from the text
    text_no_commas = text.replace(',', '')
    # Find all numbers in the cleaned string
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text_no_commas)
    if numbers:
        return float(numbers[-1])
    return None


def main():
    parser = argparse.ArgumentParser(description="Test a trained SKM-Net on the GSM8K dataset from a local jsonl file.")
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help="Path to the training configuration file (.yaml).")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the saved model checkpoint (.pth).")
    parser.add_argument('--gsm8k_data_file', type=str, default='./datasets/gsm8k/gsm8k.jsonl',
                        help="Path to the gsm8k.jsonl file.")
    parser.add_argument('--max_reasoning_steps', type=int, default=None,
                        help="Override max reasoning steps from config.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # --- Load Config ---
    if not os.path.exists(args.config):
        logging.error(f"Config file not found at {args.config}")
        return
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    params = config.get('training_params', {})

    # Override from CLI if provided
    if args.max_reasoning_steps is not None:
        params['max_reasoning_steps'] = args.max_reasoning_steps

    set_seed(args.seed)

    # --- Initialization ---
    logging.info("Initializing SKM-Net Framework for Testing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    agent_list = list(config.get('agents', {}).keys())
    if not agent_list:
        logging.error("No agents found in config file.")
        return

    agent_registry = AgentRegistry()
    num_agents = len(agent_list)

    embedding_model_name = config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    embedding_model = SentenceTransformer(embedding_model_name, device=str(device))
    embedding_dim = embedding_model.get_sentence_embedding_dimension()

    agent_input_dims = {i: embedding_dim for i in range(num_agents)}

    manager = Manager(
        state_dim=params['state_dim'],
        message_dim=params['message_dim'],
        num_agents=num_agents,
        agent_input_dims=agent_input_dims,
        graph_mode=config.get('graph_mode', 'shapley_evolution'),
        credit_alpha=config.get('credit_alpha', 0.5)
    ).to(device)

    state_updater = torch.nn.GRUCell(params['message_dim'], params['state_dim']).to(device)

    # --- Load Model Checkpoint ---
    if not os.path.exists(args.checkpoint_path):
        logging.error(f"Checkpoint file not found at {args.checkpoint_path}")
        return

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    manager.load_state_dict(checkpoint['manager_state_dict'])
    state_updater.load_state_dict(checkpoint['state_updater_state_dict'])
    manager.eval()  # Set model to evaluation mode
    logging.info(f"Successfully loaded model from {args.checkpoint_path}")

    # --- Load GSM8K Dataset from local jsonl file ---
    if not os.path.exists(args.gsm8k_data_file):
        logging.error(f"GSM8K data file not found at {args.gsm8k_data_file}")
        return
    dataset = JSONLDataset(file_path=args.gsm8k_data_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    logging.info(f"Loaded {len(dataset)} questions from {args.gsm8k_data_file}")

    # --- Testing Loop ---
    correct_predictions = 0
    total_questions = 0

    with torch.no_grad():  # No need to compute gradients
        for i, (question, answer) in enumerate(tqdm(dataloader, desc="Testing on GSM8K")):

            # Dataloader wraps items in a list/tuple, extract the single string
            question = question[0]
            answer = answer[0]

            # [DEBUG] Print loaded data to verify correctness
            print(f"\n\n[DEBUG] ========== NEW QUESTION #{total_questions + 1} ==========")
            print(f"[DEBUG] Loaded Question: {question}")
            print(f"[DEBUG] Loaded Answer String: '{answer}'")

            state = torch.zeros(1, params['state_dim']).to(device)
            raw_output = None

            # Multi-step reasoning
            for step in range(params['max_reasoning_steps']):
                print(f"\n[DEBUG] --- Reasoning Step {step + 1}/{params['max_reasoning_steps']} ---")

                action_dist = manager.actor(state)
                action = torch.argmax(action_dist.probs, dim=-1)  # Use argmax for deterministic evaluation

                agent_id = action.item()
                agent_name = agent_list[agent_id]
                agent_config = config.get('agents', {}).get(agent_name, {})

                # [DEBUG] Print selected agent
                print(f"[DEBUG] Manager selected Agent: '{agent_name}' (ID: {agent_id})")

                llm_name = agent_config.get('llm', {}).get('model_name', "")
                domain = agent_config.get('prompt_set', "")

                agent_to_call = agent_registry.get(agent_name, domain=domain, llm_name=llm_name)

                input_dict = {"task": question}

                # [DEBUG] Print input sent to the agent/LLM
                print(f"[DEBUG] Input to Agent: {json.dumps(input_dict, indent=2)}")

                raw_output = asyncio.run(cached_async_execute(
                    agent_to_call,
                    input_dict=input_dict,
                    spatial_info={},
                    temporal_info={}
                ))

                # [DEBUG] Print raw output from the agent/LLM
                print(f"[DEBUG] Raw output from Agent: {raw_output}")

                raw_output_str = str(raw_output) if raw_output is not None else ""
                raw_output_embedding = embedding_model.encode(raw_output_str, convert_to_tensor=True)
                raw_output_embedding = raw_output_embedding.float().to(device).unsqueeze(0)

                message, _ = manager.gateways[str(agent_id)](raw_output_embedding)
                state = state_updater(message, state)

                # [DEBUG] Print state norm to see if it's changing
                print(f"[DEBUG] State norm after update: {torch.norm(state).item():.4f}")

            # --- Evaluate the final answer ---
            final_answer_str = str(raw_output)

            # [DEBUG] Print details of the final evaluation
            print("\n[DEBUG] --- Final Evaluation ---")
            print(f"[DEBUG] String for final prediction extraction: '{final_answer_str}'")
            predicted_answer = extract_last_number(final_answer_str)
            print(f"[DEBUG] Extracted Predicted Answer: {predicted_answer}")

            print(f"[DEBUG] String for gold answer extraction: '{answer}'")
            true_answer = extract_last_number(answer)  # 'answer' is already the cleaned number string
            print(f"[DEBUG] Extracted Gold Answer: {true_answer}")

            is_correct = False
            if predicted_answer is not None and true_answer is not None:
                if abs(predicted_answer - true_answer) < 1e-3:
                    correct_predictions += 1
                    is_correct = True

            total_questions += 1

            # Log details for inspection
            print(f"\n--- Question {total_questions} ---")
            print(f"Question: {question}")
            print(f"Gold Answer: {true_answer}")
            print(f"Predicted Raw Output: {final_answer_str}")
            print(f"Predicted Final Answer: {predicted_answer}")
            print(f"Result: {'CORRECT' if is_correct else 'WRONG'}")
            print(f"----------------\n")

    # --- Final Accuracy ---
    accuracy = (correct_predictions / total_questions) * 100 if total_questions > 0 else 0
    logging.info(f"Testing Finished.")
    logging.info(f"Total Questions: {total_questions}")
    logging.info(f"Correct Predictions: {correct_predictions}")
    logging.info(f"GSM8K Test Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()