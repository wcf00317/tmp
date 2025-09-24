# experiments/test_skm_gsm8k.py
import asyncio
import torch
import argparse
import yaml
import os
import random
import numpy as np
import re
import logging
from sentence_transformers import SentenceTransformer
import json
import hashlib
from torch.utils.data import Dataset, DataLoader

# --- GDesigner Module Imports ---
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.manager.manager import Manager
# --- Import the robust answer extraction function ---
from datasets.gsm8k_dataset import gsm_get_predict

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Caching Implementation ---
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
    # Create a unique key based on agent name and task
    task_string = input_dict.get("task", "")
    agent_name = agent_to_call.agent_name
    cache_key = get_cache_key(f"{agent_name}:{task_string}")

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
        answer = item['answer'].split('####')[-1].strip()
        return question, answer


# --- Utility Functions ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    parser.add_argument('--solver_agent_name', type=str, default='MathSolver',
                        help="Name of the agent designated as the primary solver.")

    args = parser.parse_args()

    # --- Load Config ---
    if not os.path.exists(args.config):
        logging.error(f"Config file not found at {args.config}")
        return
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    params = config.get('training_params', {})

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

    # Ensure the designated solver agent is in the agent list from config
    if args.solver_agent_name not in agent_list:
        logging.error(f"Solver agent '{args.solver_agent_name}' not found in the agent list in {args.config}")
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
    manager.eval()
    logging.info(f"Successfully loaded model from {args.checkpoint_path}")

    # --- Load GSM8K Dataset ---
    if not os.path.exists(args.gsm8k_data_file):
        logging.error(f"GSM8K data file not found at {args.gsm8k_data_file}")
        return
    dataset = JSONLDataset(file_path=args.gsm8k_data_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    logging.info(f"Loaded {len(dataset)} questions from {args.gsm8k_data_file}")

    # --- Testing Loop ---
    correct_predictions = 0
    total_questions = 0

    with torch.no_grad():
        for i, (question, answer) in enumerate(dataloader):
            question = question[0]
            answer = answer[0]

            total_questions += 1
            print(f"\n\n[DEBUG] ========== NEW QUESTION #{total_questions} ==========")
            print(f"[DEBUG] Loaded Question: {question}")
            print(f"[DEBUG] Loaded Answer String: '{answer}'")

            state = torch.zeros(1, params['state_dim']).to(device)
            reasoning_history = []
            current_thought = ""  # To store the intermediate reasoning

            for step in range(params['max_reasoning_steps']):
                print(f"\n[DEBUG] --- Reasoning Step {step + 1}/{params['max_reasoning_steps']} ---")

                agent_name = ""
                # --- MODIFICATION: Force call solver agent on the first step ---
                if step == 0:
                    agent_name = args.solver_agent_name
                    print(f"[DEBUG] Forcing selection of solver agent: '{agent_name}'")
                else:
                    action_dist = manager.actor(state)
                    action = torch.argmax(action_dist.probs, dim=-1)
                    agent_id = action.item()
                    agent_name = agent_list[agent_id]
                    print(f"[DEBUG] Manager selected Agent: '{agent_name}' (ID: {agent_id})")

                agent_config = config.get('agents', {}).get(agent_name, {})
                llm_name = agent_config.get('llm', {}).get('model_name', "")
                domain = agent_config.get('prompt_set', "")
                agent_to_call = agent_registry.get(agent_name, domain=domain, llm_name=llm_name)

                # Pass the previous step's thoughts to the next agent
                input_dict = {"task": question, "thought": current_thought}

                raw_output = asyncio.run(cached_async_execute(
                    agent_to_call,
                    input_dict=input_dict,
                    spatial_info={},
                    temporal_info={}
                ))

                # The output might be a tuple (response, log_prob), handle it
                if isinstance(raw_output, tuple):
                    response_text = raw_output[0]
                else:
                    response_text = raw_output

                current_thought += f"\nStep {step + 1} ({agent_name}):\n{response_text}"
                reasoning_history.append({'agent_name': agent_name, 'output': response_text})
                print(f"[DEBUG] Raw output from Agent '{agent_name}': {response_text}")

                # Update state for the manager
                raw_output_str = str(response_text) if response_text is not None else ""
                raw_output_embedding = embedding_model.encode(raw_output_str, convert_to_tensor=True)
                raw_output_embedding = raw_output_embedding.float().to(device).unsqueeze(0)

                agent_id_for_gateway = str(agent_list.index(agent_name))
                message, _ = manager.gateways[agent_id_for_gateway](raw_output_embedding)
                state = state_updater(message, state)
                print(f"[DEBUG] State norm after update: {torch.norm(state).item():.4f}")

            # --- Final Answer Evaluation Logic ---
            final_answer_str = ""
            # Iterate backwards to find the last output from a non-adversarial agent
            for step_result in reversed(reasoning_history):
                # You might have other non-solver agents, add them to this list
                if 'AdversarialAgent' not in step_result['agent_name'] and 'FinalMajorVote' not in step_result[
                    'agent_name']:
                    final_answer_str = str(step_result['output'])
                    if final_answer_str and final_answer_str.lower() not in ['none', '']:
                        break

            print("\n[DEBUG] --- Final Evaluation ---")
            print(f"[DEBUG] String for final prediction extraction: '{final_answer_str}'")

            predicted_answer_str = gsm_get_predict(final_answer_str)
            print(f"[DEBUG] Extracted Predicted Answer String: '{predicted_answer_str}'")

            predicted_answer = None
            if predicted_answer_str:
                try:
                    predicted_answer = float(predicted_answer_str)
                except (ValueError, TypeError):
                    predicted_answer = None
            print(f"[DEBUG] Converted Predicted Answer: {predicted_answer}")

            true_answer = None
            try:
                true_answer = float(answer)
            except (ValueError, TypeError):
                true_answer = None
            print(f"[DEBUG] Gold Answer: {true_answer}")

            is_correct = False
            if predicted_answer is not None and true_answer is not None:
                if abs(predicted_answer - true_answer) < 1e-3:
                    correct_predictions += 1
                    is_correct = True

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