# experiments/train_skm_mmlu.py (fixed args/config handling)
import asyncio

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
# ===================== 修改开始 =====================
# 导入用于评估的 Accuracy 类
from experiments.accuracy import Accuracy
# ===================== 修改结束 =====================


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import json
import hashlib

# --- Caching Implementation ---
CACHE_FILE = 'api_response_cache.json'
API_CACHE = {}

def load_cache():
    """从文件加载API响应缓存"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("Cache file exists but load error!")
                return {}
    return {}

def save_cache():
    """将API响应缓存保存到文件"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(API_CACHE, f, indent=4)

def get_cache_key(text: str) -> str:
    """为给定的文本字符串创建一个唯一的MD5哈希值"""
    return hashlib.md5(text.encode()).hexdigest()

# 在脚本开始时加载缓存
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

        # 🟢 这里是关键：兼容 dict 和 ChatCompletion 对象
        model_text = None
        try:
            if isinstance(response, dict):
                model_text = response["choices"][0]["message"]["content"]
            elif hasattr(response, "choices"):
                model_text = response.choices[0].message.content
            else:
                model_text = str(response)
        except Exception as e:
            print("Failed to parse response:", e)
            model_text = str(response)

        API_CACHE[cache_key] = model_text
        save_cache()
        return model_text



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
    # ===================== 修改开始 =====================
    # 实例化一个 MMLUDataset 对象，以便后续可以调用它的 postprocess_answer 方法
    mmlu_eval_dataset = MMLUDataset(split='dev')
    # ===================== 修改结束 =====================


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
        epoch_correct_predictions = 0
        epoch_total_questions = 0

        for i, (questions, choices_list, answers) in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch + 1}/{params['epochs']}")):

            optimizer.zero_grad()

            batch_loss = 0.0
            batch_reward = 0.0

            # --- Loop through each item in the batch ---
            for j in range(len(questions)):
                question = questions[j]
                choices = choices_list[j]
                answer = answers[j] # 'answer' a.k.a correct_answer

                # Initialize state for the new question
                state = torch.zeros(1, params['state_dim']).to(device)  # State for a single item
                messages, log_probs, q_values, advantages, ib_losses, rewards = {}, [], [], [], [], []

                raw_output = None  # Ensure raw_output is defined
                reasoning_history = {}

                # Multi-step reasoning process for a single question
                for step in range(params['max_reasoning_steps']):
                    action_dist = manager.actor(state)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)

                    agent_id = action.item()  # action is now a single value
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

                    raw_output = asyncio.run(cached_async_execute(
                        agent_to_call,
                        input_dict=input_dict,
                        spatial_info=reasoning_history,
                        temporal_info={}
                    ))

                    print(f"---------------------------\n{agent_name}\n{llm_name}\n{domain}\n{raw_output}\n----------------------------")


                    with torch.no_grad():
                        raw_output_str = raw_output or ""
                        raw_output_embedding = embedding_model.encode(raw_output_str, convert_to_tensor=True)
                        raw_output_embedding = raw_output_embedding.float().to(device).unsqueeze(0)
                    message_input = raw_output_embedding.clone()
                    message, ib_loss = manager.gateways[str(agent_id)](message_input)

                    messages[agent_id] = message
                    advantage, q_factual = manager.compute_advantage(state, messages, action)
                    manager.update_credit_graph(agent_id, advantage)

                    log_probs.append(log_prob)
                    q_values.append(q_factual)
                    advantages.append(advantage)
                    ib_losses.append(ib_loss)
                    rewards.append(0)

                    state = state_updater(message.detach(), state.detach())

                # ===================== 修改开始: 全新的评估逻辑 ======================
                # 1. 使用 MMLUDataset 的方法从原始输出中提取标准答案 (A, B, C, D)
                processed_answer = mmlu_eval_dataset.postprocess_answer(raw_output)

                # 2. 实例化 Accuracy 类
                accuracy_checker = Accuracy()

                # 3. 更新并获取准确率 (utility)
                accuracy_checker.update(processed_answer, answer)
                utility = accuracy_checker.get()  # 正确时为 1.0, 错误时为 0.0

                # 4. 根据 utility 计算最终 reward
                final_reward = 1.0 if utility == 1.0 else -1.0
                # ===================== 修改结束 ===================================

                rewards[-1] = final_reward
                batch_reward += final_reward
                is_correct = (final_reward == 1.0)
                if is_correct:
                    epoch_correct_predictions += 1
                epoch_total_questions += 1

                # (可选) 如果你想看每一道题的结果，取消下面这几行的注释
                print(f"\n--- Question {epoch_total_questions} : {'CORRECT' if is_correct else 'WRONG'} ---")
                print(f"Model Raw Output: {raw_output}")
                print(f"Processed Answer: {processed_answer}, Correct Answer: {answer}")
                # print(f"----------------\n")

                # Calculate returns for this item
                returns = []
                R = 0
                for r in reversed(rewards):
                    R = r + params['gamma'] * R
                    returns.insert(0, R)
                returns = torch.tensor(returns, dtype=torch.float32).to(device).unsqueeze(1)

                # Calculate losses for this item
                actor_loss = -(torch.stack(log_probs) * torch.stack(advantages).detach()).mean()
                critic_loss = F.smooth_l1_loss(torch.stack(q_values), returns)
                total_ib_loss = torch.stack(ib_losses).mean()

                total_loss_for_item = actor_loss + critic_loss + params['beta'] * total_ib_loss
                batch_loss += total_loss_for_item

            # --- End of batch loop ---

            # Average the loss over the batch and backpropagate
            if len(questions) > 0:
                avg_batch_loss = batch_loss / len(questions)
                avg_batch_loss.backward()
                optimizer.step()

                total_epoch_loss += avg_batch_loss.item()
                total_epoch_reward += batch_reward

        avg_loss = total_epoch_loss / max(1, len(dataloader))
        avg_reward = total_epoch_reward / max(1, len(dataset))  # Avg reward per sample
        accuracy = epoch_correct_predictions / max(1, epoch_total_questions)

        logging.info(
            f"Epoch {epoch + 1} finished. "
            f"Average Loss: {avg_loss:.4f}, "
            f"Average Reward: {avg_reward:.4f}, "
            f"Accuracy: {accuracy:.4f} ({epoch_correct_predictions}/{epoch_total_questions})"
        )
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