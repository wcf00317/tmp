# experiments/standalone_vanilla_test.py
# 一个完全独立的脚本，使用openai库和qwen.env文件在MMLU上测试Qwen模型。
# 更新：现在可以从一个包含多个CSV文件的文件夹加载数据。

import os
import asyncio
import argparse
import pandas as pd
from tqdm.asyncio import tqdm
import re
import json
import hashlib
import logging
import glob  # 导入glob库用于查找文件

# 确保您已安装必要的库:
# pip install openai "pandas<2.0.0" python-dotenv tqdm
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- 日志和缓存设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CACHE_FILE = 'vanilla_openai_cache.json'
API_CACHE = {}


def load_cache():
    """从文件加载API响应缓存"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_cache():
    """将API响应缓存保存到文件"""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(API_CACHE, f, indent=4, ensure_ascii=False)


def get_cache_key(prompt: str, model_name: str) -> str:
    """为给定的文本和模型创建一个唯一的MD5哈希值"""
    return hashlib.md5(f"{model_name}:{prompt}".encode('utf-8')).hexdigest()


# 在脚本开始时加载缓存
API_CACHE = load_cache()


async def call_openai_compatible_api(client: AsyncOpenAI, model_name: str, prompt: str):
    """
    一个包装了OpenAI兼容API调用的异步函数，并集成了缓存功能。
    """
    cache_key = get_cache_key(prompt, model_name)
    if cache_key in API_CACHE:
        return API_CACHE[cache_key]

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        result = response.choices[0].message.content
        API_CACHE[cache_key] = result
        save_cache()
        return result
    except Exception as e:
        logging.error(f"调用API时出错: {e}")
        await asyncio.sleep(1)
        return None


async def process_row(row, client, model_name):
    """处理DataFrame中的单行数据"""
    prompt = (
        f"The following is a multiple-choice question. Please choose the single best answer.\n\n"
        f"Question: {row['question']}\n\n"
        f"Choices:\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}\n\n"
        f"Your final answer should be a single character: A, B, C or D."
    )

    raw_output = await call_openai_compatible_api(client, model_name, prompt)

    predicted_char = "Z"
    if isinstance(raw_output, str) and raw_output:
        match = re.search(r'([A-D])', raw_output)
        if match:
            predicted_char = match.group(1).upper()

    is_correct = (predicted_char == str(row['correct_answer']).strip().upper())
    return is_correct


async def main():
    parser = argparse.ArgumentParser(description="一个独立的脚本，用于在MMLU数据集上测试'Vanilla' Qwen模型。")
    # --- 修改点 ---
    parser.add_argument('--mmlu_data_path', type=str, required=True,
                        help="指向MMLU数据集文件夹的路径 (例如: './data/test')。")
    parser.add_argument('--model_name', type=str, default='qwen-max',
                        help="要使用的Qwen模型名称 (例如: 'qwen-max', 'qwen-turbo')。")
    parser.add_argument('--env_file', type=str, default='qwen.env',
                        help="包含API Key和Base URL的环境文件的路径。")
    parser.add_argument('--concurrency', type=int, default=5,
                        help="并发API请求的数量，以加快处理速度。")

    args = parser.parse_args()

    # --- 1. 从.env文件加载配置 ---
    if not os.path.exists(args.env_file):
        logging.error(f"错误: 环境文件 '{args.env_file}' 未找到。")
        return

    load_dotenv(dotenv_path=args.env_file)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL")

    if not api_key or not base_url:
        logging.error(f"错误: 未在 '{args.env_file}' 文件中找到 DASHSCOPE_API_KEY 或 DASHSCOPE_BASE_URL。")
        return

    # --- 2. 初始化OpenAI客户端 ---
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # --- 3. 读取和合并数据集 (核心修改部分) ---
    if not os.path.isdir(args.mmlu_data_path):
        logging.error(f"错误: 提供的路径不是一个文件夹: {args.mmlu_data_path}")
        return

    logging.info(f"正在从文件夹 {args.mmlu_data_path} 中扫描并加载所有 .csv 文件...")

    # 查找文件夹下所有的csv文件
    csv_files = glob.glob(os.path.join(args.mmlu_data_path, '*.csv'))

    if not csv_files:
        logging.error(f"错误: 在文件夹 {args.mmlu_data_path} 中没有找到任何 .csv 文件。")
        return

    # 读取所有csv文件并将它们合并成一个DataFrame
    df_list = []
    for file in csv_files:
        try:
            # MMLU的csv文件没有表头
            df_temp = pd.read_csv(file, header=None)
            df_list.append(df_temp)
        except Exception as e:
            logging.warning(f"读取文件 {file} 时出错: {e}")

    df = pd.concat(df_list, ignore_index=True)
    df.columns = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

    logging.info(f"成功加载并合并了 {len(csv_files)} 个文件, 共计 {len(df)} 条数据。")

    # --- 4. 并发评估 ---
    logging.info(f"开始使用模型 '{args.model_name}' 进行评估，并发数: {args.concurrency}...")

    tasks = [process_row(row, client, args.model_name) for index, row in df.iterrows()]
    results = await tqdm.gather(*tasks)

    correct_predictions = sum(results)
    total_predictions = len(df)

    # --- 5. 打印最终结果 ---
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    logging.info("=" * 50)
    logging.info("评估完成!")
    logging.info(f"模型: {args.model_name}")
    logging.info(f"数据集文件夹: {args.mmlu_data_path}")
    logging.info(f"总问题数: {total_predictions}")
    logging.info(f"正确回答数: {correct_predictions}")
    logging.info(f"最终准确率: {accuracy:.2f}%")
    logging.info("=" * 50)

    await client.close()


if __name__ == '__main__':
    asyncio.run(main())