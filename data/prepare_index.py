import os
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import faiss
import re
import psutil
import shutil

def clean_text(text: str) -> str:
    text = re.sub(r"[【】\[\]（）()《》<>]", "", str(text))
    text = re.sub(r"[§#￥@&*~^]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def convert_stars(row) -> int:
    """统计亮起的星星数，每列非空即计1"""
    return sum([str(row.get(f"stars{i}", "")).strip() != "" for i in range(1, 6)])

def prepare_dataframe(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    df["comment"] = df["comment"].astype(str).fillna("").str.strip()
    df = df[df["comment"].str.len() >= 5].copy()
    df["clean_text"] = df["comment"].apply(clean_text)

    # 确保stars字段存在
    for i in range(1, 6):
        if f"stars{i}" not in df.columns:
            df[f"stars{i}"] = ""

    df["star_rating"] = df.apply(convert_stars, axis=1)

    return df[["id", "author", "comment", "clean_text", "star_rating"]]

def build_index(input_csv, output_dir, model_name, device):
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    print("🔹 加载并清洗原始数据...")
    df = prepare_dataframe(input_csv)
    sentences = df["clean_text"].tolist()
    print(f"🔹 清洗后评论数: {len(sentences)}")

    print("🔹 加载向量模型:", model_name)
    model = SentenceTransformer(model_name, device=device)

    batch_size = 64
    embeddings = []

    print("🔹 开始向量化（Batch Size = %d）..." % batch_size)
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch = sentences[i:i + batch_size]
            vecs = model.encode(batch, convert_to_numpy=True)
            embeddings.append(vecs)

    embeddings = np.vstack(embeddings)
    print(f"✅ 向量化完成：共 {embeddings.shape[0]} 条，维度 {embeddings.shape[1]}")

    # 构建索引
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # 保存索引
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.index"))
    df.to_csv(os.path.join(output_dir, "faiss_docs.csv"), index=False)

    # 不覆盖原文件
    original_name = os.path.basename(input_csv)
    if not input_csv.startswith(output_dir):
        shutil.copy(input_csv, os.path.join(output_dir, original_name))

    # 资源统计
    ram = psutil.virtual_memory().used / 1024 / 1024
    cpu = time.time() - start_time
    print(f"🧠 内存使用：{ram:.2f} MB")
    print(f"⏱️ 总耗时：{cpu:.2f} 秒")

    with open(os.path.join(output_dir, "resource_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"内存使用: {ram:.2f} MB\n")
        f.write(f"耗时: {cpu:.2f} 秒\n")
        f.write(f"数据条数: {len(sentences)}\n")
        f.write(f"向量维度: {embeddings.shape[1]}\n")

if __name__ == "__main__":
    # 自动获取当前脚本目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    default_csv_path = os.path.join(BASE_DIR, "bilibili_gzxb.csv")
    default_output_dir = os.path.join(BASE_DIR, "..", "vector_index")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default=default_csv_path,
        help="原始数据文件路径，默认脚本同目录下的bilibili_gzxb.csv"
    )
    parser.add_argument(
        "--output_dir",
        default=default_output_dir,
        help="向量索引和清洗数据保存路径，默认项目根目录vector_index文件夹"
    )
    parser.add_argument(
        "--model_name",
        default="BAAI/bge-large-zh-v1.5",
        help="SentenceTransformer模型名"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="运行设备，默认自动检测"
    )
    args = parser.parse_args()

    # 自动设备选择：优先cuda，否则cpu
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "cuda":
            print("检测到可用CUDA，使用GPU加速。")
        else:
            print("未检测到CUDA，使用CPU运行。")

    build_index(args.input_csv, args.output_dir, args.model_name, args.device)
