import streamlit as st
import yaml
from models.embedder import Embedder
from models.retriever import Retriever
from pipeline.rag_pipeline import RAGPipeline

@st.cache_resource
def load_pipeline():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    embedder = Embedder(cfg["embed_model"], device=cfg["device"])
    retriever = Retriever(cfg["faiss_index_path"], cfg["faiss_csv_path"])
    pipeline = RAGPipeline(embedder, retriever, cfg["llm_model"], cfg["device"])
    return pipeline, cfg

pipeline, cfg = load_pipeline()

st.title("中文 RAG 评论问答机器人")
query = st.text_input("请输入你的问题：")

if query:
    with st.spinner("生成中..."):
        answer = pipeline.run(query, top_k=cfg["top_k"], max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"])
        st.markdown("### 回答：")
        st.success(answer)
