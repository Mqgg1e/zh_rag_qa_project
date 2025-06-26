from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class RAGPipeline:
    def __init__(self, embedder, retriever, llm_model_name, device):
        self.embedder = embedder
        self.retriever = retriever
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name, trust_remote_code=True).to(device).eval()
        self.device = device

    def run(self, query: str, top_k: int = 3, **gen_kwargs) -> str:
        query_vec = self.embedder.encode([query])
        context = "\n".join(self.retriever.search(query_vec, top_k))
        prompt = f"根据以下评论内容，简洁回答问题：{query}\n评论内容：\n{context}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
