import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading
import queue
import time
import psutil
import logging
import os
import faiss
from typing import List
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-service")

# Constants
MAX_BATCH_SIZE = 16
MAX_WAITING_TIME = 0.1
MIN_WORKERS = 1
MAX_WORKERS = 8
SCALING_COOLDOWN_PERIOD = 60
HEALTH_CHECK_INTERVAL = 10

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
from datasets import load_dataset
logger.info('Loading dataset')
dataset = load_dataset('MAsad789565/Coding_GPT4_Data', split='train', trust_remote_code=True)
documents = [example['assistant'] for example in dataset]
logger.info(f'Loaded {len(documents)} documents')

# Embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
embed_model.eval()
logger.info("Embedding model loaded")

# Chat model
chat_pipeline = pipeline("text-generation", model="facebook/opt-125m", device=0 if torch.cuda.is_available() else -1)
logger.info("Chat model loaded")

# Load document embeddings and FAISS index
index_path = "faiss_index.index"

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    logger.info("FAISS index loaded from disk")
else:
    doc_embeddings = np.load("doc_embeddings.npy").astype("float32")
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    faiss.write_index(index, index_path)
    logger.info("FAISS index created and saved")

# Request queue
request_queue = queue.Queue()

# Worker metrics
class WorkerMetrics:
    def __init__(self):
        self.active_workers = 0
        self.total_requests_processed = 0
        self.requests_per_minute = 0
        self.average_processing_time = 0.0
        self.last_scaling_time = time.time()
        self.request_timestamps = []
        self.processing_times = []

worker_metrics = WorkerMetrics()
worker_locks = threading.Lock()

# Worker control
worker_threads = []
worker_controls = []  # List of {"thread": ..., "stop_event": ..., "ready_event": ...}

# Utilities
@lru_cache(maxsize=1024)
def get_cached_embedding(text: str) -> np.ndarray:
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def get_embeddings(texts: List[str]) -> np.ndarray:
    inputs = embed_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> List[str]:
    D, I = index.search(query_emb, k)
    return [[documents[i] for i in ii] for ii in I]

def rag_batch_pipeline(queries: List[str], k: int = 2) -> List[str]:
    start_time = time.time()
    query_embs = get_embeddings(queries)
    top_k_docs = retrieve_top_k(query_embs, k)
    prompts = [f"Question: {q}\nContext:\n{chr}\nAnswer:" for q, chr in zip(queries, ["\n".join(c) for c in top_k_docs])]
    results = chat_pipeline(prompts, max_length=50, do_sample=True)
    responses = [res[0]["generated_text"] for res in results]
    processing_time = time.time() - start_time
    with worker_locks:
        worker_metrics.processing_times.append(processing_time)
        worker_metrics.processing_times = worker_metrics.processing_times[-100:]
        worker_metrics.average_processing_time = sum(worker_metrics.processing_times) / len(worker_metrics.processing_times)
    return responses

# Request schema
class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
def predict(payload: QueryRequest):
    with worker_locks:
        now = time.time()
        worker_metrics.request_timestamps.append(now)
        worker_metrics.request_timestamps = [ts for ts in worker_metrics.request_timestamps if now - ts <= 60]
        worker_metrics.requests_per_minute = len(worker_metrics.request_timestamps)

    response_queue = queue.Queue()
    request_queue.put((payload, response_queue))
    try:
        result = response_queue.get(timeout=10)
    except queue.Empty:
        result = "Timeout: No response from worker."

    with worker_locks:
        worker_metrics.total_requests_processed += 1

    return {"query": payload.query, "result": result}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "metrics": {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "active_workers": worker_metrics.active_workers,
            "queue_size": request_queue.qsize(),
            "requests_per_minute": worker_metrics.requests_per_minute,
            "total_requests_processed": worker_metrics.total_requests_processed,
            "average_processing_time": worker_metrics.average_processing_time
        }
    }

def worker(stop_event: threading.Event, ready_event: threading.Event):
    worker_id = threading.get_ident()
    logger.info(f"Worker {worker_id} starting...")

    with worker_locks:
        worker_metrics.active_workers += 1

    try:
        time.sleep(0.5)  # simulate warm-up if needed
        ready_event.set()
        logger.info(f"Worker {worker_id} is ready")

        while not stop_event.is_set():
            batch = []
            start_time = time.time()
            while len(batch) < MAX_BATCH_SIZE and (time.time() - start_time) < MAX_WAITING_TIME:
                try:
                    req = request_queue.get(timeout=MAX_WAITING_TIME)
                    batch.append(req)
                except queue.Empty:
                    break

            if batch:
                queries = [req[0].query for req in batch]
                ks = [req[0].k for req in batch]
                k = max(ks) if ks else 2
                results = rag_batch_pipeline(queries, k)

                for (_, resp_q), result in zip(batch, results):
                    resp_q.put(result)
                    request_queue.task_done()
    finally:
        with worker_locks:
            worker_metrics.active_workers -= 1
        logger.info(f"Worker {worker_id} exited")

def autoscaler():
    logger.info("Autoscaler started")
    while True:
        time.sleep(HEALTH_CHECK_INTERVAL)
        if time.time() - worker_metrics.last_scaling_time < SCALING_COOLDOWN_PERIOD:
            continue

        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        queue_size = request_queue.qsize()

        with worker_locks:
            curr_workers = worker_metrics.active_workers
            rpm = worker_metrics.requests_per_minute

        target = curr_workers
        if cpu > 80 or mem > 80 or (queue_size > curr_workers * 5 and rpm > curr_workers * 30):
            target = min(curr_workers + 1, MAX_WORKERS)
        elif cpu < 20 and mem < 20 and queue_size < 2 and rpm < curr_workers * 10:
            target = max(curr_workers - 1, MIN_WORKERS)

        if target > curr_workers:
            logger.info(f"Scaling up: {curr_workers} -> {target}")
            for _ in range(target - curr_workers):
                stop_event = threading.Event()
                ready_event = threading.Event()
                t = threading.Thread(target=worker, args=(stop_event, ready_event), daemon=True)
                t.start()
                ready_event.wait(timeout=10)
                worker_controls.append({"thread": t, "stop_event": stop_event, "ready_event": ready_event})
            with worker_locks:
                worker_metrics.last_scaling_time = time.time()
        elif target < curr_workers:
            logger.info(f"Scaling down: {curr_workers} -> {target}")
            to_stop = worker_controls[target:]
            worker_controls[:] = worker_controls[:target]
            for ctrl in to_stop:
                ctrl["stop_event"].set()
            with worker_locks:
                worker_metrics.last_scaling_time = time.time()

@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    # Placeholder for future rate limiting
    return await call_next(request)

if __name__ == "__main__":
    for _ in range(MIN_WORKERS):
        stop_event = threading.Event()
        ready_event = threading.Event()
        t = threading.Thread(target=worker, args=(stop_event, ready_event), daemon=True)
        t.start()
        ready_event.wait(timeout=10)
        worker_controls.append({"thread": t, "stop_event": stop_event, "ready_event": ready_event})

    threading.Thread(target=autoscaler, daemon=True).start()
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8002)