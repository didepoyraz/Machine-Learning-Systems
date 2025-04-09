import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import threading
import queue
import time


MAX_BATCH_SIZE = 16
MAX_WAITING_TIME = 0.1 #50ms? adjust this
app = FastAPI()

# Example documents in memory
# documents = [
#    "Cats are small furry carnivores that are often kept as pets.",
#    "Dogs are domesticated mammals, not natural wild animals.",
#    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
#]

# different dataset
from datasets import load_dataset

print('loading dataset')
# Load the dataset
# dataset = load_dataset('MAsad789565/Coding_GPT4_Data', split='train')
dataset = load_dataset('MAsad789565/Coding_GPT4_Data', split='train', trust_remote_code=True)

# If needed, specify a different split such as 'validation' or 'test'

# Extract content from the dataset
documents = [example['assistant'] for example in dataset]
print('dataset loaded')

# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
print("loaded embedding model")

# Basic Chat LLM
chat_pipeline = pipeline("text-generation", model="facebook/opt-125m")
# Note: try this 1.5B model if you got enough GPU memory
# chat_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

print("starting request queue")
request_queue = queue.Queue()

doc_embeddings = np.load("doc_embeddings.npy")
print("loaded embeddings")
## Hints:

### Step 3.1:
# 1. Initialize a request queue
# 2. Initialize a background thread to process the request (via calling the rag_pipeline function)
# 3. Modify the predict function to put the request in the queue, instead of processing it immediately

### Step 3.2:
# 1. Take up to MAX_BATCH_SIZE requests from the queue or wait until MAX_WAITING_TIME
# 2. Process the batched requests

def get_embedding(text: str) -> np.ndarray:
    # print("getting embeddings.")
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
# doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])
# np.save("doc_embeddings.npy", doc_embeddings)

### You may want to use your own top-k retrieval method (task 1)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    # print("Retrieving top-k")
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

def rag_pipeline(query: str, k: int = 2) -> str:
    # print("in rag pipeline")
    # whenever there is a new request the worker thread will receive it here
    # Step 1: Input embedding
    query_emb = get_embedding(query)
    
    # Step 2: Retrieval
    retrieved_docs = retrieve_top_k(query_emb, k)
    
    # Construct the prompt from query + retrieved docs
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    
    # Step 3: LLM Output
    generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    return generated

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag") 
def predict(payload: QueryRequest):
    # result = rag_pipeline(payload.query, payload.k)
    # print("predicting")
    response_queue = queue.Queue()
    request_queue.put((payload, response_queue))
    
    result = response_queue.get()
    
    return {
        "query": payload.query,
        "result": result,
    }
    
def worker():
    print("👷 Worker thread started")
    while True:
        batch = [] 
        start_time = time.time()
        
        while len(batch) < MAX_BATCH_SIZE and (time.time() - start_time) < MAX_WAITING_TIME:
            try:
                req = request_queue.get(timeout=MAX_WAITING_TIME) # the timeout here is for if the queue is empty it should break
                batch.append(req)
            except:
                break  # timeout hit but queue empty
            
        if batch:
            for payload, response_queue in batch:
                result = rag_pipeline(payload.query, payload.k)
                response_queue.put(result)
                
                request_queue.task_done() # i do not know if these are needed because i do not use join
                response_queue.task_done()      
        
if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8002)