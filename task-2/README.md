# **Task 2: Retrieval-Augmented Generation (RAG) Service**

A FastAPI-based Retrieval-Augmented Generation (RAG) service that combines document retrieval with text generation.

## **1. Environment Setup**

Create and activate the conda environment:

```bash
conda create -n task2 python=3.10 -y
conda activate task2
```

Clone the repository and install the dependencies:

```bash
git clone https://github.com/didepoyraz/Machine-Learning-Systems
cd Machine-Learning-Systems/task-2
pip install -r requirements.txt
```

## **2. Run the RAG Service**

### **Basic RAG Service**

Run:

```bash
python serving_rag.py
```

Test with:

```bash
python testing.py
```

### **Large Dataset RAG Service**

Run:

```bash
python serving_rag_datasets.py
```

Test with:

```bash
python testing_datasets.py
```

### **Auto-Scaling RAG Service with Load Balancing (Large Dataset)**

Run:

```bash
python serving_rag_datasets_auto.py
```

Test using the same script as the large dataset:

```bash
python testing_datasets.py
```

---