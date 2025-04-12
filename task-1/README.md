# **Machine Learning Systems - Task 1**

## **1. Install Required Packages**

```bash
git clone https://github.com/didepoyraz/Machine-Learning-Systems
cd Machine-Learning-Systems-task-1
conda create -n task1 python=3.10 -y
conda activate task1
pip install -r requirements.txt
```

## **2. Run the Script in Interactive Mode**

To start the demo interactively, use:

```bash
srun --gres gpu:1 --pty task.py
```

## **3. Run the Script in Batch Mode**

Create a Bash script (e.g., `test.sh`) with the following content:

```bash
#!/bin/bash
source ~/miniconda3/bin/activate
conda activate mlsys
python task.py
```

Then submit the job using:

```bash
sbatch --gres gpu:1 test.sh
```

## **4. Argument Input**

The `task.py` file accepts two optional arguments:

- **`--distance`**: Distance function to use  
  Options: `cosine`, `dot`, `manhattan`, `l2`  
  Default: `cosine`

- **`--test`**: Test type to run  
  Options: `distance`, `knn`, `kmeans`, `ann`  
  Default: `knn`

### **Examples**

#### **Interactive Mode:**

Run KNN test with Manhattan distance:

```bash
srun --gres gpu:1 --pty task.py --distance manhattan --test knn
```

#### **Batch Mode:**

Update `test.sh` to:

```bash
#!/bin/bash
source ~/miniconda3/bin/activate
conda activate mlsys
python task.py --distance manhattan --test knn
```

Then submit with:

```bash
sbatch --gres gpu:1 test.sh
```
