### **1. Install Required Packages**
Run:
```bash
git clone https://github.com/didepoyraz/Machine-Learning-Systems
cd Machine-Learning-Systems-task-1
conda create -n task1 python=3.10 -y
conda activate task1
pip install -r requirements.txt

### **2. Run the script in Interactive Mode**
To start the demo interactively, use:
```bash
srun --gres gpu:1 --pty task.py

### **3. Run the script in Batch Mode**
To run the demo in batch mode, first create a Bash script (e.g., `test.sh`) with the following content:

```bash
#!/bin/bash
source ~/miniconda3/bin/activate
conda activate mlsys
python task.py
```

Then submit the job using:
```bash
sbatch --gres gpu:1 test.sh

### **4. Argument input
The task.py file requires distance function and test input.
The distance function options are: 'cosine', 'dot, 'manhattan' and 'l2.
The test arguments are: 'distance', 'knn', 'kmeans' and 'ann'.
If no arguments are specified, the script will automatically run: distance: 'cosine' and test 'knn'.

For example, to run the knn test with manhattan distance, run:
in interactive mode:
srun --gres gpu:1 --pty task.py --distance manhattan --test knn

in batch mode:
```bash
#!/bin/bash
source ~/miniconda3/bin/activate
conda activate mlsys
python task.py --distance manhattan --test knn
```

Then submit the job using:
```bash
sbatch --gres gpu:1 test.sh
