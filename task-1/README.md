# **Machine Learning Systems - Task 1**

## **1. Install Required Packages**

```bash
git clone https://github.com/didepoyraz/Machine-Learning-Systems
cd Machine-Learning-Systems/task-1
conda create -n task1 python=3.10 -y
conda activate task1
pip install -r requirements.txt
```
## **2. Generate Test Data**

To generate data to perform tests, run:

```bash
python json_test_file_generation.py
```

To create your own test case:
1. Add a function call to json_test_generation.py with input values for the function as needed, along with the name of the file to store it as.
   - generate_and_save_knn generates random data for distance and KNN tests of N vectors with D dimensions.
   - generate_and_save_kmeans generates blob data of N vectors with D dimension in K number of clusters.
   - generate_and_save_ann generates blob data of N vectors with D dimension in K number of clusters along with a vector X.
2. Add a function to task.py which reads from the file generated above and calls the respective function.
3. Call the respective function in the main method.

## **3. Run the Script in Interactive Mode**

To start the demo interactively, use:

```bash
srun --gres gpu:1 --pty task.py
```

## **4. Run the Script in Batch Mode**

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

## **5. Argument Input**

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
