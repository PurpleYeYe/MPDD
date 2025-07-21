# Environment

    python 3.10.0
    pytorch 2.3.0
    scikit-learn 1.5.1
    pandas 2.2.2

Given `requirements.txt`, we recommend users to configure their environment via conda with the following steps:

    conda create -n mpdd python=3.10 -y   
    conda activate mpdd  
    pip install -r requirements.txt 


# Usage

Before the training and testing steps, we need to set the following dataset path:

`The path of the training set data`:
├──  MPDD-Young/
│   ├── Training/
│   │   ├──1s
│   │   ├──5s
│   │   ├──individualEmbedding
│   │   ├──labels

`Test set data path`:
├──  MPDD-Young-Test/ 
│  ├── Testing/
│  │   ├──1s
│  │   ├──5s
│  │   ├──individualEmbedding
│  │   ├──labels


## Training
The team has set the training parameters, and the user only needs to run it in the main directory: 

##Step 1:Expert model training

```bash
cd path/MPDD   # Replace with the actual primary path
```
```bash
bash train_1s_binary_model1.sh
```
Run in order: 
train_1s_binary_model1.sh, train_1s_binary_model2.sh, train_1s_binary_model3.sh, train_1s_ternary_model1.sh, train_1s_ternary_model2.sh, train_1s_ternary_model3.sh, train_5s_binary_model1.sh, train_5s_binary_model2.sh, train_5s_binary_model3.sh, train_5s_ternary_model1.sh, train_5s_ternary_model2.sh, train_5s_ternary_model3.sh
A total of 12 documents.

##Step 2:Converged network training

After the training of the above 12 models is completed, the second step of training will be carried out:
```bash
bash train_ensemble_refiner_1s_binary.sh
```
Run in order:train_ensemble_refiner_1s_binary.sh, train_ensemble_refiner_1s_ternary.sh, train_ensemble_refiner_5s_binary.sh, train_ensemble_refiner_5s_ternary.sh
Note: In the file code, you need to modify the path of the model saved in the first step of training.

## Testing
After the model is trained, you can test it (the relevant model weights are already saved in ./checkpoints/), Please run as follows:

```bash
cd path/MPDD   # Replace with the actual primary path
```
```bash
bash ensemble_refine_test_1s_binary.sh
```
Run in order:ensemble_refine_test_1s_binary.sh, ensemble_refine_test_1s_ternary.sh, ensemble_refine_test_5s_binary.sh, ensemble_refine_test_5s_ternary.sh
Note: There is a model path in the file code here that needs to be modified

After the above four files are run in turn, the results are merged into the submission.csv file in './answer_Track2/'.

