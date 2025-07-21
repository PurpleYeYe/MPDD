# Audio Augmentation Dataset

This module provides three types of audio data augmentation strategies designed to enrich multimodal audio-visual datasets.

## Features

* **Three Augmentation Strategies**:

  * **Concatenation (`concat`)**: Combines two audio samples by concatenating them at a 75%:25% ratio.
  * **Mixup (`mixup`)**: Blends two audio samples using a weighted sum (75%:25%).
  * **Combined (`combined`)**: Applies both concat and mixup methods to each sample, generating two augmented versions per original sample.

* **Multi-scale Support**:

  * Supports both **1-second** and **5-second** temporal windows.
  * Compatible with multiple audio feature types: **MFCCs**, **OpenSmile**, and **Wav2Vec**.

## Usage

### Basic Command

```bash
python dataset.py \
  --json_path <path_to_original_json> \
  --audio_path <path_to_original_audio_features> \
  --video_path <path_to_original_video_features> \
  --aug_root <output_root_dir_for_augmented_data> \
  --personalized_file <path_to_personalized_features> \
  --aug_method <concat | mixup | combined> \
  [--is_elderly] \
  [--max_len 10] \
  [--log_dir logs]
```

### Argument Description

| Argument              | Required | Default  | Description                                           |
| --------------------- | -------- | -------- | ----------------------------------------------------- |
| `--json_path`         | Yes      | -        | Path to the original JSON annotation file             |
| `--audio_path`        | Yes      | -        | Path to the original audio features                   |
| `--video_path`        | Yes      | -        | Path to the original video features                   |
| `--aug_root`          | Yes      | -        | Root directory to save augmented data                 |
| `--personalized_file` | Yes      | -        | Path to the personalized feature file                 |
| `--aug_method`        | Yes      | `concat` | Augmentation method: `concat`, `mixup`, or `combined` |
| `--is_elderly`        | No       | -        | Use elderly dataset (default is young group)          |
| `--max_len`           | No       | `10`     | Maximum sequence length for features                  |
| `--log_dir`           | No       | `logs`   | Directory to save log files                           |

## Augmentation Strategy Details

### Concatenation (`concat`)

* Generates **2 augmented samples** per original sample.
* Each augmented sample is composed of **75% primary sample** + **25% auxiliary sample** (from the same subject).
* Requires at least **2 samples per subject**.

### Mixup (`mixup`)

* Generates **2 augmented samples** per original sample.
* Each augmented sample is a **weighted sum** of the primary and auxiliary sample (75% + 25%).
* Requires at least **2 samples per subject**.

### Combined (`combined`)

* Generates **2 augmented samples per original sample** (1 using concat, 1 using mixup).
* Requires at least **3 samples per subject**.

## Output Directory Structure

The augmented data will be stored under the specified `aug_root` directory with the following structure:

```
aug_root/
├── 1s/
│   ├── Audio/
│   │   ├── mfccs/
│   │   ├── opensmile/
│   │   └── wav2vec/
│   └── Video/         # (To be manually added by the user)
├── 5s/
│   ├── Audio/
│   │   ├── mfccs/
│   │   ├── opensmile/
│   │   └── wav2vec/
│   └── Video/         # (To be manually added by the user)
└── labels/
    └── augmented_data.json





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

