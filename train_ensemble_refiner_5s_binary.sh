#!/usr/bin/env bash
set -e

# Training ensemble refiner for 5s 2-class classification

DATA_ROOT="./MPDD-Young"
MODEL1="./checkpoints/5s_2labels_wav2vec+openface/best_model_2025-07-20-23.12.59.pth"
MODEL2="./checkpoints/5s_2labels_mfccs+densenet/best_model_2025-07-20-22.36.25.pth"
MODEL3="./checkpoints/5s_2labels_mfccs+resnet/best_model_2025-07-20-23.04.29.pth"

OPT1="config_conformer.json"
OPT2="config_conformer.json"
OPT3="config_conformer.json"

python train_ensemble_refiner.py \
  --data_rootpath "$DATA_ROOT" \
  --feature_max_len 6 \
  --labelcount 2 \
  --batch_size 24 \
  --device cpu \
  --splitwindow_time 5s \
  --train_json "$DATA_ROOT/Training/labels/Training_Validation_files.json" \
  --personalized_features_file "$DATA_ROOT/Training/individualEmbedding/descriptions_embeddings_with_ids.npy" \
  --model_paths "$MODEL1" "$MODEL2" "$MODEL3" \
  --opt_paths "$OPT1" "$OPT2" "$OPT3" \
  --feature_pairs "wav2vec+openface" "mfccs+densenet" "mfccs+resnet"
