#!/usr/bin/env bash
set -e

# Training ensemble refiner for 1s 3-class classification

DATA_ROOT="./MPDD-Young"
MODEL1="./checkpoints/1s_3labels_wav2vec+openface/best_model_2025-07-21-01.48.03.pth"
MODEL2="./checkpoints/1s_3labels_mfccs+densenet/best_model_2025-07-20-21.47.37.pth"
MODEL3="./checkpoints/1s_3labels_mfccs+resnet/best_model_2025-07-21-02.12.04.pth"

OPT1="config_conformer.json"
OPT2="config_conformer.json"
OPT3="config_conformer.json"

python train_ensemble_refiner.py \
  --data_rootpath "$DATA_ROOT" \
  --feature_max_len 25 \
  --labelcount 3 \
  --batch_size 16 \
  --device cpu \
  --splitwindow_time 1s \
  --train_json "$DATA_ROOT/Training/labels/Training_Validation_files.json" \
  --personalized_features_file "$DATA_ROOT/Training/individualEmbedding/descriptions_embeddings_with_ids.npy" \
  --model_paths "$MODEL1" "$MODEL2" "$MODEL3" \
  --opt_paths "$OPT1" "$OPT2" "$OPT3" \
  --feature_pairs "wav2vec+openface" "mfccs+densenet" "mfccs+resnet"
