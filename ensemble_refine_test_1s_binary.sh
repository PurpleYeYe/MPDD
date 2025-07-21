#!/usr/bin/env bash
set -e

# Ensemble inference with refiner model for 1s binary classification

DATA_ROOT="./MPDD-Young-Test"
MODEL1="./checkpoints/1s_2labels_wav2vec+openface/best_model_2025-06-25-16.06.42.pth"
MODEL2="./checkpoints/1s_2labels_mfccs+densenet/best_model_2025-06-25-15.57.25.pth"
MODEL3="./checkpoints/1s_2labels_mfccs+resnet/best_model_2025-06-25-15.46.37.pth"

OPT1="config_conformer.json"
OPT2="config_conformer.json"
OPT3="config_conformer.json"

REFINER_MODEL="./checkpoints/ensemble_refiner_1s_2labels_2025-07-07-124424.pth"
OUTPUT_CSV="answer_Track2/ensemble_submission_1s_binary.csv"

python ensemble_inference_test.py \
  --data_rootpath "$DATA_ROOT" \
  --feature_max_len 25 \
  --labelcount 2 \
  --batch_size 16 \
  --device cpu \
  --splitwindow_time 1s \
  --test_json "$DATA_ROOT/Testing/labels/Testing_files.json" \
  --personalized_features_file "$DATA_ROOT/Testing/individualEmbedding/descriptions_embeddings_with_ids.npy" \
  --model_paths "$MODEL1" "$MODEL2" "$MODEL3" \
  --opt_paths "$OPT1" "$OPT2" "$OPT3" \
  --feature_pairs "wav2vec+openface" "mfccs+densenet" "mfccs+resnet" \
  --refiner_model "$REFINER_MODEL" \
  --output_csv "$OUTPUT_CSV"
