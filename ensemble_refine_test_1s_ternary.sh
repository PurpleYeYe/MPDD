#!/usr/bin/env bash
set -e

# Ensemble inference with refiner model for 1s 3-class classification

DATA_ROOT="./MPDD-Young-Test"
MODEL1="./checkpoints/1s_3labels_wav2vec+openface/best_model_2025-06-25-17.14.27.pth"
MODEL2="./checkpoints/1s_3labels_mfccs+densenet/best_model_2025-06-25-16.56.38.pth"
MODEL3="./checkpoints/1s_3labels_mfccs+resnet/best_model_2025-06-25-16.36.23.pth"

OPT1="config_conformer.json"
OPT2="config_conformer.json"
OPT3="config_conformer.json"

REFINER_MODEL="./checkpoints/ensemble_refiner_1s_3labels_2025-07-07-153859.pth"
OUTPUT_CSV="answer_Track2/ensemble_submission_1s_ternary.csv"

python ensemble_inference_test.py \
  --data_rootpath "$DATA_ROOT" \
  --feature_max_len 25 \
  --labelcount 3 \
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
