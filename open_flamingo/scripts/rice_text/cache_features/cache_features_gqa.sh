# bash open_flamingo/scripts/rice_text/cache_features/cache_features_gqa.sh
ANNO_BASE="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/gqa"
GQA_IMG="${ANNO_BASE}/images"
GQA_TRAIN_QUES_PATH="${ANNO_BASE}/train_ques_vqav2_format.json"
GQA_TRAIN_ANNO_PATH="${ANNO_BASE}/train_anno_vqav2_format.json"
OUT_DIR="${ANNO_BASE}/rice_features"

export CUDA_VISIBLE_DEVICES=1
python open_flamingo/scripts/cache_ricestext_features.py \
  --batch_size 512 \
  --eval_gqa \
  --gqa_image_dir_path ${GQA_IMG} \
  --gqa_train_questions_json_path ${GQA_TRAIN_QUES_PATH} \
  --gqa_train_annotations_json_path ${GQA_TRAIN_ANNO_PATH} \
  --output_dir $OUT_DIR