# bash open_flamingo/scripts/rice_text/cache_features/cache_features_coco.sh
TEXTVQA_IMG_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/textvqa/train_val_images/train_images"
BASE_JSON_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/textvqa"
TEXTVQA_TRAIN_QUES="${BASE_JSON_PATH}/train_questions_vqa_format.json"
TEXTVQA_TRAIN_ANNO="${BASE_JSON_PATH}/train_annotations_vqa_format.json"
OUT_DIR="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/textvqa/rice_features"

export CUDA_VISIBLE_DEVICES=3
python open_flamingo/scripts/cache_ricestext_features.py \
  --batch_size 512 \
  --eval_textvqa \
  --textvqa_image_dir_path ${TEXTVQA_IMG_PATH} \
  --textvqa_train_questions_json_path ${TEXTVQA_TRAIN_QUES} \
  --textvqa_train_annotations_json_path ${TEXTVQA_TRAIN_ANNO} \
  --output_dir $OUT_DIR