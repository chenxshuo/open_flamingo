# bash open_flamingo/scripts/rice_img/cache_features/cache_features_coco.sh
VIZWIZ_TRAIN_IMG="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vizwiz/train"
VIZWIZ_VAL_IMG="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vizwiz/val"

ANNO_BASE="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/vizwiz"
VIZWIZ_TRAIN_QUES_PATH="${ANNO_BASE}/train_questions_vqa_format.json"
VIZWIZ_TRAIN_ANNO_PATH="${ANNO_BASE}/train_annotations_vqa_format.json"
OUT_DIR="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vizwiz/rice_features"

export CUDA_VISIBLE_DEVICES=5
python open_flamingo/scripts/cache_rices_features.py \
  --vision_encoder_path ViT-L-14 \
  --vision_encoder_pretrained openai \
  --batch_size 1028 \
  --eval_vizwiz \
  --vizwiz_train_image_dir_path ${VIZWIZ_TRAIN_IMG} \
  --vizwiz_train_questions_json_path ${VIZWIZ_TRAIN_QUES_PATH} \
  --vizwiz_train_annotations_json_path ${VIZWIZ_TRAIN_ANNO_PATH} \
  --output_dir $OUT_DIR