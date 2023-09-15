# bash open_flamingo/scripts/rice_img/cache_features/cache_features_coco.sh

BASE_COCO_DATA_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO"
BASE_VQAv2_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2"
VQAV2_IMG_TRAIN_PATH="${BASE_COCO_DATA_PATH}/train2014/"
VQAV2_ANNO_TRAIN_PATH="${BASE_VQAv2_PATH}/v2_mscoco_train2014_annotations.json"
VQAV2_QUESTION_TRAIN_PATH="${BASE_VQAv2_PATH}/v2_OpenEnded_mscoco_train2014_questions.json"
VQAv2_OUT_DIR="${BASE_VQAv2_PATH}/rice_features"

export CUDA_VISIBLE_DEVICES=6
python open_flamingo/scripts/cache_rices_features.py \
  --vision_encoder_path ViT-L-14 \
  --vision_encoder_pretrained openai \
  --batch_size 1028 \
  --eval_vqav2 \
  --vqav2_train_image_dir_path ${VQAV2_IMG_TRAIN_PATH} \
  --vqav2_train_annotations_json_path ${VQAV2_ANNO_TRAIN_PATH} \
  --vqav2_train_questions_json_path ${VQAV2_QUESTION_TRAIN_PATH} \
  --output_dir $VQAv2_OUT_DIR