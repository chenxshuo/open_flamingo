# bash open_flamingo/scripts/rice_img/cache_features/cache_features_coco.sh
BASE_COCO_DATA_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO"
COCO_IMG_TRAIN_PATH="${BASE_COCO_DATA_PATH}/train2014"
BASE_DATA_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets"
OK_VQA_TRAIN_ANNO_PATH="${BASE_DATA_PATH}/okvqa/mscoco_train2014_annotations.json"
OK_VQA_TRAIN_QUES_PATH="${BASE_DATA_PATH}/okvqa/OpenEnded_mscoco_train2014_questions.json"
OUT_DIR="${BASE_DATA_PATH}/okvqa/rice_features"

export CUDA_VISIBLE_DEVICES=3
python open_flamingo/scripts/cache_rices_features.py \
  --vision_encoder_path ViT-L-14 \
  --vision_encoder_pretrained openai \
  --batch_size 1028 \
  --eval_ok_vqa \
  --ok_vqa_train_image_dir_path ${COCO_IMG_TRAIN_PATH} \
  --ok_vqa_train_annotations_json_path ${OK_VQA_TRAIN_ANNO_PATH} \
  --ok_vqa_train_questions_json_path ${OK_VQA_TRAIN_QUES_PATH} \
  --output_dir $OUT_DIR