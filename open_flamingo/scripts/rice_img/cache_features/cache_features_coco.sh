# bash open_flamingo/scripts/rice_img/cache_features/cache_features_coco.sh

BASE_COCO_DATA_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO"
COCO_IMG_TRAIN_PATH="${BASE_COCO_DATA_PATH}/train2014"
COCO_IMG_VAL_PATH="${BASE_COCO_DATA_PATH}/val2014"
COCO_ANNO_PATH="${BASE_COCO_DATA_PATH}/annotations-2014/captions_val2014.json"
COCO_KARPATHY_PATH="${BASE_COCO_DATA_PATH}/dataset_coco.json"
COCO_OUT_DIR="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/coco/rice_features"


export CUDA_VISIBLE_DEVICES=0
python open_flamingo/scripts/cache_rices_features.py \
  --vision_encoder_path ViT-L-14 \
  --vision_encoder_pretrained openai \
  --batch_size 1028 \
  --eval_coco \
  --coco_train_image_dir_path $COCO_IMG_TRAIN_PATH \
  --coco_val_image_dir_path $COCO_IMG_VAL_PATH \
  --coco_karpathy_json_path $COCO_KARPATHY_PATH \
  --coco_annotations_json_path $COCO_ANNO_PATH \
  --output_dir $COCO_OUT_DIR