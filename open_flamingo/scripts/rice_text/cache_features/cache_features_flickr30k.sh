# bash open_flamingo/scripts/rice_text/cache_features/cache_features_coco.sh
FLICKR_IMG_DIR="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/flickr30/flickr30k_images/flickr30k_images"
FLICKR_KARPATHY_JSON="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/flickr30k/dataset_flickr30k.json"
FLICKR_ANNOTATIONS_JSON="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/flickr30k/dataset_flickr30k_coco_style.json"
OUT_DIR="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/flickr30/rice_features"

export CUDA_VISIBLE_DEVICES=0
python open_flamingo/scripts/cache_ricestext_features.py \
  --batch_size 512 \
  --eval_flickr30 \
  --flickr_image_dir_path ${FLICKR_IMG_DIR} \
  --flickr_karpathy_json_path ${FLICKR_KARPATHY_JSON} \
  --flickr_annotations_json_path ${FLICKR_ANNOTATIONS_JSON} \
  --output_dir $OUT_DIR