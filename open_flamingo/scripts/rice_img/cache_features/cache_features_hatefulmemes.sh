# bash open_flamingo/scripts/rice_img/cache_features/cache_features_coco.sh

export CUDA_VISIBLE_DEVICES=2
python open_flamingo/scripts/cache_rices_features.py \
  --vision_encoder_path ViT-L-14 \
  --vision_encoder_pretrained openai \
  --batch_size 1028 \
  --eval_hateful_memes \
  --hateful_memes_image_dir_path "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/hateful_memes/img" \
  --hateful_memes_train_annotations_json_path "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/hateful_memes/train.jsonl" \
  --hateful_memes_test_annotations_json_path "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/hateful_memes/dev.jsonl" \
  --output_dir "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/hateful_memes/rice_features"