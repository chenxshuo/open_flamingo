# nohup bash open_flamingo/scripts/no_images/deploy_2.sh > out_no_images_deploy_2.sh 2>&1 &

bash open_flamingo/scripts/no_images/run_eval_9B_coco.sh  4 26001 64 blank_images &> ./out-blank-images-9B-coco-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_coco.sh  8 26002 32 blank_images &> ./out-blank-images-9B-coco-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_coco.sh  16 26003 16 blank_images &> ./out-blank-images-9B-coco-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_coco.sh  32 26004 8 blank_images &> ./out-blank-images-9B-coco-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_coco.sh  4 26001 64 ood_images &> ./out-ood-images-9B-coco-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_coco.sh  8 26002 32 ood_images &> ./out-ood-images-9B-coco-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_coco.sh  16 26003 16 ood_images &> ./out-ood-images-9B-coco-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_coco.sh  32 26004 8 ood_images &> ./out-ood-images-9B-coco-shot-32.text


bash open_flamingo/scripts/no_images/run_eval_9B_flickr30k.sh  4 26001 64 blank_images &> ./out-blank-images-9B-flickr30k-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_flickr30k.sh  8 26002 32 blank_images &> ./out-blank-images-9B-flickr30k-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_flickr30k.sh  16 26003 16 blank_images &> ./out-blank-images-9B-flickr30k-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_flickr30k.sh  32 26004 8 blank_images &> ./out-blank-images-9B-flickr30k-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_flickr30k.sh  4 26001 64 ood_images &> ./out-ood-images-9B-flickr30k-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_flickr30k.sh  8 26002 32 ood_images &> ./out-ood-images-9B-flickr30k-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_flickr30k.sh  16 26003 16 ood_images &> ./out-ood-images-9B-flickr30k-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_flickr30k.sh  32 26004 8 ood_images &> ./out-ood-images-9B-flickr30k-shot-32.text

