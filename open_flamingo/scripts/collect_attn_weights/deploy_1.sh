# nohup bash open_flamingo/scripts/collect_attn_weights/deploy_1.sh > ./logs/out_collect_attn_weights-4B_deploy_1.sh 2>&1 &

#bash open_flamingo/scripts/collect_attn_weights/run_eval_4B_okvqa.sh  4 26001 7 False False &> ./logs/out-collect_attn_weights-4B-okvqa-random-shot-4.text
#bash open_flamingo/scripts/collect_attn_weights/run_eval_4B_okvqa.sh  4 26001 7 True False &> ./logs/out-collect_attn_weights-4B-okvqa-no_images-shot-4.text
#bash open_flamingo/scripts/collect_attn_weights/run_eval_4B_okvqa.sh  4 26001 7 False True &> ./logs/out-collect_attn_weights-4B-okvqa-no_query_image-shot-4.text

#bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_okvqa.sh  4 26001 7 False False &> ./logs/out-collect_attn_weights-9B-okvqa-random-shot-4.text
#bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_okvqa.sh  4 26001 7 True False &> ./logs/out-collect_attn_weights-9B-okvqa-no_images-shot-4.text
#bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_okvqa.sh  4 26001 7 False True &> ./logs/out-collect_attn_weights-9B-okvqa-no_query_image-shot-4.text

bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_coco.sh  4 26001 7 False False &> ./logs/out-collect_attn_weights-9B-coco-random-shot-4.text
bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_coco.sh  4 26001 7 True False &> ./logs/out-collect_attn_weights-9B-coco-no_images-shot-4.text
bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_coco.sh  4 26001 7 False True &> ./logs/out-collect_attn_weights-9B-coco-no_query_image-shot-4.text

#bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_gqa.sh  4 26001 7 False False &> ./logs/out-collect_attn_weights-9B-gqa-random-shot-4.text
#bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_gqa.sh  4 26001 7 True False &> ./logs/out-collect_attn_weights-9B-gqa-no_images-shot-4.text
#bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_gqa.sh  4 26001 7 False True &> ./logs/out-collect_attn_weights-9B-gqa-no_query_image-shot-4.text

#bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_vqav2.sh  4 26001 7 False False &> ./logs/out-collect_attn_weights-9B-vqav2-random-shot-4.text
#bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_vqav2.sh  4 26001 7 True False &> ./logs/out-collect_attn_weights-9B-vqav2-no_images-shot-4.text
#bash open_flamingo/scripts/collect_attn_weights/run_eval_9B_vqav2.sh  4 26001 7 False True &> ./logs/out-collect_attn_weights-9B-vqav2-no_query_image-shot-4.text
