# nohup bash open_flamingo/scripts/rice_similar_text/ablation_coco/deploy.sh > ./logs/rice_similar_text/deploy.text  2>&1 &

bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_4.sh 4 26000 32 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-4-4.text
bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_4.sh 8 26000 16 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-4-8.text
bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_4.sh 16 26000 8 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-4-16.text
bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_4.sh 32 26000 4 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-4-32.text

bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_8.sh 4 26000 32 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-8-4.text
bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_8.sh 8 26000 16 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-8-8.text
bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_8.sh 16 26000 8 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-8-16.text
bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_8.sh 32 26000 4 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-8-32.text

bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_16.sh 4 26000 32 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-16-4.text
bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_16.sh 8 26000 16 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-16-8.text
bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_16.sh 16 26000 8 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-16-16.text
bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_16.sh 32 26000 4 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-16-32.text






#bash open_flamingo/scripts/rice_similar_text/ablation_coco/run_eval_9B_coco_32.sh 32 26000 4 &> ./logs/rice_similar_text/out-rice-img-9B-coco-shot-32.text
