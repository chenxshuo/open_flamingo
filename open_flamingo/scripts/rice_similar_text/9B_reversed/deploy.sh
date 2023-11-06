# nohup bash open_flamingo/scripts/rice_similar_text/9B_reversed/deploy.sh > ./logs/rice_similar_text/deploy.text  2>&1 &

bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_coco.sh 4 26000 32 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-coco-shot-4.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_coco.sh 8 26000 16 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-coco-shot-8.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_coco.sh 16 26000 8 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-coco-shot-16.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_coco.sh 32 26000 4 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-coco-shot-32.text

bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_okvqa.sh 4 27001 32 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-okvqa-shot-4.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_okvqa.sh 8 27002 16 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-okvqa-shot-8.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_okvqa.sh 16 27003 8 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-okvqa-shot-16.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_okvqa.sh 32 27004 4 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-okvqa-shot-32.text

bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_vqav2.sh 4 26000 32 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-vqav2-shot-4.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_vqav2.sh 8 26001 16 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-vqav2-shot-8.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_vqav2.sh 16 26002 8 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-vqav2-shot-16.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_vqav2.sh 32 26003 4 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-vqav2-shot-32.text

bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_gqa.sh 4 26000 32 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-gqa-shot-4.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_gqa.sh 8 26000 16 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-gqa-shot-8.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_gqa.sh 16 26000 8 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-gqa-shot-16.text
bash open_flamingo/scripts/rice_similar_text/9B_reversed/run_eval_9B_reversed_gqa.sh 32 26000 4 &> ./logs/rice_similar_text/out-rice-img-9B_reversed-gqa-shot-32.text
