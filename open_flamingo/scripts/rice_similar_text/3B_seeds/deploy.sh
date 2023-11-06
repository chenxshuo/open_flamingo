# nohup bash open_flamingo/scripts/rice_similar_text/3B_seeds/deploy.sh > ./logs/rice_similar_text/deploy.text  2>&1 &

bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_coco.sh 4 26000 16 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-coco-shot-4.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_coco.sh 8 26000 8 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-coco-shot-8.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_coco.sh 16 26000 8 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-coco-shot-16.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_coco.sh 32 26000 4 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-coco-shot-32.text

bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_okvqa.sh 4 27001 16 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-okvqa-shot-4.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_okvqa.sh 8 27002 8 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-okvqa-shot-8.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_okvqa.sh 16 27003 8 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-okvqa-shot-16.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_okvqa.sh 32 27004 4 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-okvqa-shot-32.text

bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_vqav2.sh 4 26000 16 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-vqav2-shot-4.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_vqav2.sh 8 26001 16 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-vqav2-shot-8.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_vqav2.sh 16 26002 8 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-vqav2-shot-16.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_vqav2.sh 32 26003 4 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-vqav2-shot-32.text

bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_gqa.sh 4 26000 16 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-gqa-shot-4.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_gqa.sh 8 26000 16 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-gqa-shot-8.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_gqa.sh 16 26000 8 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-gqa-shot-16.text
bash open_flamingo/scripts/rice_similar_text/3B_seeds/run_eval_3B_seeds_gqa.sh 32 26000 4 &> ./logs/rice_similar_text/out-rice-img-similar-text-3B_seeds-gqa-shot-32.text

