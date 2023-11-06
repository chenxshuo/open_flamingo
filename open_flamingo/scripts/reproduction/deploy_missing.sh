# nohup bash open_flamingo/scripts/reproduction/deploy_missing.sh > out-reproduce.text 2>&1 &

bash open_flamingo/scripts/reproduction/3B_seeds/run_eval_3B_seeds_coco.sh 0 26000 16 &> ./logs/out-reproduce-3B_seeds-coco-shot-0.text
bash open_flamingo/scripts/reproduction/3B_seeds/run_eval_3B_seeds_coco.sh 4 26001 16 &> ./logs/out-reproduce-3B_seeds-coco-shot-4.text
bash open_flamingo/scripts/reproduction/3B_seeds/run_eval_3B_seeds_coco.sh 8 26002 8 &> ./logs/out-reproduce-3B_seeds-coco-shot-8.text
bash open_flamingo/scripts/reproduction/3B_seeds/run_eval_3B_seeds_coco.sh 16 26003 6 &> ./logs/out-reproduce-3B_seeds-coco-shot-16.text
bash open_flamingo/scripts/reproduction/3B_seeds/run_eval_3B_seeds_coco.sh 32 26004 4 &> ./logs/out-reproduce-3B_seeds-coco-shot-32.text

bash open_flamingo/scripts/reproduction/4B_seeds/run_eval_4B_seeds_vqav2.sh 32 26004 2 &> ./logs/out-reproduce-4B_seeds-vqav2-shot-32.text
bash open_flamingo/scripts/reproduction/4B_seeds/run_eval_4B_seeds_gqa.sh 0 26001 16 &> ./logs/out-reproduce-4B_seeds-gqa-shot-0.text
bash open_flamingo/scripts/reproduction/4B_seeds/run_eval_4B_seeds_coco.sh 32 26004 2 &> ./logs/out-reproduce-4B_seeds-coco-shot-32.text

bash open_flamingo/scripts/rice_img/4BI_seeds/run_eval_4BI_seeds_okvqa.sh 32 26000 2 &> ./logs/rice_img/out-rice-img-4BI_seeds-okvqa-shot-32.text

bash open_flamingo/scripts/rice_img/4B_seeds/run_eval_4B_seeds_gqa.sh 4 26000 8 &> ./logs/rice_img/out-rice-img-4B_seeds-gqa-shot-4.text
bash open_flamingo/scripts/rice_img/4B_seeds/run_eval_4B_seeds_gqa.sh 32 26000 2 &> ./logs/rice_img/out-rice-img-4B_seeds-gqa-shot-32.text

bash open_flamingo/scripts/rice_img/3BI_seeds/run_eval_3BI_seeds_vqav2.sh 4 26000 16 &> ./logs/rice_img/out-rice-img-3BI_seeds-vqav2-shot-4.text
bash open_flamingo/scripts/rice_img/3BI_seeds/run_eval_3BI_seeds_vqav2.sh 8 26000 8 &> ./logs/rice_img/out-rice-img-3BI_seeds-vqav2-shot-8.text
bash open_flamingo/scripts/rice_img/3BI_seeds/run_eval_3BI_seeds_vqav2.sh 16 26000 4 &> ./logs/rice_img/out-rice-img-3BI_seeds-vqav2-shot-16.text
bash open_flamingo/scripts/rice_img/3BI_seeds/run_eval_3BI_seeds_vqav2.sh 32 26000 2 &> ./logs/rice_img/out-rice-img-3BI_seeds-vqav2-shot-32.text

bash open_flamingo/scripts/rice_similar_text/4BI_seeds/run_eval_4BI_seeds_gqa.sh 4 26000 8 &> ./logs/rice_similar_text/out-rice-img-4BI_seeds-gqa-shot-4.text
bash open_flamingo/scripts/rice_similar_text/4BI_seeds/run_eval_4BI_seeds_gqa.sh 32 26000 2 &> ./logs/rice_similar_text/out-rice-img-4BI_seeds-gqa-shot-32.text

bash open_flamingo/scripts/rice_similar_text/4B_seeds/run_eval_4B_seeds_gqa.sh 32 26000 2 &> ./logs/rice_similar_text/out-rice-img-4B_seeds-gqa-shot-32.text

bash open_flamingo/scripts/rice_similar_text/3BI_seeds/run_eval_3BI_seeds_okvqa.sh 4 27001 16 &> ./logs/rice_similar_text/out-rice-img-3BI_seeds-okvqa-shot-4.text
bash open_flamingo/scripts/rice_similar_text/3BI_seeds/run_eval_3BI_seeds_okvqa.sh 8 27002 8 &> ./logs/rice_similar_text/out-rice-img-3BI_seeds-okvqa-shot-8.text
bash open_flamingo/scripts/rice_similar_text/3BI_seeds/run_eval_3BI_seeds_okvqa.sh 16 27003 4 &> ./logs/rice_similar_text/out-rice-img-3BI_seeds-okvqa-shot-16.text
bash open_flamingo/scripts/rice_similar_text/3BI_seeds/run_eval_3BI_seeds_okvqa.sh 32 27004 2 &> ./logs/rice_similar_text/out-rice-img-3BI_seeds-okvqa-shot-32.text

bash open_flamingo/scripts/rice_similar_text/3BI_seeds/run_eval_3BI_seeds_vqav2.sh 4 26000 32 &> ./logs/rice_similar_text/out-rice-img-similar-text-3BI_seeds-vqav2-shot-4.text
bash open_flamingo/scripts/rice_similar_text/3BI_seeds/run_eval_3BI_seeds_vqav2.sh 8 26001 16 &> ./logs/rice_similar_text/out-rice-img-similar-text-3BI_seeds-vqav2-shot-8.text