# nohup bash open_flamingo/scripts/reproduction/deploy_missing_2.sh > out-reproduce.text 2>&1 &

#bash open_flamingo/scripts/rice_img/4B_seeds/run_eval_4B_seeds_gqa.sh 32 26000 2 &> ./logs/rice_img/out-rice-img-4B_seeds-gqa-shot-32.text

#bash open_flamingo/scripts/rice_similar_text/4BI_seeds/run_eval_4BI_seeds_gqa.sh 32 26000 2 &> ./logs/rice_similar_text/out-rice-img-4BI_seeds-gqa-shot-32.text

bash open_flamingo/scripts/rice_similar_text/4B_seeds/run_eval_4B_seeds_gqa.sh 32 26000 2 &> ./logs/rice_similar_text/out-rice-img-4B_seeds-gqa-shot-32.text
