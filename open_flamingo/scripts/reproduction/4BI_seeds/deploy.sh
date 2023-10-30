# nohup bash open_flamingo/scripts/reproduction/4BI_seeds/deploy.sh > ./logs/out-reproduce.text 2>&1 &

# shots = 0, 4, 8, 16; port = 26000, 26001, 26002, 26003; batch_size = 32

bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_vqav2.sh 0 26000 32 &> ./logs/out-reproduce-4BI_seeds-vqav2-shot-0.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_vqav2.sh 4 26001 16 &> ./logs/out-reproduce-4BI_seeds-vqav2-shot-4.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_vqav2.sh 8 26002 8 &> ./logs/out-reproduce-4BI_seeds-vqav2-shot-8.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_vqav2.sh 16 26003 4 &> ./logs/out-reproduce-4BI_seeds-vqav2-shot-16.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_vqav2.sh 32 26004 4 &> ./logs/out-reproduce-4BI_seeds-vqav2-shot-32.text

bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_okvqa.sh 0 26000 32 &> ./logs/out-reproduce-4BI_seeds-okvqa-shot-0.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_okvqa.sh 4 26001 16 &> ./logs/out-reproduce-4BI_seeds-okvqa-shot-4.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_okvqa.sh 8 26002 8 &> ./logs/out-reproduce-4BI_seeds-okvqa-shot-8.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_okvqa.sh 16 26003 4 &> ./logs/out-reproduce-4BI_seeds-okvqa-shot-16.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_okvqa.sh 32 26003 2 &> ./logs/out-reproduce-4BI_seeds-okvqa-shot-32.text

bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_gqa.sh 0 26001 16 &> ./logs/out-reproduce-4BI_seeds-gqa-shot-0.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_gqa.sh 4 26001 16 &> ./logs/out-reproduce-4BI_seeds-gqa-shot-4.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_gqa.sh 8 26002 8 &> ./logs/out-reproduce-4BI_seeds-gqa-shot-8.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_gqa.sh 16 26003 4 &> ./logs/out-reproduce-4BI_seeds-gqa-shot-16.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_gqa.sh 32 26003 2 &> ./logs/out-reproduce-4BI_seeds-gqa-shot-32.text

bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_coco.sh 0 26000 32 &> ./logs/out-reproduce-4BI_seeds-coco-shot-0.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_coco.sh 4 26001 16 &> ./logs/out-reproduce-4BI_seeds-coco-shot-4.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_coco.sh 8 26002 8 &> ./logs/out-reproduce-4BI_seeds-coco-shot-8.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_coco.sh 16 26003 6 &> ./logs/out-reproduce-4BI_seeds-coco-shot-16.text
bash open_flamingo/scripts/reproduction/4BI_seeds/run_eval_4BI_seeds_coco.sh 32 26004 4 &> ./logs/out-reproduce-4BI_seeds-coco-shot-32.text

