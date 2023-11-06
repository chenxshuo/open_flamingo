# nohup bash open_flamingo/scripts/reproduction/9B_seeds/deploy.sh > ./logs/out-reproduce.text 2>&1 &

# shots = 0, 4, 8, 16; port = 26000, 26001, 26002, 26003; batch_size = 64

#bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_okvqa.sh 0 26000 32 &> ./logs/out-reproduce-9B_seeds-okvqa-shot-0.text
#bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_okvqa.sh 4 26001 32 &> ./logs/out-reproduce-9B_seeds-okvqa-shot-4.text
#bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_okvqa.sh 8 26002 16 &> ./logs/out-reproduce-9B_seeds-okvqa-shot-8.text
#bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_okvqa.sh 16 26003 8 &> ./logs/out-reproduce-9B_seeds-okvqa-shot-16.text
#bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_okvqa.sh 32 26003 8 &> ./logs/out-reproduce-9B_seeds-okvqa-shot-32.text

bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_gqa.sh 0 26001 32 &> ./logs/out-reproduce-9B_seeds-gqa-shot-0.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_gqa.sh 4 26001 32 &> ./logs/out-reproduce-9B_seeds-gqa-shot-4.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_gqa.sh 8 26002 16 &> ./logs/out-reproduce-9B_seeds-gqa-shot-8.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_gqa.sh 16 26003 8 &> ./logs/out-reproduce-9B_seeds-gqa-shot-16.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_gqa.sh 32 26003 8 &> ./logs/out-reproduce-9B_seeds-gqa-shot-32.text

bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_coco.sh 0 26000 32 &> ./logs/out-reproduce-9B_seeds-coco-shot-0.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_coco.sh 4 26001 32 &> ./logs/out-reproduce-9B_seeds-coco-shot-4.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_coco.sh 8 26002 16 &> ./logs/out-reproduce-9B_seeds-coco-shot-8.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_coco.sh 16 26003 8 &> ./logs/out-reproduce-9B_seeds-coco-shot-16.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_coco.sh 32 26004 8 &> ./logs/out-reproduce-9B_seeds-coco-shot-32.text

bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_vqav2.sh 0 26000 32 &> ./logs/out-reproduce-9B_seeds-vqav2-shot-0.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_vqav2.sh 4 26001 32 &> ./logs/out-reproduce-9B_seeds-vqav2-shot-4.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_vqav2.sh 8 26002 16 &> ./logs/out-reproduce-9B_seeds-vqav2-shot-8.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_vqav2.sh 16 26003 8 &> ./logs/out-reproduce-9B_seeds-vqav2-shot-16.text
bash open_flamingo/scripts/reproduction/9B_seeds/run_eval_9B_seeds_vqav2.sh 32 26004 8 &> ./logs/out-reproduce-9B_seeds-vqav2-shot-32.text