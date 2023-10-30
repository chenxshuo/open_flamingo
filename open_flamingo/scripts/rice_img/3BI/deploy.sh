# nohup bash open_flamingo/scripts/rice_img/3BI/deploy.sh > ./logs/rice_img/deploys.log 2>&1 &

bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_coco.sh 4 26000 16 &> ./logs/rice_img/out-rice-img-3BI-coco-shot-4.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_coco.sh 8 26000 8 &> ./logs/rice_img/out-rice-img-3BI-coco-shot-8.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_coco.sh 16 26000 4 &> ./logs/rice_img/out-rice-img-3BI-coco-shot-16.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_coco.sh 32 26000 2 &> ./logs/rice_img/out-rice-img-3BI-coco-shot-32.text

bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_okvqa.sh 4 26000 16 &> ./logs/rice_img/out-rice-img-3BI-okvqa-shot-4.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_okvqa.sh 8 26000 8 &> ./logs/rice_img/out-rice-img-3BI-okvqa-shot-8.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_okvqa.sh 16 26000 4 &> ./logs/rice_img/out-rice-img-3BI-okvqa-shot-16.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_okvqa.sh 32 26000 2 &> ./logs/rice_img/out-rice-img-3BI-okvqa-shot-32.text


bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_vqav2.sh 4 26000 16 &> ./logs/rice_img/out-rice-img-3BI-vqav2-shot-4.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_vqav2.sh 8 26000 8 &> ./logs/rice_img/out-rice-img-3BI-vqav2-shot-8.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_vqav2.sh 16 26000 4 &> ./logs/rice_img/out-rice-img-3BI-vqav2-shot-16.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_vqav2.sh 32 26000 2 &> ./logs/rice_img/out-rice-img-3BI-vqav2-shot-32.text

bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_gqa.sh 4 26000 16 &> ./logs/rice_img/out-rice-img-3BI-gqa-shot-4.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_gqa.sh 8 26000 8 &> ./logs/rice_img/out-rice-img-3BI-gqa-shot-8.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_gqa.sh 16 26000 4 &> ./logs/rice_img/out-rice-img-3BI-gqa-shot-16.text
bash open_flamingo/scripts/rice_img/3BI/run_eval_3BI_gqa.sh 32 26000 2 &> ./logs/rice_img/out-rice-img-3BI-gqa-shot-32.text
