# nohup bash open_flamingo/scripts/no_images/3BI/deploy_1.sh > ./logs/out_no_images-3BI_deploy_1.sh 2>&1 &

bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  4 26001 32 blank_images &> ./logs/out-no-images-3BI-vqav2-blank_images-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  8 26002 32 blank_images &> ./logs/out-no-images-3BI-vqav2-blank_images-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  16 26003 16 blank_images &> ./logs/out-no-images-3BI-vqav2-blank_images-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  32 26004 8 blank_images &> ./logs/out-no-images-3BI-vqav2-blank_images-shot-32.text

bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  4 26001 32 ood_images &> ./logs/out-no-images-3BI-vqav2-ood_images-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  8 26002 32 ood_images &> ./logs/out-no-images-3BI-vqav2-ood_images-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  16 26003 16 ood_images &> ./logs/out-no-images-3BI-vqav2-ood_images-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  32 26004 16 ood_images &> ./logs/out-no-images-3BI-vqav2-ood_images-shot-32.text

bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  4 26001 32 no_images &> ./logs/out-no-images-3BI-vqav2-no_images-3BI-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  8 26002 32 no_images &> ./logs/out-no-images-3BI-vqav2-no_images-3BI-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  16 26003 16 no_images &> ./logs/out-no-images-3BI-vqav2-no_images-3BI-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_vqav2.sh  32 26004 16 no_images &> ./logs/out-no-images-3BI-vqav2-no_images-3BI-shot-32.text


bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  4 26001 32 blank_images &> ./logs/out-no-images-3BI-okvqa-blank_images-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  8 26002 32 blank_images &> ./logs/out-no-images-3BI-okvqa-blank_images-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  16 26003 16 blank_images &> ./logs/out-no-images-3BI-okvqa-blank_images-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  32 26004 16 blank_images &> ./logs/out-no-images-3BI-okvqa-blank_images-shot-32.text

bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  4 26001 32 ood_images &> ./logs/out-no-images-3BI-okvqa-ood_images-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  8 26002 32 ood_images &> ./logs/out-no-images-3BI-okvqa-ood_images-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  16 26003 16 ood_images &> ./logs/out-no-images-3BI-okvqa-ood_images-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  32 26004 16 ood_images &> ./logs/out-no-images-3BI-okvqa-ood_images-shot-32.text

bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  4 26001 32 no_images &> ./logs/out-no-images-3BI-okvqa-no_images-3BI-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  8 26002 32 no_images &> ./logs/out-no-images-3BI-okvqa-no_images-3BI-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  16 26003 16 no_images &> ./logs/out-no-images-3BI-okvqa-no_images-3BI-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_okvqa.sh  32 26004 8 no_images &> ./logs/out-no-images-3BI-okvqa-no_images-3BI-shot-32.text


bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  4 26001 32 blank_images &> ./logs/out-no-images-3BI-gqa-blank_images-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  8 26002 32 blank_images &> ./logs/out-no-images-3BI-gqa-blank_images-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  16 26003 16 blank_images &> ./logs/out-no-images-3BI-gqa-blank_images-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  32 26004 8 blank_images &> ./logs/out-no-images-3BI-gqa-blank_images-shot-32.text

bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  4 26001 32 no_images &> ./logs/out-no-images-3BI-gqa-no_images-3BI-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  8 26002 32 no_images &> ./logs/out-no-images-3BI-gqa-no_images-3BI-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  16 26003 16 no_images &> ./logs/out-no-images-3BI-gqa-no_images-3BI-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  32 26004 8 no_images &> ./logs/out-no-images-3BI-gqa-no_images-3BI-shot-32.text

bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  4 26001 32 ood_images &> ./logs/out-no-images-3BI-gqa-ood_images-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  8 26002 32 ood_images &> ./logs/out-no-images-3BI-gqa-ood_images-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  16 26003 16 ood_images &> ./logs/out-no-images-3BI-gqa-ood_images-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_gqa.sh  32 26004 8 ood_images &> ./logs/out-no-images-3BI-gqa-ood_images-shot-32.text

bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  4 26001 16 blank_images &> ./logs/out-no-images-3BI-coco-blank_images-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  8 26002 8 blank_images &> ./logs/out-no-images-3BI-coco-blank_images-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  16 26003 4 blank_images &> ./logs/out-no-images-3BI-coco-blank_images-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  32 26004 4 blank_images &> ./logs/out-no-images-3BI-coco-blank_images-shot-32.text

bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  4 26001 16 no_images &> ./logs/out-no-images-3BI-coco-no_images-3BI-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  8 26002 8 no_images &> ./logs/out-no-images-3BI-coco-no_images-3BI-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  16 26003 4 no_images &> ./logs/out-no-images-3BI-coco-no_images-3BI-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  32 26004 4 no_images &> ./logs/out-no-images-3BI-coco-no_images-3BI-shot-32.text

bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  4 26001 16 ood_images &> ./logs/out-no-images-3BI-coco-ood_images-shot-4.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  8 26002 8 ood_images &> ./logs/out-no-images-3BI-coco-ood_images-shot-8.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  16 26003 4 ood_images &> ./logs/out-no-images-3BI-coco-ood_images-shot-16.text
bash open_flamingo/scripts/no_images/3BI/run_eval_3BI_coco.sh  32 26004 4 ood_images &> ./logs/out-no-images-3BI-coco-ood_images-shot-32.text


