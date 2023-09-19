# nohup bash open_flamingo/scripts/no_images/deploy_1.sh > out_no_images_deploy_1.sh 2>&1 &
#
bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  4 26001 64 blank_images &> ./out-no-images-9B-vqav2-blank_images-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  8 26002 32 blank_images &> ./out-no-images-9B-vqav2-blank_images-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  16 26003 16 blank_images &> ./out-no-images-9B-vqav2-blank_images-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  32 26004 8 blank_images &> ./out-no-images-9B-vqav2-blank_images-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  4 26001 64 ood_images &> ./out-no-images-9B-vqav2-ood_images-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  8 26002 32 ood_images &> ./out-no-images-9B-vqav2-ood_images-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  16 26003 16 ood_images &> ./out-no-images-9B-vqav2-ood_images-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  32 26004 8 ood_images &> ./out-no-images-9B-vqav2-ood_images-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  4 26001 64 blank_images &> ./out-no-images-9B-okvqa-blank_images-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  8 26002 32 blank_images &> ./out-no-images-9B-okvqa-blank_images-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  16 26003 16 blank_images &> ./out-no-images-9B-okvqa-blank_images-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  32 26004 8 blank_images &> ./out-no-images-9B-okvqa-blank_images-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  4 26001 64 ood_images &> ./out-no-images-9B-okvqa-ood_images-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  8 26002 32 ood_images &> ./out-no-images-9B-okvqa-ood_images-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  16 26003 16 ood_images &> ./out-no-images-9B-okvqa-ood_images-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  32 26004 8 ood_images &> ./out-no-images-9B-okvqa-ood_images-shot-32.text


bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  4 26001 64 blank_images &> ./out-no-images-9B-textvqa-blank_images-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  8 26002 32 blank_images &> ./out-no-images-9B-textvqa-blank_images-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  16 26003 16 blank_images &> ./out-no-images-9B-textvqa-blank_images-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  32 26004 8 blank_images &> ./out-no-images-9B-textvqa-blank_images-shot-32.text
bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  4 26001 64 ood_images &> ./out-no-images-9B-textvqa-ood_images-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  8 26002 32 ood_images &> ./out-no-images-9B-textvqa-ood_images-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  16 26003 16 ood_images &> ./out-no-images-9B-textvqa-ood_images-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  32 26004 8 ood_images &> ./out-no-images-9B-textvqa-ood_images-shot-32.text


bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  4 26001 64 blank_images &> ./out-no-images-9B-vizwiz-blank_images-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  8 26002 32 blank_images &> ./out-no-images-9B-vizwiz-blank_images-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  16 26003 16 blank_images &> ./out-no-images-9B-vizwiz-blank_images-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  32 26004 8 blank_images &> ./out-no-images-9B-vizwiz-blank_images-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  4 26001 64 ood_images &> ./out-no-images-9B-vizwiz-ood_images-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  8 26002 32 ood_images &> ./out-no-images-9B-vizwiz-ood_images-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  16 26003 16 ood_images &> ./out-no-images-9B-vizwiz-ood_images-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  32 26004 8 ood_images &> ./out-no-images-9B-vizwiz-ood_images-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  4 26001 64 blank_images &> ./out-no-images-9B-gqa-blank_images-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  8 26002 32 blank_images &> ./out-no-images-9B-gqa-blank_images-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  16 26003 16 blank_images &> ./out-no-images-9B-gqa-blank_images-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  32 26004 8 blank_images &> ./out-no-images-9B-gqa-blank_images-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  4 26001 64 ood_images &> ./out-no-images-9B-gqa-ood_images-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  8 26002 32 ood_images &> ./out-no-images-9B-gqa-ood_images-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  16 26003 16 ood_images &> ./out-no-images-9B-gqa-ood_images-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  32 26004 8 ood_images &> ./out-no-images-9B-gqa-ood_images-shot-32.text

