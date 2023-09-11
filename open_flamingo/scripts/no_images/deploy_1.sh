# nohup bash open_flamingo/scripts/no_images/deploy_1.sh > out_no_images_deploy_1.sh 2>&1 &

bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  4 26001 64 &> ./out-no-images-9B-vqav2-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  8 26002 32 &> ./out-no-images-9B-vqav2-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  16 26003 16 &> ./out-no-images-9B-vqav2-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_vqav2.sh  32 26004 8 &> ./out-no-images-9B-vqav2-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  4 26001 64 &> ./out-no-images-9B-okvqa-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  8 26002 32 &> ./out-no-images-9B-okvqa-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  16 26003 16 &> ./out-no-images-9B-okvqa-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_okvqa.sh  32 26004 8 &> ./out-no-images-9B-okvqa-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  4 26001 64 &> ./out-no-images-9B-textvqa-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  8 26002 32 &> ./out-no-images-9B-textvqa-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  16 26003 16 &> ./out-no-images-9B-textvqa-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_textvqa.sh  32 26004 8 &> ./out-no-images-9B-textvqa-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  4 26001 64 &> ./out-no-images-9B-vizwiz-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  8 26002 32 &> ./out-no-images-9B-vizwiz-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  16 26003 16 &> ./out-no-images-9B-vizwiz-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_vizwiz.sh  32 26004 8 &> ./out-no-images-9B-vizwiz-shot-32.text

bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  4 26001 64 &> ./out-no-images-9B-gqa-shot-4.text
bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  8 26002 32 &> ./out-no-images-9B-gqa-shot-8.text
bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  16 26003 16 &> ./out-no-images-9B-gqa-shot-16.text
bash open_flamingo/scripts/no_images/run_eval_9B_gqa.sh  32 26004 8 &> ./out-no-images-9B-gqa-shot-32.text

