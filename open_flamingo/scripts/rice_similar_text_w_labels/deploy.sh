# nohup bash open_flamingo/scripts/rice_similar_text_w_labels/deploy.sh > ./logs/rice_similar_text_w_labels/deploy.text  2>&1 &

#bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_okvqa.sh 4 27001 32 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-okvqa-shot-4.text
bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_okvqa.sh 8 27002 16 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-okvqa-shot-8.text
bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_okvqa.sh 16 27003 4 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-okvqa-shot-16.text
bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_okvqa.sh 32 27004 8 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-okvqa-shot-32.text

bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_textvqa.sh 4 26000 32 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-textvqa-shot-4.text
bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_textvqa.sh 8 26000 16 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-textvqa-shot-8.text
bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_textvqa.sh 16 26000 8 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-textvqa-shot-16.text
bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_textvqa.sh 32 26000 8 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-textvqa-shot-32.text

#bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_vizwiz.sh 4 26000 32 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-vizwiz-shot-4.text
#bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_vizwiz.sh 8 26000 16 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-vizwiz-shot-8.text
#bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_vizwiz.sh 16 26000 8 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-vizwiz-shot-16.text
#bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_vizwiz.sh 32 26000 8 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-vizwiz-shot-32.text
#
bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_vqav2.sh 4 26000 32 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-vqav2-shot-4.text
bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_vqav2.sh 8 26001 16 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-vqav2-shot-8.text
bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_vqav2.sh 16 26002 8 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-vqav2-shot-16.text
bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_vqav2.sh 32 26003 8 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-vqav2-shot-32.text
#
#bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_gqa.sh 4 26000 32 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-gqa-shot-4.text
#bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_gqa.sh 8 26000 16 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-gqa-shot-8.text
#bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_gqa.sh 16 26000 8 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-gqa-shot-16.text
#bash open_flamingo/scripts/rice_similar_text_w_labels/run_eval_9B_gqa.sh 32 26000 8 &> ./logs/rice_similar_text_w_labels/out-rice-img-9B-gqa-shot-32.text
