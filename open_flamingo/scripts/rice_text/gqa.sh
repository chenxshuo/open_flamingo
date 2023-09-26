
#nohup bash open_flamingo/scripts/rice_text/gqa.sh > ./logs/rice_text/gqa.log 2>&1 &

#bash open_flamingo/scripts/rice_img/run_eval_9B_gqa.sh 4 26000 32 &> ./logs/rice_img/out-rice-img-9B-gqa-shot-4.text
#bash open_flamingo/scripts/rice_img/run_eval_9B_gqa.sh 8 26001 16 &> ./logs/rice_img/out-rice-img-9B-gqa-shot-8.text
#bash open_flamingo/scripts/rice_img/run_eval_9B_gqa.sh 16 26002 8 &> ./logs/rice_img/out-rice-img-9B-gqa-shot-16.text
#bash open_flamingo/scripts/rice_img/run_eval_9B_gqa.sh 32 26003 8 &> ./logs/rice_img/out-rice-img-9B-gqa-shot-32.text


#bash open_flamingo/scripts/rice_only_text/run_eval_9B_gqa.sh 8 26001 16 &> ./logs/rice_only_text/out-rice-img-9B-gqa-shot-8.text
bash open_flamingo/scripts/rice_only_text/run_eval_9B_gqa.sh 4 26003 32 &> ./logs/rice_only_text/out-rice-img-9B-gqa-shot-4.text
bash open_flamingo/scripts/rice_only_text/run_eval_9B_gqa.sh 16 26000 8 &> ./logs/rice_only_text/out-rice-img-9B-gqa-shot-16.text
bash open_flamingo/scripts/rice_only_text/run_eval_9B_gqa.sh 32 26002 8 &> ./logs/rice_only_text/out-rice-img-9B-gqa-shot-32.text


bash open_flamingo/scripts/rice_text/run_eval_9B_gqa.sh 4 26000 32 &> ./logs/rice_text/out-rice-img-9B-gqa-shot-4.text
bash open_flamingo/scripts/rice_text/run_eval_9B_gqa.sh 8 26001 16 &> ./logs/rice_text/out-rice-img-9B-gqa-shot-8.text
bash open_flamingo/scripts/rice_text/run_eval_9B_gqa.sh 16 26002 8 &> ./logs/rice_text/out-rice-img-9B-gqa-shot-16.text
bash open_flamingo/scripts/rice_text/run_eval_9B_gqa.sh 32 26003 8 &> ./logs/rice_text/out-rice-img-9B-gqa-shot-32.text
