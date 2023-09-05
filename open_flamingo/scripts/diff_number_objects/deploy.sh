# nohup bash ./open_flamingo/scripts/diff_number_objects/deploy.sh > ./logs/3B-vqav2-number_objects-deploy.log 2>&1 &
bash ./open_flamingo/scripts/diff_number_objects/run_eval_vqa_diff_number_objects.sh 0 26000 &> ./logs/3B-vqav2-number_objects-0.log
bash ./open_flamingo/scripts/diff_number_objects/run_eval_vqa_diff_number_objects.sh 2 26001 &> ./logs/3B-vqav2-number_objects-2.log
bash ./open_flamingo/scripts/diff_number_objects/run_eval_vqa_diff_number_objects.sh 4 26002 &> ./logs/3B-vqav2-number_objects-4.log
bash ./open_flamingo/scripts/diff_number_objects/run_eval_vqa_diff_number_objects.sh 6 26003 &> ./logs/3B-vqav2-number_objects-6.log
bash ./open_flamingo/scripts/diff_number_objects/run_eval_vqa_diff_number_objects.sh 8 26004 &> ./logs/3B-vqav2-number_objects-8.log
bash ./open_flamingo/scripts/diff_number_objects/run_eval_vqa_diff_number_objects.sh 16 26005 &> ./logs/3B-vqav2-number_objects-16.log
bash ./open_flamingo/scripts/diff_number_objects/run_eval_vqa_diff_number_objects.sh 32 26006 &> ./logs/3B-vqav2-number_objects-32.log
bash ./open_flamingo/scripts/diff_number_objects/run_eval_vqa_diff_number_objects.sh 64 26007 &> ./logs/3B-vqav2-number_objects-64.log


