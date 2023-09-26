
# nohup bash open_flamingo/scripts/store_demos_predictions/vqav2/deploy.sh > logs/deploy.log 2>&1 &

bash open_flamingo/scripts/store_demos_predictions/vqav2/run_eval_9B_vqav2_reproduction.sh 16 26001 8 &> logs/9B_vqav2_reproduction_shot_16.log
bash open_flamingo/scripts/store_demos_predictions/vqav2/run_eval_9B_vqav2_rice_image.sh 16 26002 8 &> logs/9B_vqav2_rice_image_shot_16.log
bash open_flamingo/scripts/store_demos_predictions/vqav2/run_eval_9B_vqav2_rice_only_text.sh 16 26003 8 &> logs/9B_vqav2_rice_only_text_shot_16.log
bash open_flamingo/scripts/store_demos_predictions/vqav2/run_eval_9B_vqav2_rice_text.sh 16 26004 8 &> logs/9B_vqav2_rice_text_16.log


