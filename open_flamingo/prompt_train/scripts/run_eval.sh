##
#
#bash scripts/run_eval_one.sh imagenet-1k 8  False False \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_24_accuracy_0.9875"
#
##bash scripts/run_eval_one.sh imagenet-1k 16  False False \
##   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-01_20-40-17/epoch_14_accuracy_0.96875"
#
#
##for EVALUATE_DATASET in "imagenet-a" #"imagenet-r" "imagenet-v2" "imagenet-s" "imagenet-c"
##for EVALUATE_DATASET in  "imagenet-r" # "imagenet-v2" "imagenet-s" "imagenet-c"
##for EVALUATE_DATASET in  "imagenet-v2" "imagenet-s" "imagenet-c"
##for EVALUATE_DATASET in "imagenet-1k" "imagenet-a" "imagenet-r" "imagenet-v2" "imagenet-s" "imagenet-c"
#for EVALUATE_DATASET in "imagenet-a" "imagenet-r" "imagenet-v2" "imagenet-s" "imagenet-c"
##for EVALUATE_DATASET in "imagenet-r" "imagenet-v2" "imagenet-c"
#do
#  echo "EVALUATE_DATASET: $EVALUATE_DATASET"
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 8  False False \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_24_accuracy_0.9875" \
#    > logs/3B_robust_false_${EVALUATE_DATASET}_novel_false_8.log 2>&1
#
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 8  True False \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_19_accuracy_0.9925" \
#    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_false_8.log 2>&1
#
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 8  False True \
#  "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_24_accuracy_0.9875" \
#    > logs/3B_robust_false_${EVALUATE_DATASET}_novel_true_8.log 2>&1
#
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 8  True True \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_19_accuracy_0.9925" \
#    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_true_8.log 2>&1
#
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 16  False False \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-01_20-40-17/epoch_14_accuracy_0.96875" \
#    > logs/3B_robust_false_${EVALUATE_DATASET}_novel_false_16.log 2>&1
#
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 16  True False \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-01_20-40-17/epoch_19_accuracy_0.97" \
#    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_false_16.log 2>&1
#
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 32  False False \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_9_accuracy_0.958125" \
#    > logs/3B_robust_false_${EVALUATE_DATASET}_novel_false_32.log 2>&1
#
#
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 32  True False \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_19_accuracy_0.964375" \
#    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_false_32.log 2>&1
##  pids+=($!)
#
##  echo "active processes: ${#pids[@]} for $EVALUATE_DATASET"
##  for ((i=${#pids[@]}; i>1; i--)) ; do
##      wait -n
##  done
#  echo "all processes finished for $EVALUATE_DATASET"
#done

# nohup bash scripts/run_eval.sh > logs/9B_sit_3.log 2>&1 &
#
#{
#export CUDA_VISIBLE_DEVICES=0
#for EVALUATE_DATASET in "imagenet-a" "imagenet-r" "imagenet-v2" "imagenet-s" "imagenet-c"
#do
#  echo "EVALUATE_DATASET: $EVALUATE_DATASET"
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 16  4 3 False False \
#   "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_False/media_prompts_4/text_prompts_per_media_3/2024-06-10_15-50-54/epoch_27_accuracy_0.98625" \
#    > logs/9B_robust_shorter_sit_2_false_${EVALUATE_DATASET}_novel_false_16.log 2>&1
#
##  bash scripts/run_eval_one.sh $EVALUATE_DATASET 16  4 3 False True \
##   "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_False/media_prompts_4/text_prompts_per_media_3/2024-06-10_15-50-54/epoch_27_accuracy_0.98625" \
##    > logs/9B_robust_shorter_sit_2_false_${EVALUATE_DATASET}_novel_true_16.log 2>&1
#done
#} &
#
#{
#export CUDA_VISIBLE_DEVICES=1
#for EVALUATE_DATASET in "imagenet-a" "imagenet-r" "imagenet-v2" "imagenet-s" "imagenet-c"
#do
#  echo "EVALUATE_DATASET: $EVALUATE_DATASET"
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 32 4 3 False False \
#   "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_False/media_prompts_4/text_prompts_per_media_3/2024-06-10_15-50-54/epoch_11_accuracy_0.96375" \
#    > logs/9B_robust_shorter_sit_2_false_${EVALUATE_DATASET}_novel_false_32.log 2>&1
#
##  bash scripts/run_eval_one.sh $EVALUATE_DATASET 32 4 3 False True \
##   "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_False/media_prompts_4/text_prompts_per_media_3/2024-06-10_15-50-54/epoch_11_accuracy_0.96375" \
##    > logs/9B_robust_shorter_sit_2_true_${EVALUATE_DATASET}_novel_true_8.log 2>&1
#
#
#done
#} &
#
#wait
##

CUDA_VISIBLE_DEVICES=0
EVALUATE_DATASET="imagenet-1k"
bash scripts/run_eval_one.sh $EVALUATE_DATASET 8  True True \
 "./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-06-07_05-35-27/epoch_19_accuracy_0.99"
#  > logs/9B_robust_sit_2_${EVALUATE_DATASET}_novel_true_8.log 2>&1
