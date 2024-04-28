#

#for EVALUATE_DATASET in "imagenet-a" #"imagenet-r" "imagenet-v2" "imagenet-s" "imagenet-c"
#for EVALUATE_DATASET in  "imagenet-r" # "imagenet-v2" "imagenet-s" "imagenet-c"
#for EVALUATE_DATASET in  "imagenet-v2" "imagenet-s" "imagenet-c"
#for EVALUATE_DATASET in "imagenet-1k" "imagenet-a" "imagenet-r" "imagenet-v2" "imagenet-s" "imagenet-c"
for EVALUATE_DATASET in "imagenet-r" "imagenet-v2" "imagenet-c"
do
  echo "EVALUATE_DATASET: $EVALUATE_DATASET"
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 8  False "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_False/media_prompts_5/text_prompts_per_media_3/2024-04-12_11-38-09" > logs/3B_robust_false_${EVALUATE_DATASET}_8.log 2>&1
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 16 False "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_False/media_prompts_5/text_prompts_per_media_3/2024-04-12_11-41-11" > logs/3B_robust_false_${EVALUATE_DATASET}_16.log 2>&1
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 32 False "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_False/media_prompts_5/text_prompts_per_media_3/2024-04-12_11-46-26" > logs/3B_robust_false_${EVALUATE_DATASET}_32.log 2>&1

#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 8  True False \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-04-18_16-20-30/epoch_18_accuracy_0.9025" \
#    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_false_8.log 2>&1

  bash scripts/run_eval_one.sh $EVALUATE_DATASET 8  True False \
   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-04-21_23-06-43/epoch_13_accuracy_0.945" \
    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_False_8.log 2>&1

  bash scripts/run_eval_one.sh $EVALUATE_DATASET 8  True True \
   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-04-21_23-06-43/epoch_13_accuracy_0.945" \
    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_True_8.log 2>&1

#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 16  True False \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-04-21_17-35-35/epoch_20_accuracy_0.9725" \
#    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_false_16.log 2>&1
#
#
#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 32  True False \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-04-21_17-35-35/epoch_20_accuracy_0.9725" \
#    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_false_32.log 2>&1

#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 16  True True \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-04-18_16-20-30/epoch_21_accuracy_0.9925" \
#    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_true_16.log 2>&1

#  bash scripts/run_eval_one.sh $EVALUATE_DATASET 32  True True \
#   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-04-18_16-20-30/epoch_21_accuracy_0.9925" \
#    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_true_32.log 2>&1
done

#
#python main.py \
#      --model_type "OF-3B" \
#      --number_of_classes 16 \
#      --number_of_media_prompts 5 \
#      --number_of_text_prompts_per_media 3 \
#      --bs 32 \
#      --evaluate_dataset "imagenet-a" \
#      --evaluation_mode "classification" \
#      --only_load_and_eval \
#      --load_from_dir "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_False/media_prompts_5/text_prompts_per_media_3/2024-04-12_11-41-11"
#
#
#python main.py \
#      --model_type "OF-3B" \
#      --number_of_classes 32 \
#      --number_of_media_prompts 5 \
#      --number_of_text_prompts_per_media 3 \
#      --bs 32 \
#      --evaluate_dataset "imagenet-a" \
#      --evaluation_mode "classification" \
#      --only_load_and_eval \
#      --load_from_dir "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_False/media_prompts_5/text_prompts_per_media_3/2024-04-12_11-46-26"
#
#
#python main.py \
#      --model_type "OF-3B" \
#      --number_of_classes 8 \
#      --number_of_media_prompts 5 \
#      --number_of_text_prompts_per_media 3 \
#      --bs 32 \
#      --evaluate_dataset "imagenet-a" \
#      --evaluation_mode "classification" \
#      --use_robust_prompting \
#      --only_load_and_eval \
#      --load_from_dir "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_5/text_prompts_per_media_3/2024-04-12_11-58-10"
#
#python main.py \
#      --model_type "OF-3B" \
#      --number_of_classes 16 \
#      --number_of_media_prompts 5 \
#      --number_of_text_prompts_per_media 3 \
#      --bs 32 \
#      --evaluate_dataset "imagenet-a" \
#      --evaluation_mode "classification" \
#      --use_robust_prompting \
#      --only_load_and_eval \
#      --load_from_dir "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_True/media_prompts_5/text_prompts_per_media_3/2024-04-12_12-02-35"
#
#python main.py \
#      --model_type "OF-3B" \
#      --number_of_classes 32 \
#      --number_of_media_prompts 5 \
#      --number_of_text_prompts_per_media 3 \
#      --bs 32 \
#      --evaluate_dataset "imagenet-a" \
#      --evaluation_mode "classification" \
#      --use_robust_prompting \
#      --only_load_and_eval \
#      --load_from_dir "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_True/media_prompts_5/text_prompts_per_media_3/2024-04-12_12-10-57"
