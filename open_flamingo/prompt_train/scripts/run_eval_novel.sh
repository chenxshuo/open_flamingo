

EVALUATE_DATASET="imagenet-r"
# TRAINED="./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_19_accuracy_0.9925" # eval acc: 69.25
# TRAINED="./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_4_accuracy_0.6625" # eval acc: 60.5
# TRAINED="./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_9_accuracy_0.87" # eval acc: 65.5
TRAINED="./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_14_accuracy_0.98" # eval acc: 71.25


# robust;
bash scripts/run_eval_one.sh $EVALUATE_DATASET 8 8 3 True True \
    $TRAINED # \
#    > logs/3B_robust_true_${EVALUATE_DATASET}_novel_true_8.log 2>&1
