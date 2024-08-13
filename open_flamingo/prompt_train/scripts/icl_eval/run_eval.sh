

# data, classes, robust, novel;
bash scripts/icl_eval/run_eval_one.sh "imagenet-1k" 8  False True \
   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_24_accuracy_0.9875"

bash scripts/icl_eval/run_eval_one.sh "imagenet-1k" 8  True True \
   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_19_accuracy_0.9925"



bash scripts/icl_eval/run_eval_one.sh "imagenet-1k" 8  False False \
   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_24_accuracy_0.9875"

bash scripts/icl_eval/run_eval_one.sh "imagenet-r" 8  False False \
   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_24_accuracy_0.9875"

bash scripts/icl_eval/run_eval_one.sh "imagenet-r" 32  False False \
   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-01_18-15-47/epoch_9_accuracy_0.958125"



bash scripts/icl_eval/run_eval_one.sh "imagenet-a" 8 True False \
   "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-04-21_23-06-43/epoch_13_accuracy_0.945"
