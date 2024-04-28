#

python main.py \
      --model_type "OF-3B" \
      --number_of_classes 8 \
      --number_of_media_prompts 5 \
      --number_of_text_prompts_per_media 3 \
      --bs 32 \
      --evaluate_dataset "imagenet-a" \
      --evaluation_mode "classification" \
      --only_load_and_eval \
      --load_from_dir "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_False/media_prompts_5/text_prompts_per_media_3/2024-04-12_11-38-09"

python main.py \
      --model_type "OF-3B" \
      --number_of_classes 16 \
      --number_of_media_prompts 5 \
      --number_of_text_prompts_per_media 3 \
      --bs 32 \
      --evaluate_dataset "imagenet-a" \
      --evaluation_mode "classification" \
      --only_load_and_eval \
      --load_from_dir "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_False/media_prompts_5/text_prompts_per_media_3/2024-04-12_11-41-11"


python main.py \
      --model_type "OF-3B" \
      --number_of_classes 32 \
      --number_of_media_prompts 5 \
      --number_of_text_prompts_per_media 3 \
      --bs 32 \
      --evaluate_dataset "imagenet-a" \
      --evaluation_mode "classification" \
      --only_load_and_eval \
      --load_from_dir "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_False/media_prompts_5/text_prompts_per_media_3/2024-04-12_11-46-26"


python main.py \
      --model_type "OF-3B" \
      --number_of_classes 8 \
      --number_of_media_prompts 5 \
      --number_of_text_prompts_per_media 3 \
      --bs 32 \
      --evaluate_dataset "imagenet-a" \
      --evaluation_mode "classification" \
      --use_robust_prompting \
      --only_load_and_eval \
      --load_from_dir "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_5/text_prompts_per_media_3/2024-04-12_11-58-10"

python main.py \
      --model_type "OF-3B" \
      --number_of_classes 16 \
      --number_of_media_prompts 5 \
      --number_of_text_prompts_per_media 3 \
      --bs 32 \
      --evaluate_dataset "imagenet-a" \
      --evaluation_mode "classification" \
      --use_robust_prompting \
      --only_load_and_eval \
      --load_from_dir "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_True/media_prompts_5/text_prompts_per_media_3/2024-04-12_12-02-35"

python main.py \
      --model_type "OF-3B" \
      --number_of_classes 32 \
      --number_of_media_prompts 5 \
      --number_of_text_prompts_per_media 3 \
      --bs 32 \
      --evaluate_dataset "imagenet-a" \
      --evaluation_mode "classification" \
      --use_robust_prompting \
      --only_load_and_eval \
      --load_from_dir "./experiments/model_OF-3B/evaluate_dataset_imagenet-1k/classes_32/use_robust_prompting_True/media_prompts_5/text_prompts_per_media_3/2024-04-12_12-10-57"
