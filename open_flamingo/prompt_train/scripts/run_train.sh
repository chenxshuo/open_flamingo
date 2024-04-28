#!/bin/bash

#  bash scripts/run.sh
# training OF-3B on imagenet-1k

# not use robust prompting
for NUMBER_OF_CLASSES in 8 16 32; do
    echo "not use robust prompting"
    echo "NUMBER_OF_CLASSES: $NUMBER_OF_CLASSES"
    python main.py \
        --model_type "OF-3B" \
        --number_of_classes $NUMBER_OF_CLASSES \
        --number_of_media_prompts 5 \
        --number_of_text_prompts_per_media 3 \
        --epochs 200 \
        --lr 1e-1 \
        --bs 32 \
        --evaluate_dataset "imagenet-1k" \
        --evaluation_mode "classification" \

done


# use robust prompting
for NUMBER_OF_CLASSES in 8 16 32; do
  echo "use robust prompting"
  echo "NUMBER_OF_CLASSES: $NUMBER_OF_CLASSES"
  python main.py \
      --model_type "OF-3B" \
      --number_of_classes $NUMBER_OF_CLASSES \
      --number_of_media_prompts 5 \
      --number_of_text_prompts_per_media 3 \
      --epochs 200 \
      --lr 1e-1 \
      --bs 32 \
      --use_robust_prompting \
      --robust_scales 224 299 384 \
      --evaluate_dataset "imagenet-1k" \
      --evaluation_mode "classification" \

done