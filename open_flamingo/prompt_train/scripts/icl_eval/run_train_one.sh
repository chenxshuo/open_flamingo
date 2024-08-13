#!/bin/bash

NUMBER_OF_CLASSES=$1
ROBUST_PROMPTING=$2

MODEL_TYPE="OF-3B"
NUMBER_OF_MEDIA_PROMPTS=4
NUMBER_OF_TEXT_PROMPTS_PER_MEDIA=3
EPOCHS=50
BS=16

NUMBER_OF_ICL_DEMOS=4


if [ "$ROBUST_PROMPTING" = "True" ]; then
    python main.py \
      --model_type $MODEL_TYPE \
      --number_of_classes $NUMBER_OF_CLASSES \
      --number_of_media_prompts $NUMBER_OF_MEDIA_PROMPTS \
      --number_of_text_prompts_per_media $NUMBER_OF_TEXT_PROMPTS_PER_MEDIA \
      --epochs $EPOCHS \
      --lr 1e-1 \
      --bs $BS \
      --use_robust_prompting \
      --robust_scales 224 112 150 \
      --evaluate_dataset "imagenet-1k" \
      --evaluation_mode "classification" \
      --do_icl \
      --num_shots 4 \
      --do_rices \
      --icl_insertion_position demo-prompting-query

elif [ "$ROBUST_PROMPTING" = "False" ]; then
  python main.py \
      --model_type $MODEL_TYPE \
      --number_of_classes $NUMBER_OF_CLASSES \
      --number_of_media_prompts $NUMBER_OF_MEDIA_PROMPTS \
      --number_of_text_prompts_per_media $NUMBER_OF_TEXT_PROMPTS_PER_MEDIA \
      --epochs $EPOCHS \
      --lr 1e-1 \
      --bs $BS \
      --evaluate_dataset "imagenet-1k" \
      --evaluation_mode "classification" \
      --do_icl \
      --num_shots 4 \
      --do_rices \
      --icl_insertion_position demo-prompting-query

else
  echo "Invalid value for ROBUST_PROMPTING: $ROBUST_PROMPTING"
  exit 1
fi
