#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --partition=mcml-hgx-a100-80x4
#SBTACH --time=1-00:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=c1094829085@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --qos=mcml

NUMBER_OF_CLASSES=$1
ROBUST_PROMPTING=$2

MODEL_TYPE="OF-9B"

NUMBER_OF_MEDIA_PROMPTS=$3

if [ -z "$NUMBER_OF_MEDIA_PROMPTS" ]; then
  NUMBER_OF_MEDIA_PROMPTS=8
fi

NUMBER_OF_TEXT_PROMPTS_PER_MEDIA=$4
if [ -z "$NUMBER_OF_TEXT_PROMPTS_PER_MEDIA" ]; then
  NUMBER_OF_TEXT_PROMPTS_PER_MEDIA=3
fi

EPOCHS=50
BS=16

echo "NUMBER_OF_CLASSES: $NUMBER_OF_CLASSES"
echo "ROBUST_PROMPTING: $ROBUST_PROMPTING"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "NUMBER_OF_MEDIA_PROMPTS: $NUMBER_OF_MEDIA_PROMPTS"
echo "NUMBER_OF_TEXT_PROMPTS_PER_MEDIA: $NUMBER_OF_TEXT_PROMPTS_PER_MEDIA"

if [ "$2" = "True" ]; then
    python main.py \
      --model_type $MODEL_TYPE \
      --number_of_classes $1 \
      --number_of_media_prompts $NUMBER_OF_MEDIA_PROMPTS \
      --number_of_text_prompts_per_media $NUMBER_OF_TEXT_PROMPTS_PER_MEDIA \
      --epochs $EPOCHS \
      --lr 1e-1 \
      --bs $BS \
      --use_robust_prompting \
      --robust_scales 224 112 150 \
      --evaluate_dataset "imagenet-1k" \
      --evaluation_mode "classification"
elif [ "$2" = "False" ]; then
  python main.py \
      --model_type $MODEL_TYPE \
      --number_of_classes $1 \
      --number_of_media_prompts $NUMBER_OF_MEDIA_PROMPTS \
      --number_of_text_prompts_per_media $NUMBER_OF_TEXT_PROMPTS_PER_MEDIA \
      --epochs $EPOCHS \
      --lr 1e-1 \
      --bs $BS \
      --evaluate_dataset "imagenet-1k" \
      --evaluation_mode "classification"
else
  echo "Invalid value for ROBUST_PROMPTING: $2"
  exit 1
fi
#EOT
