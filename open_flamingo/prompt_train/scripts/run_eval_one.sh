#

EVALUATE_DATASET=$1
NUMBER_OF_CLASSES=$2




MODEL_TYPE="OF-9B"

NUMBER_OF_MEDIA_PROMPTS=8
#if [ -z "$NUMBER_OF_MEDIA_PROMPTS" ]; then
#  NUMBER_OF_MEDIA_PROMPTS=8
#fi
#
NUMBER_OF_TEXT_PROMPTS_PER_MEDIA=3
#if [ -z "$NUMBER_OF_TEXT_PROMPTS_PER_MEDIA" ]; then
#  NUMBER_OF_TEXT_PROMPTS_PER_MEDIA=3
#fi

ROBUST_PROMPTING=$3
EVAL_ON_NOVEL_CLASSES=$4
LOAD_FROM_DIR=$5

BS=32

if [ "$ROBUST_PROMPTING" = "True" ]; then
  ROBUST_PROMPTING="--use_robust_prompting"
elif [ "$ROBUST_PROMPTING" = "False" ]; then
  ROBUST_PROMPTING=""
else
  echo "Invalid value for ROBUST_PROMPTING: $ROBUST_PROMPTING"
  exit 1
fi

if [ "$EVAL_ON_NOVEL_CLASSES" = "True" ]; then
  EVAL_ON_NOVEL_CLASSES="--eval_novel_classes"
elif [ "$EVAL_ON_NOVEL_CLASSES" = "False" ]; then
  EVAL_ON_NOVEL_CLASSES=""
else
  echo "Invalid value for EVAL_ON_NOVEL_CLASSES: $EVAL_ON_NOVEL_CLASSES"
  exit 1
fi

echo "NUMBER_OF_CLASSES: $NUMBER_OF_CLASSES"
echo "ROBUST_PROMPTING: $ROBUST_PROMPTING"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "NUMBER_OF_MEDIA_PROMPTS: $NUMBER_OF_MEDIA_PROMPTS"
echo "NUMBER_OF_TEXT_PROMPTS_PER_MEDIA: $NUMBER_OF_TEXT_PROMPTS_PER_MEDIA"


python main.py \
      --model_type $MODEL_TYPE \
      --number_of_classes $NUMBER_OF_CLASSES \
      --number_of_media_prompts $NUMBER_OF_MEDIA_PROMPTS \
      --number_of_text_prompts_per_media $NUMBER_OF_TEXT_PROMPTS_PER_MEDIA \
      --bs $BS \
      --evaluate_dataset $EVALUATE_DATASET \
      --evaluation_mode "classification" \
      $ROBUST_PROMPTING \
      $EVAL_ON_NOVEL_CLASSES \
      --only_load_and_eval \
      --load_from_dir $LOAD_FROM_DIR
