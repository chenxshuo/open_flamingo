#

EVALUATE_DATASET=$1
NUMBER_OF_CLASSES=$2
ROBUST_PROMPTING=$3
EVAL_ON_NOVEL_CLASSES=$4
LOAD_FROM_DIR=$5


MODEL_TYPE="OF-3B"
NUMBER_OF_MEDIA_PROMPTS=8
NUMBER_OF_TEXT_PROMPTS_PER_MEDIA=3
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
      --load_from_dir $LOAD_FROM_DIR \
      --do_icl \
      --num_shots 4 \
      --do_rices \
      --icl_insertion_position demo-prompting-query
