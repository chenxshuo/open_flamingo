

NUM_CLASSES=$1
DATASET=$2
DO_FEW_SHOT=$3
EVAL_NOVEL_CLASSES=$4


# set up dir to load
if [ "$NUM_CLASSES" = "8" ]; then
  if [ "$DO_FEW_SHOT" = "True" ]; then
    DIR="./experiments/evaluate_dataset_imagenet-1k/classes_8/2024-04-17_15-37-15"
  elif [ "$DO_FEW_SHOT" = "False" ]; then
    DIR="./experiments/evaluate_dataset_imagenet-1k/classes_8/2024-04-17_16-22-56"
  else
    echo "Invalid few shot argument"
    exit 1
  fi
elif [ "$NUM_CLASSES" = "16" ]; then
  if [ "$DO_FEW_SHOT" = "True" ]; then
    DIR="./experiments/evaluate_dataset_imagenet-1k/classes_16/2024-04-17_19-13-09"
  elif [ "$DO_FEW_SHOT" = "False" ]; then
    DIR="./experiments/evaluate_dataset_imagenet-1k/classes_16/2024-04-17_19-29-50"
  else
    echo "Invalid few shot argument"
    exit 1
  fi
elif [ "$NUM_CLASSES" = "32" ]; then
  if [ "$DO_FEW_SHOT" = "True" ]; then
    DIR="./experiments/evaluate_dataset_imagenet-1k/classes_32/2024-04-18_01-05-07"
  elif [ "$DO_FEW_SHOT" = "False" ]; then
    DIR="./experiments/evaluate_dataset_imagenet-1k/classes_32/2024-04-18_01-38-09"
  else
    echo "Invalid few shot argument"
    exit 1
  fi
else
  echo "Invalid number of classes"
  exit 1
fi

echo "Loading from $DIR"
echo "Evaluating on $DATASET with $NUM_CLASSES classes"
echo "Few shot: $DO_FEW_SHOT"
echo "Eval novel classes: $EVAL_NOVEL_CLASSES"

if [ "$EVAL_NOVEL_CLASSES" = "False" ]; then
  if [ "$DO_FEW_SHOT" = "True" ]; then
    python supervised.py \
      --number_of_classes $NUM_CLASSES \
      --eval_dataset $DATASET \
      --do_few_shot \
      --only_load_and_eval \
      --load_from_dir $DIR
  elif [ "$DO_FEW_SHOT" = "False" ]; then
    python supervised.py \
    --number_of_classes $NUM_CLASSES \
    --train_bs 64 \
    --eval_dataset $DATASET \
    --only_load_and_eval \
    --load_from_dir $DIR
  else
    echo "Invalid few shot argument"
    exit 1
  fi

elif [ "$EVAL_NOVEL_CLASSES" = "True" ]; then

  if [ "$NUM_CLASSES" != "8" ]; then
    echo "Invalid number of classes for evaluating novel classes"
    exit 1
  fi

  if [ "$DO_FEW_SHOT" = "True" ]; then
    python supervised.py \
      --number_of_classes $NUM_CLASSES \
      --eval_dataset $DATASET \
      --do_few_shot \
      --eval_novel_classes \
      --only_load_and_eval \
      --load_from_dir $DIR
  elif [ "$DO_FEW_SHOT" = "False" ]; then
    python supervised.py \
    --number_of_classes $NUM_CLASSES \
    --eval_dataset $DATASET \
    --eval_novel_classes \
    --only_load_and_eval \
    --load_from_dir $DIR
  else
    echo "Invalid few shot argument"
    exit 1
  fi
else
  echo "Invalid eval novel classes argument"
  exit 1

fi


