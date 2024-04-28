

NUM_CLASSES=$1
DATASET=$2
DO_FEW_SHOT=$3
if [ "$DO_FEW_SHOT" = "True" ]; then
  echo "Doing few shot"
  python supervised.py \
    --number_of_classes $NUM_CLASSES \
    --eval_dataset $DATASET \
    --do_few_shot
  exit 0
else
  python supervised.py \
  --number_of_classes $NUM_CLASSES \
  --train_bs 64 \
  --eval_dataset $DATASET
fi

