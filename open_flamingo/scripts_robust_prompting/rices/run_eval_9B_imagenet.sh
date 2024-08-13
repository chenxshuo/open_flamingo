#!/bin/bash
#SBATCH -N 1
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH -mem=100G
#SBTACH -t 3:00:00
#SBATCH --job-name=eval-9B-imagenet
#SBTACH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=shuo.chen@campus.lmu.de
#SBATCH --mail-type=ALL

export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"

# 9B
LM_MODEL="anas-awadalla/mpt-7b"
CROSS_ATTN_EVERY_N_LAYERS=4


USER_NAME=`whoami`

DATASET_NAME=$1
NUMBER_CLASS=$2
CUDA=$3
MASTER_PORT=$4

#DATASET_NAME="imagenet-1k"
#NUMBER_CLASS=8
#CUDA=0
#MASTER_PORT=26000

SHOTS=$5
if [ -z "$SHOTS" ] ;
then
  SHOTS=8
fi

VISUAL_DEMO_MODE=$6
if [ -z "$VISUAL_DEMO_MODE" ] ;
then
  VISUAL_DEMO_MODE="random"
fi

echo "Shot number: $SHOTS"
echo "Visual demo mode: $VISUAL_DEMO_MODE"
BS=16



if [ $USER_NAME == "di93zun" ]; then
    export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
    CKPT_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-9B-vitl-mpt7b/snapshots/e6e175603712c7007fe3b9c0d50bdcfbd83adfc2/checkpoint.pt"
    IMAGENET_TRAIN_ROOT="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train"
    IMAGENET_TRAIN_ANNO="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet_annotation_train_${NUMBER_CLASS}_classes_5_per_class.json"

    IMAGENET_VAL_ROOT="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets"
    IMAGENET_VAL_ANNO="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets"
    if [[ $DATASET_NAME == "imagenet-1k" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet/subset-32-classes/val"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet/imagenet_annotation_val_${NUMBER_CLASS}_classes.json"

    elif [[ $DATASET_NAME == "imagenet-1k-novel" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet/novel-8-classes/val"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet/novel-8-classes/imagenet1k_novel_classes_val.json"

    elif [[ $DATASET_NAME == "imagenet-a" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet-A/imagenet-a"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet-A/imagenet_a_annotation_val_${NUMBER_CLASS}_classes.json"

    elif [[ $DATASET_NAME == "imagenet-a-novel" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet-A/imagenet-a"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet-A/imagenet_a_novel_classes_val.json"

    elif [[ $DATASET_NAME == "imagenet-c" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet-C/imagenet-C-severity-5"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet-C/imagenet_c_annotation_val_${NUMBER_CLASS}_classes.json"

    elif [[ $DATASET_NAME == "imagenet-c-novel" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet-C/novel-8-classes-imagenet-C-severity-5"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet-C/imagenet_c_novel_classes_val.json"

    elif [[ $DATASET_NAME == "imagenet-r" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet-R/imagenet-r"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet-R/imagenet_r_annotation_val_${NUMBER_CLASS}_classes.json"

    elif [[ $DATASET_NAME == "imagenet-r-novel" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet-R/imagenet-r"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet-R/imagenet_r_novel_classes_val.json"


    elif [[ $DATASET_NAME == "imagenet-s" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet-S/sketch"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet-S/imagenet_s_annotation_val_${NUMBER_CLASS}_classes.json"

    elif [[ $DATASET_NAME == "imagenet-s-novel" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet-S/sketch"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet-S/imagenet_s_novel_classes_val.json"


    elif [[ $DATASET_NAME == "imagenet-v2" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet-V2/imagenetv2-top-images-format-val"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet-V2/imagenet_v2_annotation_val_${NUMBER_CLASS}_classes.json"

    elif [[ $DATASET_NAME == "imagenet-v2-novel" ]]; then
      IMAGENET_VAL_ROOT="${IMAGENET_VAL_ROOT}/imagenet-V2/imagenetv2-top-images-format-val"
      IMAGENET_VAL_ANNO="${IMAGENET_VAL_ANNO}/imagenet-V2/imagenet_v2_novel_classes_val.json"

    else
      echo "Unknown dataset: $DATASET_NAME"
      exit 1
    fi

else
    echo "Unknown user: $USER_NAME"
    exit 1
fi


#
#SHOTS=$1
#MASTER_PORT=$2
#BS=$3



export CUDA_VISIBLE_DEVICES=$CUDA
NUM_GPUs=`echo $CUDA_VISIBLE_DEVICES | grep -P -o '\d' | wc -l`
TIMESTAMP=`date +"%Y-%m-%d-%T"`
COMMENT="9B-rices-${DATASET_NAME}-number-class-${NUMBER_CLASS}-shots-${SHOTS}"
RESULTS_FILE="results_${TIMESTAMP}_${COMMENT}.json"
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUs" --master_port=${MASTER_PORT} open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path ${LM_MODEL} \
    --lm_tokenizer_path ${LM_MODEL} \
    --cross_attn_every_n_layers ${CROSS_ATTN_EVERY_N_LAYERS} \
    --checkpoint_path ${CKPT_PATH} \
    --results_file ${RESULTS_FILE} \
    --precision amp_bf16 \
    --batch_size ${BS} \
    --num_trials 1 \
    --shots ${SHOTS} \
    --trial_seeds 42 \
    --demo_mode  "gold" \
    --visual_demo_mode $VISUAL_DEMO_MODE \
    --rices \
    --eval_imagenet \
    --imagenet_train_root $IMAGENET_TRAIN_ROOT\
    --imagenet_val_root $IMAGENET_VAL_ROOT \
    --imagenet_train_annotation  $IMAGENET_TRAIN_ANNO\
    --imagenet_val_annotation $IMAGENET_VAL_ANNO