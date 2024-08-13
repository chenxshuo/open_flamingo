#!/bin/bash

#3B
LM_MODEL="anas-awadalla/mpt-1b-redpajama-200b"
CROSS_ATTN_EVERY_N_LAYERS=1

NUMBER_CLASS=$1

USER_NAME=`whoami`
if [ $USER_NAME == "di93zun" ]; then
    export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
    CKPT_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt"
    IMAGENET_TRAIN_ROOT="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/subset-32-classes/train"
    IMAGENET_TRAIN_ANNO="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet/imagenet_annotation_train_${NUMBER_CLASS}_classes_5_per_class.json"
    IMAGENET_VAL_ROOT="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-S/sketch"
    IMAGENET_VAL_ANNO="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet-S/imagenet_s_annotation_val_${NUMBER_CLASS}_classes.json"

elif [ $USER_NAME == "b207dd13" ]; then
    export HF_HOME="/home/atuin/b207dd/b207dd13/.cache/huggingface"
    CKPT_PATH="/home/atuin/b207dd/b207dd13/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt"
    IMAGENET_TRAIN_ROOT="/home/atuin/b207dd/b207dd13/in-context/dataset/imagenet/subset-32-classes/train"
    IMAGENET_TRAIN_ANNO="/home/atuin/b207dd/b207dd13/in-context/dataset/imagenet/imagenet_annotation_train_${NUMBER_CLASS}_classes_5_per_class.json"

    IMAGENET_VAL_ROOT="/home/atuin/b207dd/b207dd13/in-context/dataset/imagenet-S/sketch"
    IMAGENET_VAL_ANNO="/home/atuin/b207dd/b207dd13/in-context/dataset/imagenet-S/imagenet_s_annotation_val_${NUMBER_CLASS}_classes.json"
else
    echo "Unknown user: $USER_NAME"
    exit 1
fi


#SHOTS=$1
#MASTER_PORT=$2
#BS=$3

SHOTS=$3
MASTER_PORT=$2
BS=16

export CUDA_VISIBLE_DEVICES=0
NUM_GPUs=`echo $CUDA_VISIBLE_DEVICES | grep -P -o '\d' | wc -l`
TIMESTAMP=`date +"%Y-%m-%d-%T"`
COMMENT="4B-reproduce-gqa-shots-${SHOTS}"
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
    --visual_demo_mode "random" \
    --eval_imagenet \
    --imagenet_train_root $IMAGENET_TRAIN_ROOT\
    --imagenet_val_root $IMAGENET_VAL_ROOT \
    --imagenet_train_annotation  $IMAGENET_TRAIN_ANNO\
    --imagenet_val_annotation $IMAGENET_VAL_ANNO