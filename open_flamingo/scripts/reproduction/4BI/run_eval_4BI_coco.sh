#!/bin/bash
export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
BASE_COCO_DATA_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO"
COCO_IMG_TRAIN_PATH="${BASE_COCO_DATA_PATH}/train2014"
COCO_IMG_VAL_PATH="${BASE_COCO_DATA_PATH}/val2014"
COCO_ANNO_PATH="${BASE_COCO_DATA_PATH}/annotations-2014/captions_val2014.json"
COCO_KARPATHY_PATH="${BASE_COCO_DATA_PATH}/dataset_coco.json"

# 4BI
CKPT_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-4B-vitl-rpj3b-langinstruct/snapshots/ef1d867b2bdf3e0ffec6d9870a07e6bd51eb7e88/checkpoint.pt"
LM_MODEL="togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
CROSS_ATTN_EVERY_N_LAYERS=2

SHOTS=$1
MASTER_PORT=$2
BS=$3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUs=`echo $CUDA_VISIBLE_DEVICES | grep -P -o '\d' | wc -l`
TIMESTAMP=`date +"%Y-%m-%d-%T"`
COMMENT="4BI-reproduce-coco-shots-${SHOTS}"
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
    --eval_coco \
    --coco_train_image_dir_path ${COCO_IMG_TRAIN_PATH} \
    --coco_val_image_dir_path ${COCO_IMG_VAL_PATH} \
    --coco_karpathy_json_path ${COCO_KARPATHY_PATH} \
    --coco_annotations_json_path ${COCO_ANNO_PATH}
