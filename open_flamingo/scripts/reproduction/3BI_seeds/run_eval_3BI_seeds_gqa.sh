#!/bin/bash
export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
ANNO_BASE="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/gqa"
GQA_IMG="${ANNO_BASE}/images"
GQA_TRAIN_QUES_PATH="${ANNO_BASE}/train_ques_vqav2_format.json"
GQA_TRAIN_ANNO_PATH="${ANNO_BASE}/train_anno_vqav2_format.json"
GQA_VAL_QUES_PATH="${ANNO_BASE}/test_ques_vqav2_format.json"
GQA_VAL_ANNO_PATH="${ANNO_BASE}/test_anno_vqav2_format.json"


#3BI
CKPT_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b-langinstruct/snapshots/656bbbcd4508db84ccc83c02361011c6fe92ae93/checkpoint.pt"
LM_MODEL="anas-awadalla/mpt-1b-redpajama-200b-dolly"
CROSS_ATTN_EVERY_N_LAYERS=1

SHOTS=$1
MASTER_PORT=$2
BS=$3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
    --num_trials 4 \
    --shots ${SHOTS} \
    --trial_seeds 12 22 32 52 \
    --demo_mode  "gold" \
    --visual_demo_mode "random" \
    --eval_gqa \
    --gqa_image_dir_path ${GQA_IMG} \
    --gqa_train_questions_json_path ${GQA_TRAIN_QUES_PATH} \
    --gqa_train_annotations_json_path ${GQA_TRAIN_ANNO_PATH} \
    --gqa_test_questions_json_path ${GQA_VAL_QUES_PATH} \
    --gqa_test_annotations_json_path ${GQA_VAL_ANNO_PATH} \
