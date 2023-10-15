#!/bin/bash
export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
#mnt
TEXTVQA_IMG_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/textvqa/train_val_images/train_images"
BASE_JSON_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/datasets--openflamingo--eval_benchmark/snapshots/2391a430b8bb92b7cf0677a541a180a310497d4f/textvqa"
TEXTVQA_TRAIN_QUES="${BASE_JSON_PATH}/train_questions_vqa_format.json"
TEXTVQA_TRAIN_ANNO="${BASE_JSON_PATH}/train_annotations_vqa_format.json"
TEXTVQA_VAL_QUES="${BASE_JSON_PATH}/val_questions_vqa_format.json"
TEXTVQA_VAL_ANNO="${BASE_JSON_PATH}/val_annotations_vqa_format.json"
OUT_DIR="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/textvqa/rice_features"

# 9B
CKPT_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-9B-vitl-mpt7b/snapshots/e6e175603712c7007fe3b9c0d50bdcfbd83adfc2/checkpoint.pt"
LM_MODEL="anas-awadalla/mpt-7b"
CROSS_ATTN_EVERY_N_LAYERS=4

SHOTS=$1
MASTER_PORT=$2
BS=$3
VISUAL_MODE=$4

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUs=`echo $CUDA_VISIBLE_DEVICES | grep -P -o '\d' | wc -l`
TIMESTAMP=`date +"%Y-%m-%d-%T"`
COMMENT="9B-rice-textvqa_${SHOTS}"
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
    --visual_demo_mode $VISUAL_MODE \
    --rices \
    --rices_every_nth \
    --cached_demonstration_features ${OUT_DIR} \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai \
    --eval_textvqa \
    --textvqa_image_dir_path ${TEXTVQA_IMG_PATH} \
    --textvqa_train_questions_json_path ${TEXTVQA_TRAIN_QUES} \
    --textvqa_train_annotations_json_path ${TEXTVQA_TRAIN_ANNO} \
    --textvqa_test_questions_json_path ${TEXTVQA_VAL_QUES} \
    --textvqa_test_annotations_json_path ${TEXTVQA_VAL_ANNO}
