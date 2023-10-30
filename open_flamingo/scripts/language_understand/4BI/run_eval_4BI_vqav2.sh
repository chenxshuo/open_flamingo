#!/bin/bash
export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
BASE_COCO_DATA_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO"
BASE_VQAv2_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/vqav2"
VQAV2_IMG_TRAIN_PATH="${BASE_COCO_DATA_PATH}/train2014/"
VQAV2_ANNO_TRAIN_PATH="${BASE_VQAv2_PATH}/v2_mscoco_train2014_annotations.json"
VQAV2_QUESTION_TRAIN_PATH="${BASE_VQAv2_PATH}/v2_OpenEnded_mscoco_train2014_questions.json"
VQAV2_IMG_TEST_PATH="${BASE_COCO_DATA_PATH}/val2014"

#VQAV2_ANNO_TEST_PATH="${BASE_COCO_DATA_PATH}/v2_mscoco_val2014_annotations.json"
#VQAV2_QUESTION_TEST_PATH="${BASE_COCO_DATA_PATH}/v2_OpenEnded_mscoco_val2014_questions.json"
VQAV2_ANNO_TEST_PATH="${BASE_VQAv2_PATH}/karpathy_test_ann_vqav2_format.json"
VQAV2_QUESTION_TEST_PATH="${BASE_VQAv2_PATH}/karpathy_test_ques_vqav2_format.json"

# 4BI
CKPT_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-4B-vitl-rpj3b-langinstruct/snapshots/ef1d867b2bdf3e0ffec6d9870a07e6bd51eb7e88/checkpoint.pt"
LM_MODEL="togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
CROSS_ATTN_EVERY_N_LAYERS=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUs=`echo $CUDA_VISIBLE_DEVICES | grep -P -o '\d' | wc -l`
TIMESTAMP=`date +"%Y-%m-%d-%T"`

MODE=$1
MASTER_PORT=$2
SHOT=$3
BS=$4

#MODE="only_labels"
#MODE="fixed_pseudo_question_length"
VISUAL_MODE="random"
COMMENT="9B-vqav2-$MODE-$VISUAL_MODE"

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
    --shots ${SHOT} \
    --trial_seeds 42 \
    --demo_mode  $MODE \
    --visual_demo_mode $VISUAL_MODE \
    --eval_vqav2 \
    --vqav2_train_image_dir_path ${VQAV2_IMG_TRAIN_PATH} \
    --vqav2_train_annotations_json_path ${VQAV2_ANNO_TRAIN_PATH} \
    --vqav2_train_questions_json_path ${VQAV2_QUESTION_TRAIN_PATH} \
    --vqav2_test_image_dir_path ${VQAV2_IMG_TEST_PATH} \
    --vqav2_test_annotations_json_path ${VQAV2_ANNO_TEST_PATH} \
    --vqav2_test_questions_json_path ${VQAV2_QUESTION_TEST_PATH}

