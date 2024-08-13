#!/bin/bash
export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
BASE_COCO_DATA_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO"
COCO_IMG_TRAIN_PATH="${BASE_COCO_DATA_PATH}/train2014"
COCO_IMG_VAL_PATH="${BASE_COCO_DATA_PATH}/val2014"
BASE_DATA_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets"
OK_VQA_TRAIN_ANNO_PATH="${BASE_DATA_PATH}/okvqa/mscoco_train2014_annotations.json"
OK_VQA_TRAIN_QUES_PATH="${BASE_DATA_PATH}/okvqa/OpenEnded_mscoco_train2014_questions.json"
OK_VQA_VAL_ANNO_PATH="${BASE_DATA_PATH}/okvqa/mscoco_val2014_annotations.json"
OK_VQA_VAL_QUES_PATH="${BASE_DATA_PATH}/okvqa/OpenEnded_mscoco_val2014_questions.json"

#3B
CKPT_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt"
LM_MODEL="anas-awadalla/mpt-1b-redpajama-200b"
CROSS_ATTN_EVERY_N_LAYERS=1


SHOTS=$1
MASTER_PORT=$2
BS=$3

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUs=`echo $CUDA_VISIBLE_DEVICES | grep -P -o '\d' | wc -l`
TIMESTAMP=`date +"%Y-%m-%d-%T"`
COMMENT="4B-reproduce-okvqa-shots-${SHOTS}"
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
    --eval_ok_vqa \
    --ok_vqa_train_image_dir_path ${COCO_IMG_TRAIN_PATH} \
    --ok_vqa_train_annotations_json_path ${OK_VQA_TRAIN_ANNO_PATH} \
    --ok_vqa_train_questions_json_path ${OK_VQA_TRAIN_QUES_PATH} \
    --ok_vqa_test_image_dir_path ${COCO_IMG_VAL_PATH} \
    --ok_vqa_test_annotations_json_path ${OK_VQA_VAL_ANNO_PATH} \
    --ok_vqa_test_questions_json_path ${OK_VQA_VAL_QUES_PATH} \


