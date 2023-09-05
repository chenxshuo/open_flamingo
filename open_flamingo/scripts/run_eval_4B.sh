#!/bin/bash
export HF_HOME="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface"
BASE_DATA_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/VL_adapter/datasets/COCO"
COCO_IMG_TRAIN_PATH="${BASE_DATA_PATH}/train2014/train2014"
COCO_IMG_VAL_PATH="${BASE_DATA_PATH}/val2014"
COCO_ANNO_PATH="${BASE_DATA_PATH}/annotations-2014/captions_val2014.json"
COCO_KARPATHY_PATH="${BASE_DATA_PATH}/dataset_coco.json"

VQAV2_IMG_TRAIN_PATH="${BASE_DATA_PATH}/train2014/train2014"
VQAV2_ANNO_TRAIN_PATH="${BASE_DATA_PATH}/v2_mscoco_train2014_annotations.json"
VQAV2_QUESTION_TRAIN_PATH="${BASE_DATA_PATH}/v2_OpenEnded_mscoco_train2014_questions.json"
VQAV2_IMG_TEST_PATH="${BASE_DATA_PATH}/val2014"
VQAV2_ANNO_TEST_PATH="${BASE_DATA_PATH}/v2_mscoco_val2014_annotations.json"
VQAV2_QUESTION_TEST_PATH="${BASE_DATA_PATH}/v2_OpenEnded_mscoco_val2014_questions.json"


# 9B
#CKPT_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-9B-vitl-mpt7b/snapshots/e6e175603712c7007fe3b9c0d50bdcfbd83adfc2/checkpoint.pt"
#LM_MODEL="anas-awadalla/mpt-7b"
#CROSS_ATTN_EVERY_N_LAYERS=4
# 4B
CKPT_PATH="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-4B-vitl-rpj3b-langinstruct/snapshots/ef1d867b2bdf3e0ffec6d9870a07e6bd51eb7e88/checkpoint.pt"
LM_MODEL="togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
CROSS_ATTN_EVERY_N_LAYERS=2

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUs=`echo $CUDA_VISIBLE_DEVICES | grep -P -o '\d' | wc -l`
TIMESTAMP=`date +"%Y-%m-%d-%T"`
COMMENT="4B-reproduce-vqav2"
RESULTS_FILE="results_${TIMESTAMP}_${COMMENT}.json"
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUs" open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path ${LM_MODEL} \
    --lm_tokenizer_path ${LM_MODEL} \
    --cross_attn_every_n_layers ${CROSS_ATTN_EVERY_N_LAYERS} \
    --checkpoint_path ${CKPT_PATH} \
    --results_file ${RESULTS_FILE} \
    --precision amp_bf16 \
    --batch_size 128 \
    --num_trials 1 \
    --trial_seeds 52 \
    --shots 0 \
    --demo_mode  "gold" \
    --visual_demo_mode "random" \
    --eval_vqav2 \
    --vqav2_train_image_dir_path ${VQAV2_IMG_TRAIN_PATH} \
    --vqav2_train_annotations_json_path ${VQAV2_ANNO_TRAIN_PATH} \
    --vqav2_train_questions_json_path ${VQAV2_QUESTION_TRAIN_PATH} \
    --vqav2_test_image_dir_path ${VQAV2_IMG_TEST_PATH} \
    --vqav2_test_annotations_json_path ${VQAV2_ANNO_TEST_PATH} \
    --vqav2_test_questions_json_path ${VQAV2_QUESTION_TEST_PATH}

#--eval_coco \
#    --coco_train_image_dir_path ${COCO_IMG_TRAIN_PATH} \
#    --coco_val_image_dir_path ${COCO_IMG_VAL_PATH} \
#    --coco_karpathy_json_path ${COCO_KARPATHY_PATH} \
#    --coco_annotations_json_path ${COCO_ANNO_PATH} \
#



# --shots 16 32 \

#--eval_flickr30 \
#--eval_ok_vqa \
#--eval_textvqa \
#--eval_vizwiz \
#--eval_hateful_memes \

#--flickr_image_dir_path "/path/to/flickr30k/flickr30k-images" \
#--flickr_karpathy_json_path "/path/to/flickr30k/dataset_flickr30k.json" \
#--flickr_annotations_json_path "/path/to/flickr30k/dataset_flickr30k_coco_style.json" \
#--ok_vqa_train_image_dir_path "/path/to/okvqa/train2014" \
#--ok_vqa_train_annotations_json_path "/path/to/okvqa/mscoco_train2014_annotations.json" \
#--ok_vqa_train_questions_json_path "/path/to/okvqa/OpenEnded_mscoco_train2014_questions.json" \
#--ok_vqa_test_image_dir_path "/path/to/okvqa/val2014" \
#--ok_vqa_test_annotations_json_path "/path/to/okvqa/mscoco_val2014_annotations.json" \
#--ok_vqa_test_questions_json_path "/path/to/okvqa/OpenEnded_mscoco_val2014_questions.json" \
#--textvqa_image_dir_path "/path/to/textvqa/train_images/" \
#--textvqa_train_questions_json_path "/path/to/textvqa/train_questions_vqa_format.json" \
#--textvqa_train_annotations_json_path "/path/to/textvqa/train_annotations_vqa_format.json" \
#--textvqa_test_questions_json_path "/path/to/textvqa/val_questions_vqa_format.json" \
#--textvqa_test_annotations_json_path "/path/to/textvqa/val_annotations_vqa_format.json" \
#--vizwiz_train_image_dir_path "/path/to/v7w/train" \
#--vizwiz_test_image_dir_path "/path/to/v7w/val" \
#--vizwiz_train_questions_json_path "/path/to/v7w/train_questions_vqa_format.json" \
#--vizwiz_train_annotations_json_path "/path/to/v7w/train_annotations_vqa_format.json" \
#--vizwiz_test_questions_json_path "/path/to/v7w/val_questions_vqa_format.json" \
#--vizwiz_test_annotations_json_path "/path/to/v7w/val_annotations_vqa_format.json" \
#--hateful_memes_image_dir_path "/path/to/hateful_memes/img" \
#--hateful_memes_train_annotations_json_path "/path/to/hateful_memes/train.json" \
#--hateful_memes_test_annotations_json_path "/path/to/hateful_memes/dev.json" \

