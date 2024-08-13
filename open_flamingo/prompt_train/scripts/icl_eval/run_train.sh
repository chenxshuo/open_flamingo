#!/bin/bash

#  bash scripts/icl_eval/run.sh
# training OF-3B on imagenet-1k

## not use robust prompting
#for NUMBER_OF_CLASSES in 8 16 32; do
#    echo "not use robust prompting"
#    echo "NUMBER_OF_CLASSES: $NUMBER_OF_CLASSES"
#    bash scripts/icl_eval/run_train_one.sh $NUMBER_OF_CLASSES False > logs/3B_train_robust_false_imagenet-1k_${NUMBER_OF_CLASSES}.log 2>&1
#done
#

#
## use robust prompting
#for NUMBER_OF_CLASSES in 8 16 32; do
#  echo "use robust prompting"
#  echo "NUMBER_OF_CLASSES: $NUMBER_OF_CLASSES"
#  bash scripts/icl_eval/run_train_one.sh $NUMBER_OF_CLASSES True > logs/3B_train_robust_true_imagenet-1k_${NUMBER_OF_CLASSES}.log 2>&1
#done

#bash scripts/icl_eval/run_train_one.sh 8 True 2>&1 | tee logs/3B_train_robust_true_imagenet-1k_8.log

#bash scripts/icl_eval/run_train_one.sh 8 False 2>&1 | tee logs/3B_train_robust_false_imagenet-1k_8_softtoken.log

CUDA_VISIBLE_DEVICES=1
bash scripts/icl_eval/run_train_one.sh 8 False 2>&1 | tee logs/3B_train_icl_robust_false_imagenet-1k_8.log


#nohup bash scripts/icl_eval/run_train_one.sh 8 False > logs/3B_train_robust_false_imagenet-1k_8.log 2>&1 &
#nohup bash scripts/icl_eval/run_train_one.sh 16 False > logs/3B_train_robust_false_imagenet-1k_16.log 2>&1 &
#nohup bash scripts/icl_eval/run_train_one.sh 32 False > logs/3B_train_robust_false_imagenet-1k_32.log 2>&1 &


CUDA_VISIBLE_DEVICES=1
#nohup bash scripts/icl_eval/run_train_one.sh 8 True > logs/3B_train_robust_true_imagenet-1k_8.log 2>&1 &
#nohup bash scripts/icl_eval/run_train_one.sh 16 True > logs/3B_train_robust_true_imagenet-1k_16.log 2>&1 &
#nohup bash scripts/icl_eval/run_train_one.sh 32 True > logs/3B_train_robust_true_imagenet-1k_32.log 2>&1 &

#bash scripts/icl_eval/run_train_one.sh 8 True 2>&1 | tee logs/3B_train_robust_true_imagenet-1k_8_five_crop.log
#bash scripts/icl_eval/run_train_one.sh 8 True 2>&1 | tee logs/3B_train_robust_true_imagenet-1k_8_center_crop.log
#bash scripts/icl_eval/run_train_one.sh 8 True 2>&1 | tee logs/3B_train_robust_true_imagenet-1k_8_crop_ratio.log
