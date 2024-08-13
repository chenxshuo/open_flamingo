# 8 classes
# base dir
{
CUDA_VISIBLE_DEVICES=0
BASE_DIR="./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-27_10-09-06"
echo "Evaluate 8 classes"
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        dir_name=$(basename $dir)
        echo "Evaluating $dir_name"
        for dataset in "imagenet-a" "imagenet-r" "imagenet-v2" "imagenet-s" "imagenet-c"
          do
            echo "Evaluating $dataset"
            bash scripts/run_eval_one.sh $dataset 8 False False "$dir" \
             > logs/9B_robust_false_${dataset}_novel_false_8_basedir_${dir_name}.log 2>&1
        done
    fi
done
} &

# 16 classes
# base dir
{
CUDA_VISIBLE_DEVICES=1
echo "Evaluate 16 classes"
BASE_DIR="./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_16/use_robust_prompting_False/media_prompts_8/text_prompts_per_media_3/2024-05-27_10-22-20"
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        dir_name=$(basename $dir)
        echo "Evaluating $dir_name"
        for dataset in "imagenet-a" "imagenet-r" "imagenet-v2" "imagenet-s" "imagenet-c"
          do
            echo "Evaluating $dataset"
            bash scripts/run_eval_one.sh $dataset 16 False False "$dir" \
             > logs/9B_robust_false_${dataset}_novel_false_16_basedir_${dir_name}.log 2>&1
        done
    fi
done
} &

wait
