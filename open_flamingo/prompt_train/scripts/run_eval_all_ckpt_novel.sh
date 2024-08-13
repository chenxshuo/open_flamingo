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
        for dataset in "imagenet-1k" # "imagenet-a" "imagenet-r" "imagenet-v2" "imagenet-s" "imagenet-c"
          do
            echo "Evaluating $dataset"
            bash scripts/run_eval_one.sh $dataset 8 False True "$dir" \
             > logs/9B_robust_false_${dataset}_novel_true_8_basedir_${dir_name}.log 2>&1
        done
    fi
done
} &

# 8 classes
# base dir
# robust true
{
  CUDA_VISIBLE_DEVICES=1
  BASE_DIR="./experiments/model_OF-9B/evaluate_dataset_imagenet-1k/classes_8/use_robust_prompting_True/media_prompts_8/text_prompts_per_media_3/2024-05-27_10-57-32"
echo "Evaluate 8 classes"
for dir in "$BASE_DIR"/*; do
    if [ -d "$dir" ]; then
        dir_name=$(basename $dir)
        echo "Evaluating $dir_name"
        for dataset in "imagenet-1k" # "imagenet-a" "imagenet-r" "imagenet-v2" "imagenet-s" "imagenet-c"
          do
            echo "Evaluating $dataset"
            # data - number of classes - robust prompting - novel classes - load from dir
            bash scripts/run_eval_one.sh $dataset 8 True True "$dir" \
             > logs/9B_robust_true_${dataset}_novel_true_8_basedir_${dir_name}.log 2>&1
        done
    fi
done
} &

wait