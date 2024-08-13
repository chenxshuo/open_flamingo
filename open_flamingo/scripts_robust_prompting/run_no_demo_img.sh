# nohup bash open_flamingo/scripts_robust_prompting/run_no_demo_img.sh > logs/run_no_demo_img.log 2>&1 &

{
# number of classes; port; shot;
declare -a DATASETS=(
                      "imagenet-1k" "imagenet-1k-novel"
                      "imagenet-a" "imagenet-a-novel"
                      "imagenet-c" "imagenet-c-novel"
                      )

# dataset - number class - cuda number - port number
base_port=`shuf -i 22000-26000 -n 1`
for dataset in "${DATASETS[@]}"
do
  for c in 8 16 32
  do
    if [[ $dataset == *"novel"* ]]; then
      if [[ $c -ne 8 ]]; then
        continue
      fi
    fi
    echo "Running $dataset with $c classes"

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet.sh \
      $dataset $c 2 $base_port 4 no_images > logs/9B_${dataset}_numclass-$c-shot-4_no_demo_images.log 2>&1

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
      $dataset $c 2 $base_port 4 no_images > logs/9B_rices_${dataset}_numclass-$c-shot-4_no_demo_images.log 2>&1

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet.sh \
      $dataset $c 2 $base_port 8 no_images > logs/9B_${dataset}_numclass-$c-shot-8_no_demo_images.log 2>&1

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
      $dataset $c 2 $base_port 8 no_images > logs/9B_rices_${dataset}_numclass-$c-shot-8_no_demo_images.log 2>&1

  done
done
} &

{
# number of classes; port; shot;
declare -a DATASETS=(
                      "imagenet-r" "imagenet-r-novel"
                      "imagenet-s" "imagenet-s-novel"
                      "imagenet-v2" "imagenet-v2-novel"
                      )

# dataset - number class - cuda number - port number
base_port=`shuf -i 22000-26000 -n 1`
for dataset in "${DATASETS[@]}"
do
  for c in 8 16 32
  do
    if [[ $dataset == *"novel"* ]]; then
      if [[ $c -ne 8 ]]; then
        continue
      fi
    fi
    echo "Running $dataset with $c classes"

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet.sh \
      $dataset $c 3 $base_port 4 no_images > logs/9B_${dataset}_numclass-$c-shot-4_no_demo_images.log 2>&1

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
      $dataset $c 3 $base_port 4 no_images > logs/9B_rices_${dataset}_numclass-$c-shot-4_no_demo_images.log 2>&1

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet.sh \
      $dataset $c 3 $base_port 8 no_images > logs/9B_${dataset}_numclass-$c-shot-8_no_demo_images.log 2>&1

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
      $dataset $c 3 $base_port 8 no_images > logs/9B_rices_${dataset}_numclass-$c-shot-8_no_demo_images.log 2>&1

  done
done
} &

wait