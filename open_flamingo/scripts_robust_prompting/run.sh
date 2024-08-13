
{
# number of classes; port; shot;
declare -a DATASETS=(
                      "imagenet-1k" "imagenet-1k-novel"
                      "imagenet-a" "imagenet-a-novel"
                      "imagenet-c" "imagenet-c-novel"
#                      "imagenet-r" "imagenet-r-novel"
#                      "imagenet-s" "imagenet-s-novel"
#                      "imagenet-v2" "imagenet-v2-novel"
                      )

# dataset - number class - cuda number - port number
base_port=26000
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
#    base_port=$((base_port+1))
#    bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet.sh \
#      $dataset $c 0 $base_port > logs/9B_${dataset}_numclass-$c-shot-8.log 2>&1
#    base_port=$((base_port+1))
#    bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
#      $dataset $c 0 $base_port > logs/9B_rices_${dataset}_numclass-$c-shot-8.log 2>&1

#    base_port=$((base_port+1))
#    bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet_zero.sh \
#      $dataset $c 0 $base_port > logs/9B_${dataset}_numclass-$c-shot-0.log 2>&1

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet_zero_no_demo_text.sh \
      $dataset $c 0 $base_port > logs/9B_${dataset}_numclass-$c-shot-0_no_demo.log 2>&1

#    base_port=$((base_port+1))
#    bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet_zero.sh \
#      $dataset $c 0 $base_port > logs/9B_rices_${dataset}_numclass-$c-shot-0.log 2>&1

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet_zero_no_demo_text.sh \
      $dataset $c 0 $base_port > logs/9B_rices_${dataset}_numclass-$c-shot-0_no_demo.log 2>&1
  done
done
} &

#bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet.sh \
#      imagenet-1k 8 0 26000 > logs/9B_imagenet-1k_numclass-8-shot-8.log 2>&1

{
# number of classes; port; shot;
declare -a DATASETS=(
#                      "imagenet-1k" "imagenet-1k-novel"
#                      "imagenet-a" "imagenet-a-novel"
#                      "imagenet-c" "imagenet-c-novel"
                      "imagenet-r" "imagenet-r-novel"
                      "imagenet-s" "imagenet-s-novel"
                      "imagenet-v2" "imagenet-v2-novel"
                      )

# dataset - number class - cuda number - port number
base_port=25000
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
#    base_port=$((base_port+1))
#    bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet.sh \
#      $dataset $c 0 $base_port > logs/9B_${dataset}_numclass-$c-shot-8.log 2>&1
#    base_port=$((base_port+1))
#    bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
#      $dataset $c 0 $base_port > logs/9B_rices_${dataset}_numclass-$c-shot-8.log 2>&1

#    base_port=$((base_port+1))
#    bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet_zero.sh \
#      $dataset $c 0 $base_port > logs/9B_${dataset}_numclass-$c-shot-0.log 2>&1

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet_zero_no_demo_text.sh \
      $dataset $c 1 $base_port > logs/9B_${dataset}_numclass-$c-shot-0_no_demo.log 2>&1

#    base_port=$((base_port+1))
#    bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet_zero.sh \
#      $dataset $c 0 $base_port > logs/9B_rices_${dataset}_numclass-$c-shot-0.log 2>&1

    base_port=$((base_port+1))
    bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet_zero_no_demo_text.sh \
      $dataset $c 1 $base_port > logs/9B_rices_${dataset}_numclass-$c-shot-0_no_demo.log 2>&1
  done
done
} &

wait