
{
  dataset="imagenet-c-novel"
  c=8
bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
      $dataset $c 1 26004 4 > logs/9B_rices_${dataset}_numclass-$c-shot-4.log 2>&1
} &

wait