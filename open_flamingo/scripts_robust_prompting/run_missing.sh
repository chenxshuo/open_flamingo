

# nohup bash open_flamingo/scripts_robust_prompting/run_missing.sh > logs/run_missing.log 2>&1 &
#{
#dataset="imagenet-1k"
#c=8
#
#bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet.sh \
#      $dataset $c 0 26000 2 > logs/9B_${dataset}_numclass-$c-shot-2.log 2>&1
#
#bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
#      $dataset $c 0 26001 2 > logs/9B_rices_${dataset}_numclass-$c-shot-2.log 2>&1
#} &

{
dataset="imagenet-1k"
c=8

#bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet.sh \
#      $dataset $c 1 26002 4 > logs/9B_${dataset}_numclass-$c-shot-4.log 2>&1

bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
      $dataset $c 1 26003 4 > logs/9B_rices_${dataset}_numclass-$c-shot-4.log 2>&1
} &

#{
#dataset="imagenet-1k"
#c=16
#
#bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
#      $dataset $c 2 26003 2 > logs/9B_rices_${dataset}_numclass-$c-shot-2.log 2>&1
#
#bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
#      $dataset $c 2 26003 4 > logs/9B_rices_${dataset}_numclass-$c-shot-4.log 2>&1
#
#c=32
#
#bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet.sh \
#      $dataset $c 2 26002 4 > logs/9B_${dataset}_numclass-$c-shot-4.log 2>&1
#
#} &
#
#wait
