
base_port=`shuf -i 22000-26000 -n 1`
echo "base port: $base_port"

dataset="imagenet-r"
c=8

bash open_flamingo/scripts_robust_prompting/reproduction/run_eval_9B_imagenet.sh \
  $dataset $c 1 $base_port 8

#bash open_flamingo/scripts_robust_prompting/rices/run_eval_9B_imagenet.sh \
#  $dataset $c 0 $base_port 4
