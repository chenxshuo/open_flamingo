
data=('r' 'c' 's' 'a')
class=('8' '16' '32')

for dataset in "${data[@]}"; do
  for load in "${class[@]}"; do
    echo "Running imagenet-$dataset-class-$load-m4-t3"
    python main.py evaluate_dataset=imagenet-$dataset load_from=baseline-class-$load-m4-t3 > ./logs/imagenet-$dataset-class-$load-m4-t3.log 2>&1
  done
done
