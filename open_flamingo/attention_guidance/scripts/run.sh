## evaluate baseline ProL on base class
#data=('1k' 'v2' 'r' 'c' 's' 'a')
#class=('8' '16' '32')
#
#for dataset in "${data[@]}"; do
#  for load in "${class[@]}"; do
#    echo "Running imagenet-$dataset-class-$load-m4-t3"
#    python main.py evaluate_dataset=imagenet-$dataset load_from=baseline-class-$load-m4-t3 > ./logs/imagenet-$dataset-class-$load-m4-t3.log 2>&1
#  done
#done

# evaluate baseline ProL on novel class
#data=('1k' 'v2' 'r' 'c' 's' 'a')
#class=('8')
#for dataset in "${data[@]}"; do
#  for load in "${class[@]}"; do
#    echo "Running imagenet-$dataset-class-$load-m4-t3"
#    python main.py evaluate_dataset=imagenet-$dataset load_from=baseline-class-$load-m8-t3 eval_novel_classes=True > ./logs/imagenet-$dataset-class-$load-novel-class-m8-t3.log 2>&1
#  done
#done


#python main.py evaluate_dataset=imagenet-1k load_from=robust-sit eval_novel_classes=True device=cuda:1

#python main.py evaluate_dataset=imagenet-v2

nohup python main_train.py seed=22 device=cuda:0 > ./logs/train_22.log 2>&1 &

nohup python main_train.py seed=32 device=cuda:1 > ./logs/train_32.log 2>&1 &

nohup python main_train.py seed=42 device=cuda:2 > ./logs/train_42.log 2>&1 &

