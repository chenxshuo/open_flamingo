

#for n in 8 16 32
#do
#  for d in imagenet-1k imagenet-a imagenet-r imagenet-v2 imagenet-s imagenet-c
#  do
#    bash knn.sh $n $d > logs/knn_${d}_${n}.log 2>&1
#  done
#done


#for n in 8 16 32
#do
#    bash knn.sh $n imagenet-c > logs/knn_imagenet-s_${n}.log 2>&1
#done

# train
#bash supervised.sh 8 imagenet-1k False > logs/supervised_imagenet-1k_8_No_Few-Shot.log 2>&1
#
#for n in 16 32
#do
#    bash supervised.sh $n imagenet-1k True > logs/supervised_imagenet-1k_${n}_Few_shot.log 2>&1
#    bash supervised.sh $n imagenet-1k False > logs/supervised_imagenet-1k_${n}_No_Few-Shot.log 2>&1
#done

#bash supervised_eval.sh 8 imagenet-1k True True

# evaluate on supervised
#for d in imagenet-1k imagenet-a imagenet-r imagenet-v2 imagenet-s imagenet-c
for d in imagenet-c
do
  for n in 8 16 32
  do
    echo "Starting evaluation for dataset $d with $n classes"
    pids=()
    nohup bash supervised_eval.sh $n $d True False > logs/supervised_eval_${d}_${n}_Few_shot_No_Novel.log 2>&1 &
    pids+=($!)
    sleep 10 # wait for the first process to start
    nohup bash supervised_eval.sh $n $d False False > logs/supervised_eval_${d}_${n}_No_Few_shot_No_Novel.log 2>&1 &
    pids+=($!)
#    nohup bash supervised_eval.sh $n $d True True > logs/supervised_eval_${d}_${n}_Few_shot_Novel.log 2>&1 &
#    pids+=($!)
#    nohup bash supervised_eval.sh $n $d False True > logs/supervised_eval_${d}_${n}_No_Few_shot_Novel.log 2>&1 &
#    pids+=($!)

    echo "active processes: ${#pids[@]} for dataset $d with $n classes"
    for ((i=${#pids[@]}; i>1; i--)) ; do
        wait -n
    done
    echo "all processes finished for dataset $d with $n classes"
  done
done
