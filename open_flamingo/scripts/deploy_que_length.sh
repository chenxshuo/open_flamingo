# nohup bash open_flamingo/scripts/deploy_que_length.sh > out 2>&1 &


# on m4
#bash open_flamingo/scripts/run_eval_vqa_pseudo_que_length.sh 0 &> pseudo_que_length_0_vqa_OBI-4B.log
#bash open_flamingo/scripts/run_eval_vqa_pseudo_que_length.sh 2 &> pseudo_que_length_2_vqa_OBI-4B.log
#bash open_flamingo/scripts/run_eval_vqa_pseudo_que_length.sh 4 &> pseudo_que_length_4_vqa_OBI-4B.log

# on m3
#bash open_flamingo/scripts/run_eval_vqa_pseudo_que_length.sh 6 > pseudo_que_length_6_vqa_OBI-4B.log 2>&1 &
#bash open_flamingo/scripts/run_eval_vqa_pseudo_que_length.sh 8 > pseudo_que_length_8_vqa_OBI-4B.log 2>&1 &
#bash open_flamingo/scripts/run_eval_vqa_pseudo_que_length.sh 10 > pseudo_que_length_10_vqa_OBI-4B.log 2>&1 &

# on m1
bash open_flamingo/scripts/run_eval_vqa_pseudo_que_length.sh 12 > pseudo_que_length_12_vqa_OBI-4B.log 2>&1 &
bash open_flamingo/scripts/run_eval_vqa_pseudo_que_length.sh 14 > pseudo_que_length_14_vqa_OBI-4B.log 2>&1 &
bash open_flamingo/scripts/run_eval_vqa_pseudo_que_length.sh 16 > pseudo_que_length_16_vqa_OBI-4B.log 2>&1 &
