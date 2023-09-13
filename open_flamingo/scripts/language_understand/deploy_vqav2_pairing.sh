# nohup bash ./open_flamingo/scripts/language_understand/deploy_vqav2_pairing.sh > ./logs/9B-vqa-language_understand_vqav2.log 2>&1 &
# VQAv2

#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh no_labels 26000 4 32 &> ./logs/eval_vqav2_language_no_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh no_questions_no_labels 26004 4 32 &> ./logs/eval_vqav2_language_no_questions_no_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh only_labels 26003 4 32 &> ./logs/eval_vqav2_language_only_labels_shot4.log
#
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh no_labels 26001 8 16 &> ./logs/eval_vqav2_language_no_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh no_questions_no_labels 26005 8 16 &> ./logs/eval_vqav2_language_no_questions_no_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh only_labels 26003 8 16 &> ./logs/eval_vqav2_language_only_labels_shot8.log
#
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh no_labels 26001 16 8 &> ./logs/eval_vqav2_language_no_labels_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh no_questions_no_labels 26005 16 8 &> ./logs/eval_vqav2_language_no_questions_no_labels_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh only_labels 26003 16 8 &> ./logs/eval_vqav2_language_only_labels_shot16.log
#
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh no_labels 26001 32 8 &> ./logs/eval_vqav2_language_no_labels_shot32.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh no_questions_no_labels 26005 32 8 &> ./logs/eval_vqav2_language_no_questions_no_labels_shot32.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh only_labels 26003 32 8 &> ./logs/eval_vqav2_language_only_labels_shot32.log
