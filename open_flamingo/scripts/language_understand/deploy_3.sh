# nohup bash ./open_flamingo/scripts/language_understand/deploy_3.sh > ./logs/9B-vqav2-language_understand.log 2>&1 &

# VQAv2
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_strings_as_labels 26000 4 64 &> ./logs/eval_vqav2_language_random_strings_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_words_as_labels 26001 4 64 &> ./logs/eval_vqav2_language_random_words_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_outer_label_as_labels 26002 4 64 &> ./logs/eval_vqav2_language_random_outer_label_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_label_for_same_question_type_as_labels 26003 4 64 &> ./logs/eval_vqav2_language_random_label_for_same_question_type_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_label_for_same_question_as_labels 26004 4 64 &> ./logs/eval_vqav2_language_random_label_for_same_question_as_labels_shot4.log

bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_strings_as_labels 26000 8 64 &> ./logs/eval_vqav2_language_random_strings_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_words_as_labels 26001 8 64 &> ./logs/eval_vqav2_language_random_words_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_outer_label_as_labels 26002 8 64 &> ./logs/eval_vqav2_language_random_outer_label_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_label_for_same_question_type_as_labels 26003 8 64 &> ./logs/eval_vqav2_language_random_label_for_same_question_type_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_label_for_same_question_as_labels 26004 8 64 &> ./logs/eval_vqav2_language_random_label_for_same_question_as_labels_shot8.log



