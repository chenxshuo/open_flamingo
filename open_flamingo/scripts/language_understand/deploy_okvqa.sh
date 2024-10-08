
# nohup bash ./open_flamingo/scripts/language_understand/deploy_okvqa.sh > ./logs/9B-okvqa-language_understand.log 2>&1 &
# OKVQA

#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_outer_label_as_labels 26000 4 64 &> ./logs/eval_okvqa_language_random_outer_label_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_outer_label_as_labels 26001 8 64 &> ./logs/eval_okvqa_language_random_outer_label_as_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_outer_label_as_labels 26002 16 32 &> ./logs/eval_okvqa_language_random_outer_label_as_labels_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_outer_label_as_labels 26003 32 16 &> ./logs/eval_okvqa_language_random_outer_label_as_labels_shot32.log
#
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_label_for_same_question_as_labels 26000 4 64 &> ./logs/eval_okvqa_language_random_label_for_same_question_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_label_for_same_question_as_labels 26001 8 64 &> ./logs/eval_okvqa_language_random_label_for_same_question_as_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_label_for_same_question_as_labels 26002 16 32 &> ./logs/eval_okvqa_language_random_label_for_same_question_as_labels_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_label_for_same_question_as_labels 26003 32 16 &> ./logs/eval_okvqa_language_random_label_for_same_question_as_labels_shot32.log
#
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_question_inputs 26004 4 64 &> ./logs/eval_okvqa_language_random_question_inputs_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_question_inputs 26005 8 64 &> ./logs/eval_okvqa_language_random_question_inputs_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_question_inputs 26006 16 8 &> ./logs/eval_okvqa_language_random_question_inputs_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_question_inputs 26007 32 8 &> ./logs/eval_okvqa_language_random_question_inputs_shot32.log
#
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh no_labels 26004 4 64 &> ./logs/eval_okvqa_language_no_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh no_labels 26005 8 64 &> ./logs/eval_okvqa_language_no_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh no_labels 26006 16 8 &> ./logs/eval_okvqa_language_no_labels_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh no_labels 26007 32 8 &> ./logs/eval_okvqa_language_no_labels_shot32.log

bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh no_questions_no_labels 26004 4 64 &> ./logs/eval_okvqa_language_no_question_no_labels_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh no_questions_no_labels 26005 8 64 &> ./logs/eval_okvqa_language_no_question_no_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh no_questions_no_labels 26006 16 8 &> ./logs/eval_okvqa_language_no_question_no_labels_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh no_questions_no_labels 26007 32 8 &> ./logs/eval_okvqa_language_no_question_no_labels_shot32.log

#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh only_labels 26004 4 64 &> ./logs/eval_okvqa_language_only_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh only_labels 26005 8 64 &> ./logs/eval_okvqa_language_only_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh only_labels 26006 16 8 &> ./logs/eval_okvqa_language_only_labels_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh only_labels 26007 32 8 &> ./logs/eval_okvqa_language_only_labels_shot32.log


#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_strings_as_labels 26000 4 64 &> ./logs/eval_okvqa_language_random_strings_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_strings_as_labels 26001 8 64 &> ./logs/eval_okvqa_language_random_strings_as_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_strings_as_labels 26002 16 32 &> ./logs/eval_okvqa_language_random_strings_as_labels_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_strings_as_labels 26003 32 16 &> ./logs/eval_okvqa_language_random_strings_as_labels_shot32.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_words_as_labels 26004 4 64 &> ./logs/eval_okvqa_language_random_words_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_words_as_labels 26005 8 64 &> ./logs/eval_okvqa_language_random_words_as_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_words_as_labels 26006 16 32 &> ./logs/eval_okvqa_language_random_words_as_labels_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_words_as_labels 26007 32 16 &> ./logs/eval_okvqa_language_random_words_as_labels_shot32.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_strings_inputs 26000 4 64 &> ./logs/eval_okvqa_language_random_strings_inputs_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_strings_inputs 26001 8 64 &> ./logs/eval_okvqa_language_random_strings_inputs_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_strings_inputs 26002 16 32 &> ./logs/eval_okvqa_language_random_strings_inputs_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh random_strings_inputs 26003 32 8 &> ./logs/eval_okvqa_language_random_strings_inputs_shot32.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh ood_inputs 26004 4 64 &> ./logs/eval_okvqa_language_ood_inputs_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh ood_inputs 26005 8 64 &> ./logs/eval_okvqa_language_ood_inputs_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh ood_inputs 26006 16 8 &> ./logs/eval_okvqa_language_ood_inputs_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_okvqa.sh ood_inputs 26007 32 8 &> ./logs/eval_okvqa_language_ood_inputs_shot32.log
