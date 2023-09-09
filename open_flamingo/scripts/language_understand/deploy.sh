# nohup bash ./open_flamingo/scripts/language_understand/deploy.sh > ./logs/9B-vqa-language_understand.log 2>&1 &

# VQAv2
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_strings_as_labels 26000 4 64 &> ./logs/eval_vqav2_language_random_strings_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_words_as_labels 26001 4 64 &> ./logs/eval_vqav2_language_random_words_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_outer_label_as_labels 26002 4 64 &> ./logs/eval_vqav2_language_random_outer_label_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_label_for_same_question_type_as_labels 26003 4 64 &> ./logs/eval_vqav2_language_random_label_for_same_question_type_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_label_for_same_question_as_labels 26004 4 64 &> ./logs/eval_vqav2_language_random_label_for_same_question_as_labels_shot4.log
#
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_strings_as_labels 26000 8 64 &> ./logs/eval_vqav2_language_random_strings_as_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_words_as_labels 26001 8 64 &> ./logs/eval_vqav2_language_random_words_as_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_outer_label_as_labels 26002 8 64 &> ./logs/eval_vqav2_language_random_outer_label_as_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_label_for_same_question_type_as_labels 26003 8 64 &> ./logs/eval_vqav2_language_random_label_for_same_question_type_as_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_vqav2.sh random_label_for_same_question_as_labels 26004 8 64 &> ./logs/eval_vqav2_language_random_label_for_same_question_as_labels_shot8.log



# textvqa
#bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_strings_as_labels 26000 4 64 &> ./logs/eval_textvqa_language_random_strings_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_strings_as_labels 26001 8 64 &> ./logs/eval_textvqa_language_random_strings_as_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_strings_as_labels 26002 16 32 &> ./logs/eval_textvqa_language_random_strings_as_labels_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_strings_as_labels 26003 32 16 &> ./logs/eval_textvqa_language_random_strings_as_labels_shot32.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_words_as_labels 26004 4 64 &> ./logs/eval_textvqa_language_random_words_as_labels_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_words_as_labels 26005 8 64 &> ./logs/eval_textvqa_language_random_words_as_labels_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_words_as_labels 26006 16 32 &> ./logs/eval_textvqa_language_random_words_as_labels_shot16.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_words_as_labels 26007 32 16 &> ./logs/eval_textvqa_language_random_words_as_labels_shot32.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_strings_inputs 26000 4 64 &> ./logs/eval_textvqa_language_random_strings_inputs_shot4.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_strings_inputs 26001 8 64 &> ./logs/eval_textvqa_language_random_strings_inputs_shot8.log
#bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_strings_inputs 26002 16 32 &> ./logs/eval_textvqa_language_random_strings_inputs_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh random_strings_inputs 26003 32 16 &> ./logs/eval_textvqa_language_random_strings_inputs_shot32.log
bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh ood_inputs 26004 4 64 &> ./logs/eval_textvqa_language_ood_inputs_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh ood_inputs 26005 8 64 &> ./logs/eval_textvqa_language_ood_inputs_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh ood_inputs 26006 16 32 &> ./logs/eval_textvqa_language_ood_inputs_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_textvqa.sh ood_inputs 26007 32 16 &> ./logs/eval_textvqa_language_ood_inputs_shot32.log

# vizwiz
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_strings_as_labels 26000 4 64 &> ./logs/eval_vizwiz_language_random_strings_as_labels_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_strings_as_labels 26001 8 64 &> ./logs/eval_vizwiz_language_random_strings_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_strings_as_labels 26002 16 32 &> ./logs/eval_vizwiz_language_random_strings_as_labels_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_strings_as_labels 26003 32 16 &> ./logs/eval_vizwiz_language_random_strings_as_labels_shot32.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_words_as_labels 26004 4 64 &> ./logs/eval_vizwiz_language_random_words_as_labels_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_words_as_labels 26005 8 64 &> ./logs/eval_vizwiz_language_random_words_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_words_as_labels 26006 16 32 &> ./logs/eval_vizwiz_language_random_words_as_labels_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_words_as_labels 26007 32 16 &> ./logs/eval_vizwiz_language_random_words_as_labels_shot32.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_strings_inputs 26000 4 64 &> ./logs/eval_vizwiz_language_random_strings_inputs_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_strings_inputs 26001 8 64 &> ./logs/eval_vizwiz_language_random_strings_inputs_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_strings_inputs 26002 16 32 &> ./logs/eval_vizwiz_language_random_strings_inputs_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh random_strings_inputs 26003 32 16 &> ./logs/eval_vizwiz_language_random_strings_inputs_shot32.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh ood_inputs 26004 4 64 &> ./logs/eval_vizwiz_language_ood_inputs_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh ood_inputs 26005 8 64 &> ./logs/eval_vizwiz_language_ood_inputs_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh ood_inputs 26006 16 32 &> ./logs/eval_vizwiz_language_ood_inputs_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_vizwiz.sh ood_inputs 26007 32 16 &> ./logs/eval_vizwiz_language_ood_inputs_shot32.log
