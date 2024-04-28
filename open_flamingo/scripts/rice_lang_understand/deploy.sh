# RICES selection - then evaluate language information importance

#nohup bash ./open_flamingo/scripts/rice_lang_understand/deploy.sh > ./logs/9B-rice_lang_understand.log 2>&1 &

bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_label_for_same_question_as_labels 26000 4 32 &> ./logs/eval_rice_gqa_language_random_label_for_same_question_as_labels_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_label_for_same_question_as_labels 26001 8 16 &> ./logs/eval_rice_gqa_language_random_label_for_same_question_as_labels_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_label_for_same_question_as_labels 26002 16 8 &> ./logs/eval_rice_gqa_language_random_label_for_same_question_as_labels_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_label_for_same_question_as_labels 26003 32 4 &> ./logs/eval_rice_gqa_language_random_label_for_same_question_as_labels_shot32.log

bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_question_inputs 26000 4 32 &> ./logs/eval_rice_gqa_language_random_question_inputs_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_question_inputs 26001 8 16 &> ./logs/eval_rice_gqa_language_random_question_inputs_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_question_inputs 26002 16 8 &> ./logs/eval_rice_gqa_language_random_question_inputs_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_question_inputs 26003 32 4 &> ./logs/eval_rice_gqa_language_random_question_inputs_shot32.log

bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_words_as_labels 26000 4 32 &> ./logs/eval_rice_gqa_language_random_words_as_labels_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_words_as_labels 26001 8 16 &> ./logs/eval_rice_gqa_language_random_words_as_labels_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_words_as_labels 26002 16 8 &> ./logs/eval_rice_gqa_language_random_words_as_labels_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_gqa.sh random_words_as_labels 26003 32 4 &> ./logs/eval_rice_gqa_language_random_words_as_labels_shot32.log

bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_label_for_same_question_as_labels 26000 4 32 &> ./logs/eval_rice_coco_language_random_label_for_same_question_as_labels_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_label_for_same_question_as_labels 26001 8 16 &> ./logs/eval_rice_coco_language_random_label_for_same_question_as_labels_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_label_for_same_question_as_labels 26002 16 8 &> ./logs/eval_rice_coco_language_random_label_for_same_question_as_labels_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_label_for_same_question_as_labels 26003 32 4 &> ./logs/eval_rice_coco_language_random_label_for_same_question_as_labels_shot32.log

bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_question_inputs 26000 4 32 &> ./logs/eval_rice_coco_language_random_question_inputs_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_question_inputs 26001 8 16 &> ./logs/eval_rice_coco_language_random_question_inputs_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_question_inputs 26002 16 8 &> ./logs/eval_rice_coco_language_random_question_inputs_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_question_inputs 26003 32 4 &> ./logs/eval_rice_coco_language_random_question_inputs_shot32.log

bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_words_as_labels 26000 4 32 &> ./logs/eval_rice_coco_language_random_words_as_labels_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_words_as_labels 26001 8 16 &> ./logs/eval_rice_coco_language_random_words_as_labels_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_words_as_labels 26002 16 8 &> ./logs/eval_rice_coco_language_random_words_as_labels_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_coco.sh random_words_as_labels 26003 32 4 &> ./logs/eval_rice_coco_language_random_words_as_labels_shot32.log

bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_label_for_same_question_as_labels 26000 4 32 &> ./logs/eval_rice_okvqa_language_random_label_for_same_question_as_labels_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_label_for_same_question_as_labels 26001 8 16 &> ./logs/eval_rice_okvqa_language_random_label_for_same_question_as_labels_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_label_for_same_question_as_labels 26002 16 8 &> ./logs/eval_rice_okvqa_language_random_label_for_same_question_as_labels_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_label_for_same_question_as_labels 26003 32 4 &> ./logs/eval_rice_okvqa_language_random_label_for_same_question_as_labels_shot32.log

bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_question_inputs 26000 4 32 &> ./logs/eval_rice_okvqa_language_random_question_inputs_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_question_inputs 26001 8 16 &> ./logs/eval_rice_okvqa_language_random_question_inputs_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_question_inputs 26002 16 8 &> ./logs/eval_rice_okvqa_language_random_question_inputs_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_question_inputs 26003 32 4 &> ./logs/eval_rice_okvqa_language_random_question_inputs_shot32.log

bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_words_as_labels 26000 4 32 &> ./logs/eval_rice_okvqa_language_random_words_as_labels_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_words_as_labels 26001 8 16 &> ./logs/eval_rice_okvqa_language_random_words_as_labels_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_words_as_labels 26002 16 8 &> ./logs/eval_rice_okvqa_language_random_words_as_labels_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_okvqa.sh random_words_as_labels 26003 32 4 &> ./logs/eval_rice_okvqa_language_random_words_as_labels_shot32.log


bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_label_for_same_question_as_labels 26000 4 32 &> ./logs/eval_rice_vqav2_language_random_label_for_same_question_as_labels_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_label_for_same_question_as_labels 26001 8 16 &> ./logs/eval_rice_vqav2_language_random_label_for_same_question_as_labels_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_label_for_same_question_as_labels 26002 16 8 &> ./logs/eval_rice_vqav2_language_random_label_for_same_question_as_labels_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_label_for_same_question_as_labels 26003 32 4 &> ./logs/eval_rice_vqav2_language_random_label_for_same_question_as_labels_shot32.log

bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_question_inputs 26000 4 32 &> ./logs/eval_rice_vqav2_language_random_question_inputs_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_question_inputs 26001 8 16 &> ./logs/eval_rice_vqav2_language_random_question_inputs_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_question_inputs 26002 16 8 &> ./logs/eval_rice_vqav2_language_random_question_inputs_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_question_inputs 26003 32 4 &> ./logs/eval_rice_vqav2_language_random_question_inputs_shot32.log

bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_words_as_labels 26000 4 32 &> ./logs/eval_rice_vqav2_language_random_words_as_labels_shot4.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_words_as_labels 26001 8 16 &> ./logs/eval_rice_vqav2_language_random_words_as_labels_shot8.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_words_as_labels 26002 16 8 &> ./logs/eval_rice_vqav2_language_random_words_as_labels_shot16.log
bash open_flamingo/scripts/rice_lang_understand/run_eval_9B_vqav2.sh random_words_as_labels 26003 32 4 &> ./logs/eval_rice_vqav2_language_random_words_as_labels_shot32.log
