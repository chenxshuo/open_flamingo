# nohup bash ./open_flamingo/scripts/language_understand/deploy_caption.sh > ./logs/9B-caption-language_understand.log 2>&1 &

bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_outer_label_as_labels 26000 4 64 &> ./logs/eval_flickr30k_language_random_outer_label_as_labels_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_outer_label_as_labels 26001 8 64 &> ./logs/eval_flickr30k_language_random_outer_label_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_outer_label_as_labels 26002 16 32 &> ./logs/eval_flickr30k_language_random_outer_label_as_labels_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_outer_label_as_labels 26003 32 16 &> ./logs/eval_flickr30k_language_random_outer_label_as_labels_shot32.log

bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_strings_as_labels 26000 4 64 &> ./logs/eval_flickr30k_language_random_strings_as_labels_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_strings_as_labels 26001 8 64 &> ./logs/eval_flickr30k_language_random_strings_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_strings_as_labels 26002 16 32 &> ./logs/eval_flickr30k_language_random_strings_as_labels_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_strings_as_labels 26003 32 16 &> ./logs/eval_flickr30k_language_random_strings_as_labels_shot32.log

bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_words_as_labels 26004 4 64 &> ./logs/eval_flickr30k_language_random_words_as_labels_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_words_as_labels 26005 8 64 &> ./logs/eval_flickr30k_language_random_words_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_words_as_labels 26006 16 32 &> ./logs/eval_flickr30k_language_random_words_as_labels_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh random_words_as_labels 26007 32 16 &> ./logs/eval_flickr30k_language_random_words_as_labels_shot32.log

bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh ood_inputs 26004 4 64 &> ./logs/eval_flickr30k_language_ood_inputs_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh ood_inputs 26005 8 64 &> ./logs/eval_flickr30k_language_ood_inputs_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh ood_inputs 26006 16 32 &> ./logs/eval_flickr30k_language_ood_inputs_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_flickr30k.sh ood_inputs 26007 32 16 &> ./logs/eval_flickr30k_language_ood_inputs_shot32.log

bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_outer_label_as_labels 26000 4 64 &> ./logs/eval_coco_language_random_outer_label_as_labels_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_outer_label_as_labels 26001 8 64 &> ./logs/eval_coco_language_random_outer_label_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_outer_label_as_labels 26002 16 32 &> ./logs/eval_coco_language_random_outer_label_as_labels_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_outer_label_as_labels 26003 32 16 &> ./logs/eval_coco_language_random_outer_label_as_labels_shot32.log

bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_strings_as_labels 26000 4 64 &> ./logs/eval_coco_language_random_strings_as_labels_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_strings_as_labels 26001 8 64 &> ./logs/eval_coco_language_random_strings_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_strings_as_labels 26002 16 32 &> ./logs/eval_coco_language_random_strings_as_labels_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_strings_as_labels 26003 32 16 &> ./logs/eval_coco_language_random_strings_as_labels_shot32.log

bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_words_as_labels 26004 4 64 &> ./logs/eval_coco_language_random_words_as_labels_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_words_as_labels 26005 8 64 &> ./logs/eval_coco_language_random_words_as_labels_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_words_as_labels 26006 16 32 &> ./logs/eval_coco_language_random_words_as_labels_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh random_words_as_labels 26007 32 16 &> ./logs/eval_coco_language_random_words_as_labels_shot32.log

bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh ood_inputs 26004 4 64 &> ./logs/eval_coco_language_ood_inputs_shot4.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh ood_inputs 26005 8 64 &> ./logs/eval_coco_language_ood_inputs_shot8.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh ood_inputs 26006 16 32 &> ./logs/eval_coco_language_ood_inputs_shot16.log
bash open_flamingo/scripts/language_understand/run_eval_9B_coco.sh ood_inputs 26007 32 16 &> ./logs/eval_coco_language_ood_inputs_shot32.log
