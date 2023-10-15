# nohup bash open_flamingo/scripts/rice_img/every_nth/deploy_2.sh &> ./logs/deploy_2_rices_every_nth.log 2>&1 &


bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_flickr30k.sh 4 26000 32 random &> ./logs/eval_flickr30k_rices_every_nth_shot_4_visual_random.log
bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_flickr30k.sh 4 26001 32 no_images &> ./logs/eval_flickr30k_rices_every_nth_shot_4_visual_no_images.log
bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_flickr30k.sh 8 26002 32 random &> ./logs/eval_flickr30k_rices_every_nth_shot_8_visual_random.log
bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_flickr30k.sh 8 26003 32 no_images &> ./logs/eval_flickr30k_rices_every_nth_shot_8_visual_no_images.log
bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_flickr30k.sh 16 26004 8 random &> ./logs/eval_flickr30k_rices_every_nth_shot_16_visual_random.log
bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_flickr30k.sh 16 26005 8 no_images &> ./logs/eval_flickr30k_rices_every_nth_shot_16_visual_no_images.log
bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_flickr30k.sh 32 26006 8 random &> ./logs/eval_flickr30k_rices_every_nth_shot_32_visual_random.log
bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_flickr30k.sh 32 26007 8 no_images &> ./logs/eval_flickr30k_rices_every_nth_shot_32_visual_no_images.log


#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vqav2.sh 4 26000 32 random &> ./logs/eval_vqav2_rices_every_nth_shot_4_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vqav2.sh 4 26001 32 no_images &> ./logs/eval_vqav2_rices_every_nth_shot_4_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vqav2.sh 8 26002 32 random &> ./logs/eval_vqav2_rices_every_nth_shot_8_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vqav2.sh 8 26003 32 no_images &> ./logs/eval_vqav2_rices_every_nth_shot_8_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vqav2.sh 16 26004 8 random &> ./logs/eval_vqav2_rices_every_nth_shot_16_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vqav2.sh 16 26005 8 no_images &> ./logs/eval_vqav2_rices_every_nth_shot_16_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vqav2.sh 32 26006 8 random &> ./logs/eval_vqav2_rices_every_nth_shot_32_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vqav2.sh 32 26007 8 no_images &> ./logs/eval_vqav2_rices_every_nth_shot_32_visual_no_images.log
#
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_gqa.sh 4 26000 32 random &> ./logs/eval_gqa_rices_every_nth_shot_4_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_gqa.sh 4 26001 32 no_images &> ./logs/eval_gqa_rices_every_nth_shot_4_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_gqa.sh 8 26002 32 random &> ./logs/eval_gqa_rices_every_nth_shot_8_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_gqa.sh 8 26003 32 no_images &> ./logs/eval_gqa_rices_every_nth_shot_8_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_gqa.sh 16 26004 8 random &> ./logs/eval_gqa_rices_every_nth_shot_16_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_gqa.sh 16 26005 8 no_images &> ./logs/eval_gqa_rices_every_nth_shot_16_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_gqa.sh 32 26006 8 random &> ./logs/eval_gqa_rices_every_nth_shot_32_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_gqa.sh 32 26007 8 no_images &> ./logs/eval_gqa_rices_every_nth_shot_32_visual_no_images.log
##
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_okvqa.sh 4 26000 32 random &> ./logs/eval_okvqa_rices_every_nth_shot_4_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_okvqa.sh 4 26001 32 no_images &> ./logs/eval_okvqa_rices_every_nth_shot_4_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_okvqa.sh 8 26002 32 random &> ./logs/eval_okvqa_rices_every_nth_shot_8_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_okvqa.sh 8 26003 32 no_images &> ./logs/eval_okvqa_rices_every_nth_shot_8_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_okvqa.sh 16 26004 8 random &> ./logs/eval_okvqa_rices_every_nth_shot_16_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_okvqa.sh 16 26005 8 no_images &> ./logs/eval_okvqa_rices_every_nth_shot_16_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_okvqa.sh 32 26006 8 random &> ./logs/eval_okvqa_rices_every_nth_shot_32_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_okvqa.sh 32 26007 8 no_images &> ./logs/eval_okvqa_rices_every_nth_shot_32_visual_no_images.log
#
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_textvqa.sh 4 26000 32 random &> ./logs/eval_textvqa_rices_every_nth_shot_4_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_textvqa.sh 4 26001 32 no_images &> ./logs/eval_textvqa_rices_every_nth_shot_4_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_textvqa.sh 8 26002 32 random &> ./logs/eval_textvqa_rices_every_nth_shot_8_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_textvqa.sh 8 26003 32 no_images &> ./logs/eval_textvqa_rices_every_nth_shot_8_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_textvqa.sh 16 26004 8 random &> ./logs/eval_textvqa_rices_every_nth_shot_16_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_textvqa.sh 16 26005 8 no_images &> ./logs/eval_textvqa_rices_every_nth_shot_16_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_textvqa.sh 32 26006 8 random &> ./logs/eval_textvqa_rices_every_nth_shot_32_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_textvqa.sh 32 26007 8 no_images &> ./logs/eval_textvqa_rices_every_nth_shot_32_visual_no_images.log
#
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vizwiz.sh 4 26000 32 random &> ./logs/eval_vizwiz_rices_every_nth_shot_4_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vizwiz.sh 4 26001 32 no_images &> ./logs/eval_vizwiz_rices_every_nth_shot_4_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vizwiz.sh 8 26002 32 random &> ./logs/eval_vizwiz_rices_every_nth_shot_8_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vizwiz.sh 8 26003 32 no_images &> ./logs/eval_vizwiz_rices_every_nth_shot_8_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vizwiz.sh 16 26004 8 random &> ./logs/eval_vizwiz_rices_every_nth_shot_16_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vizwiz.sh 16 26005 8 no_images &> ./logs/eval_vizwiz_rices_every_nth_shot_16_visual_no_images.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vizwiz.sh 32 26006 8 random &> ./logs/eval_vizwiz_rices_every_nth_shot_32_visual_random.log
#bash open_flamingo/scripts/rice_img/every_nth/run_eval_9B_vizwiz.sh 32 26007 8 no_images &> ./logs/eval_vizwiz_rices_every_nth_shot_32_visual_no_images.log