# nohup bash open_flamingo/scripts/rice_only_text/deploy.sh > ./logs/rice_only_text/deploys.log 2>&1 &
# RICE to select img demos and remove imgs

#bash open_flamingo/scripts/rice_only_text/run_eval_9B_coco.sh 4 26000 32 &> ./logs/rice_only_text/out-rice-img-9B-coco-shot-4.text
bash open_flamingo/scripts/rice_only_text/run_eval_9B_coco.sh 8 26000 16 &> ./logs/rice_only_text/out-rice-img-9B-coco-shot-8.text
bash open_flamingo/scripts/rice_only_text/run_eval_9B_coco.sh 16 26000 8 &> ./logs/rice_only_text/out-rice-img-9B-coco-shot-16.text
bash open_flamingo/scripts/rice_only_text/run_eval_9B_coco.sh 32 26000 8 &> ./logs/rice_only_text/out-rice-img-9B-coco-shot-32.text

bash open_flamingo/scripts/rice_only_text/run_eval_9B_flickr30k.sh 4 26000 32 &> ./logs/rice_only_text/out-rice-img-9B-flickr30k-shot-4.text
bash open_flamingo/scripts/rice_only_text/run_eval_9B_flickr30k.sh 8 26000 16 &> ./logs/rice_only_text/out-rice-img-9B-flickr30k-shot-8.text
bash open_flamingo/scripts/rice_only_text/run_eval_9B_flickr30k.sh 16 26000 8 &> ./logs/rice_only_text/out-rice-img-9B-flickr30k-shot-16.text
bash open_flamingo/scripts/rice_only_text/run_eval_9B_flickr30k.sh 32 26000 8 &> ./logs/rice_only_text/out-rice-img-9B-flickr30k-shot-32.text
