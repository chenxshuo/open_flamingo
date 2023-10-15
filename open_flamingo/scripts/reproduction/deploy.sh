# nohup bash open_flamingo/scripts/reproduction/deploy.sh > out-reproduce.text 2>&1 &

# shots = 0, 4, 8, 16; port = 26000, 26001, 26002, 26003; batch_size = 64

#bash open_flamingo/scripts/reproduction/run_eval_9B_vqav2.sh 0 26000 64 &> out-reproduce-9B-vqav2-shot-0.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_vqav2.sh 4 26001 64 &> out-reproduce-9B-vqav2-shot-4.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_vqav2.sh 8 26002 32 &> out-reproduce-9B-vqav2-shot-8.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_vqav2.sh 16 26003 16 &> out-reproduce-9B-vqav2-shot-16.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_vqav2.sh 32 26004 8 &> out-reproduce-9B-vqav2-shot-32.text
#
#bash open_flamingo/scripts/reproduction/run_eval_9B_okvqa.sh 0 26000 64 &> out-reproduce-9B-okvqa-shot-0.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_okvqa.sh 4 26001 64 &> out-reproduce-9B-okvqa-shot-4.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_okvqa.sh 8 26002 64 &> out-reproduce-9B-okvqa-shot-8.text
bash open_flamingo/scripts/reproduction/run_eval_9B_okvqa.sh 16 26003 32 &> out-reproduce-9B-okvqa-shot-16.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_okvqa.sh 32 26003 32 &> out-reproduce-9B-okvqa-shot-32.text
#
#
##bash open_flamingo/scripts/reproduction/run_eval_3B_textvqa.sh 4 26001 64

#bash open_flamingo/scripts/reproduction/run_eval_9B_gqa.sh 0 26001 64 &> out-reproduce-9B-gqa-shot-0.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_gqa.sh 4 26001 64 &> out-reproduce-9B-gqa-shot-4.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_gqa.sh 8 26002 64 &> out-reproduce-9B-gqa-shot-8.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_gqa.sh 16 26003 32 &> out-reproduce-9B-gqa-shot-16.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_gqa.sh 32 26003 16 &> out-reproduce-9B-gqa-shot-32.text

#bash open_flamingo/scripts/reproduction/run_eval_9B_textvqa.sh 0 26000 64 &> out-reproduce-9B-textvqa-shot-0.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_textvqa.sh 4 26001 64 &> out-reproduce-9B-textvqa-shot-4.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_textvqa.sh 8 26002 64 &> out-reproduce-9B-textvqa-shot-8.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_textvqa.sh 16 26003 32 &> out-reproduce-9B-textvqa-shot-16.text
#echo "Done with textvqa shot 16"
#bash open_flamingo/scripts/reproduction/run_eval_9B_textvqa.sh 32 26003 16 &> out-reproduce-9B-textvqa-shot-32.text
#echo "Done with textvqa shot 32"


#bash open_flamingo/scripts/reproduction/run_eval_9B_vizwiz.sh 0 26000 64 &> out-reproduce-9B-vizwiz-shot-0.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_vizwiz.sh 4 26001 64 &> out-reproduce-9B-vizwiz-shot-4.text
#echo "Done with vizwiz shot 4"
#bash open_flamingo/scripts/reproduction/run_eval_9B_vizwiz.sh 8 26002 64 &> out-reproduce-9B-vizwiz-shot-8.text
#echo "Done with vizwiz shot 8"
#bash open_flamingo/scripts/reproduction/run_eval_9B_vizwiz.sh 16 26003 32 &> out-reproduce-9B-vizwiz-shot-16.text
#echo "Done with vizwiz shot 16"
#bash open_flamingo/scripts/reproduction/run_eval_9B_vizwiz.sh 32 26004 16 &> out-reproduce-9B-vizwiz-shot-32.text
#echo "Done with vizwiz shot 32"

#bash open_flamingo/scripts/reproduction/run_eval_9B_flickr30k.sh 0 26000 64 &> out-reproduce-9B-flickr30k-shot-0.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_flickr30k.sh 4 26001 64 &> out-reproduce-9B-flickr30k-shot-4.text
#echo "Done with flickr30k shot 4"
#bash open_flamingo/scripts/reproduction/run_eval_9B_flickr30k.sh 8 26002 64 &> out-reproduce-9B-flickr30k-shot-8.text
#echo "Done with flickr30k shot 8"
#bash open_flamingo/scripts/reproduction/run_eval_9B_flickr30k.sh 16 26003 32 &> out-reproduce-9B-flickr30k-shot-16.text
#echo "Done with flickr30k shot 16"
bash open_flamingo/scripts/reproduction/run_eval_9B_flickr30k.sh 32 26004 16 &> out-reproduce-9B-flickr30k-shot-32.text
#echo "Done with flickr30k shot 16"



#bash open_flamingo/scripts/reproduction/run_eval_9B_coco.sh 0 26000 64 &> out-reproduce-9B-coco-shot-0.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_coco.sh 4 26001 64 &> out-reproduce-9B-coco-shot-4.text
#echo "Done with coco shot 4"
#bash open_flamingo/scripts/reproduction/run_eval_9B_coco.sh 8 26002 64 &> out-reproduce-9B-coco-shot-8.text
#echo "Done with coco shot 8"
#bash open_flamingo/scripts/reproduction/run_eval_9B_coco.sh 16 26003 32 &> out-reproduce-9B-coco-shot-16.text
#echo "Done with coco shot 16"
bash open_flamingo/scripts/reproduction/run_eval_9B_coco.sh 32 26004 16 &> out-reproduce-9B-coco-shot-32.text
#echo "Done with coco shot 32"
#
#bash open_flamingo/scripts/reproduction/run_eval_9B_hatefulmemes.sh 0 26000 64 &> out-reproduce-9B-hatefulmemes-shot-0.text
#bash open_flamingo/scripts/reproduction/run_eval_9B_hatefulmemes.sh 4 26001 32 &> out-reproduce-9B-hatefulmemes-shot-4.text
#echo "Done with hatefulmemes shot 4"
#bash open_flamingo/scripts/reproduction/run_eval_9B_hatefulmemes.sh 8 26002 16 &> out-reproduce-9B-hatefulmemes-shot-8.text
#echo "Done with hatefulmemes shot 8"
#bash open_flamingo/scripts/reproduction/run_eval_9B_hatefulmemes.sh 16 26003 16 &> out-reproduce-9B-hatefulmemes-shot-16.text
#echo "Done with hatefulmemes shot 16"
#bash open_flamingo/scripts/reproduction/run_eval_9B_hatefulmemes.sh 32 26004 8 &> out-reproduce-9B-hatefulmemes-shot-32.text
#echo "Done with hatefulmemes shot 32"
