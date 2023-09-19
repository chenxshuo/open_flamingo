# nohup bash open_flamingo/scripts/rice_text/cache_features/deploy.sh &> ./logs/rice_text/deploy.log 2>&1 &

#nohup bash open_flamingo/scripts/rice_text/cache_features/cache_features_coco.sh > ./logs/rice_text/cache_features_coco.log 2>&1 &

#nohup bash open_flamingo/scripts/rice_text/cache_features/cache_features_flickr30k.sh > ./logs/rice_text/cache_features_flickr30k.log 2>&1 &
nohup bash open_flamingo/scripts/rice_text/cache_features/cache_features_gqa.sh > ./logs/rice_text/cache_features_gqa.log 2>&1 &
#nohup bash open_flamingo/scripts/rice_text/cache_features/cache_features_hatefulmemes.sh > ./logs/rice_text/cache_features_hatefulmemes.log 2>&1 &
#nohup bash open_flamingo/scripts/rice_text/cache_features/cache_features_vqav2.sh > ./logs/rice_text/cache_features_vqa.log 2>&1 &
#nohup bash open_flamingo/scripts/rice_text/cache_features/cache_features_vizwiz.sh > ./logs/rice_text/cache_features_vizwiz.log 2>&1 &
#nohup bash open_flamingo/scripts/rice_text/cache_features/cache_features_okvqa.sh > ./logs/rice_text/cache_features_okvqa.log 2>&1 &
nohup bash open_flamingo/scripts/rice_text/cache_features/cache_features_textvqa.sh > ./logs/rice_text/cache_features_textvqa.log 2>&1 &
