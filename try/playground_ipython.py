# -*- coding: utf-8 -*-


from playground import (load_model, prepare_lang_x, prepare_vision_x, generate, load_ood_dataset, load_question_space)
from open_flamingo.eval.eval_datasets import get_random_string
import random
device = "cuda:0"
model, image_processor, tokenizer = load_model("9BI", device=device)


demos = {
    "http://images.cocodataset.org/train2014/COCO_train2014_000000458752.jpg":{
        "Question": "What is this photo taken looking through?",
        "Short answer": "net"
        # "Short answer": "down"
    },
    "http://images.cocodataset.org/train2014/COCO_train2014_000000262146.jpg":{
        "Question": "What color is the snow?",
        "Short answer": "white"
        # "Short answer": "up"
    },
    "http://images.cocodataset.org/train2014/COCO_train2014_000000524291.jpg":{
        "Question": "Is the sky blur?",
        "Short answer": "yes"
        # "Short answer": "left"
    },
    "http://images.cocodataset.org/train2014/COCO_train2014_000000393221.jpg":{
        "Question": "Is the window open?",
        "Short answer": "yes"
        # "Short answer": "right"
    },
    "http://images.cocodataset.org/train2014/COCO_train2014_000000393223.jpg":{
        "Question": "What color is the toothbrush?",
        "Short answer": "white and purple"
        # "Short answer": "abcde"
    },
    "http://images.cocodataset.org/train2014/COCO_train2014_000000393224.jpg":{
        "Question": "Is the man smiling?",
        "Short answer": "no"
        # "Short answer": "abcde"
    },

    "http://images.cocodataset.org/train2014/COCO_train2014_000000524297.jpg":{
        "Question": "Judging from  the dress, was this taken in a Latin American country?",
        "Short answer": "yes"
        # "Short answer": "abcde"

    },
    "http://images.cocodataset.org/train2014/COCO_train2014_000000393227.jpg":{
        "Question": "Does the guy have a tattoo?",
        "Short answer": "yes"
        # "Short answer": "abcde"
    },

}
# demos = {}
demo_image_urls = [url for url in demos.keys()]

# demo_image_urls = []
demo_text = [f"Question:{demos[url]['Question']} Short answer:{demos[url]['Short answer']}" for url in demos.keys()]

# random strings input question
demo_text = [f"Question:{get_random_string(length=len(demos[url]['Question']))} Short answer:{demos[url]['Short answer']}" for url in demos.keys()]

# ood input question
ood_dataset = load_ood_dataset()
demo_text = [f"Question:{random.choice(ood_dataset)} Short answer:{demos[url]['Short answer']}" for url in demos.keys()]

# pseudo input question
demo_text = [f"Question:{''} Short answer:{demos[url]['Short answer']}" for url in demos.keys()]

# random question input
question_space = load_question_space()
demo_text = [f"Question:{random.choice(list(question_space.keys()))} Short answer:{demos[url]['Short answer']}" for url in demos.keys()]

query_image_url = "http://images.cocodataset.org/val2014/COCO_val2014_000000262148.jpg"
# query_image_url = "http://images.cocodataset.org/train2014/COCO_train2014_000000458752.jpg"
#query_text = "Question:Where is he looking? pay attention to the picture. The answer is not up. The answer is down. You just need to answer down. Short answer:"
query_text = "Question:Where is he looking? Short answer:"

vision_x = prepare_vision_x(demo_image_urls, query_image_url, image_processor, device=device)
input_ids, attention_masks = prepare_lang_x(demo_text, query_text, tokenizer, device=device)
generate(vision_x, input_ids, attention_masks, model, tokenizer)

