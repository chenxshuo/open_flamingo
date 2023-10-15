# -*- coding: utf-8 -*-

"""TODO."""

from playground import (load_model, prepare_lang_x, prepare_vision_x, generate, load_ood_dataset, load_question_space)
device = "cuda:0"
model, image_processor, tokenizer = load_model("9BI", device=device)
base_train_url = "http://images.cocodataset.org/train2014/"
test_img_url = "http://images.cocodataset.org/val2014/"

# if __name__ == "__main__":

# test_question = "What brand uses these animals as advertising?"
# test_image = test_img_url + "COCO_val2014_000000318671.jpg"


# stored demos during exps
# demo_extracted = [
#             "COCO_train2014_000000426408.jpg <image>Question:What century is this? Short answer:19th<|endofchunk|>\n",
#             "COCO_train2014_000000423832.jpg <image>Question:Why is she eating carbohydrates? Short answer:energy<|endofchunk|>\n",
#             "COCO_train2014_000000019967.jpg <image>Question:How much calorie can be got in the food they are holding? Short answer:300<|endofchunk|>\n",
#             "COCO_train2014_000000215288.jpg <image>Question:How would the water taste that the man is riding on? Short answer:salty<|endofchunk|>\n",
#             "COCO_train2014_000000344146.jpg <image>Question:What game are they playing? Short answer:video<|endofchunk|>\n",
#             "COCO_train2014_000000082676.jpg <image>Question:What year was this model of motorcycle introduced? Short answer:1970<|endofchunk|>\n",
#             "COCO_train2014_000000100516.jpg <image>Question:Who is the sponsor? Short answer:polo<|endofchunk|>\n",
#             "COCO_train2014_000000134871.jpg <image>Question:How many feet deep are these most likely dug? Short answer:6<|endofchunk|>\n",
#             "COCO_train2014_000000556021.jpg <image>Question:Is this a hallway or living room? Short answer:hallway<|endofchunk|>\n",
#             "COCO_train2014_000000553668.jpg <image>Question:What might the object the girl is holding be made of? Short answer:foam<|endofchunk|>\n",
#             "COCO_train2014_000000192513.jpg <image>Question:How do we know this is a professional game? Short answer:crowd<|endofchunk|>\n",
#             "COCO_train2014_000000571541.jpg <image>Question:What is the name of show in which these antique vehicles are participating? Short answer:car show<|endofchunk|>\n",
#             "COCO_train2014_000000478032.jpg <image>Question:What type of truck do they call the large boxed truck in the photo? Short answer:semi<|endofchunk|>\n",
#             "COCO_train2014_000000005430.jpg <image>Question:Who is credited with inventing this item? Short answer:steve job<|endofchunk|>\n",
#             "COCO_train2014_000000109095.jpg <image>Question:What is the weather in this photo like? Short answer:snowy<|endofchunk|>\n",
#             "COCO_train2014_000000579725.jpg <image>Question:What kind of flower is this? Short answer:rose<|endofchunk|>\n"
# ]

# demo_extracted = [
            # "COCO_train2014_000000485788.jpg <image>Question:What type of flower is in the arrangement? Short answer:carnation<|endofchunk|>\n",
            # "COCO_train2014_000000063893.jpg <image>Question:What is the largest mammal in this pictures shoes made of? Short answer:iron<|endofchunk|>\n",
            # "COCO_train2014_000000155061.jpg <image>Question:What roles would this train serve? Short answer:transportation<|endofchunk|>\n",
            # "COCO_train2014_000000223454.jpg <image>Question:What is being pulled in this photo? Short answer:carriage<|endofchunk|>\n",
            # "COCO_train2014_000000142299.jpg <image>Question:How is this vehicle powered? Short answer:horse<|endofchunk|>\n",
            # "COCO_train2014_000000011968.jpg <image>Question:Name breed of horse? Short answer:stallion<|endofchunk|>\n",
            # "COCO_train2014_000000002444.jpg <image>Question:What beer maker is famous for the use of the animals in the image? Short answer:budweiser<|endofchunk|>\n",
            # "COCO_train2014_000000242145.jpg <image>Question:What do these tools help with? Short answer:carry<|endofchunk|>\n",
            # "COCO_train2014_000000134958.jpg <image>Question:What breed of horse is shown? Short answer:clydesdale<|endofchunk|>\n",
            # "COCO_train2014_000000099536.jpg <image>Question:What is the horse pulling? Short answer:carriage<|endofchunk|>\n",
            # "COCO_train2014_000000042371.jpg <image>Question:When was that red spoked implement invented? Short answer:4000 bc<|endofchunk|>\n",
            # "COCO_train2014_000000296439.jpg <image>Question:A castle like the one in the background can be viewed at the start of movies from what studio? Short answer:disney<|endofchunk|>\n",
            # "COCO_train2014_000000312958.jpg <image>Question:Is the horse pulling people or resting? Short answer:rest<|endofchunk|>\n",
            # "COCO_train2014_000000304693.jpg <image>Question:What kind of license do you need to drive on of these? Short answer:driver<|endofchunk|>\n",
            # "COCO_train2014_000000288383.jpg <image>Question:Who is the namesake of this popular theme park? Short answer:disney<|endofchunk|>\n",
            # "COCO_train2014_000000160325.jpg <image>Question:Which famous characters might you see in this place? Short answer:mickey mouse<|endofchunk|>\n"
        # ]

test_question = "Is the zebra in it's natural habitat?"
test_image = test_img_url + "COCO_val2014_000000075162.jpg"

demo_extracted = [
            "COCO_train2014_000000049987.jpg <image>Question:Where is the zebra looking? Short answer:ground<|endofchunk|>\n",
            "COCO_train2014_000000049987.jpg <image>Question:Is there a tree in the photo? Short answer:no<|endofchunk|>\n",
            "COCO_train2014_000000049987.jpg <image>Question:What is the wall made of? Short answer:stone<|endofchunk|>\n",
            "COCO_train2014_000000049987.jpg <image>Question:What is the zebra doing? Short answer:grazing<|endofchunk|>\n",
            "COCO_train2014_000000049987.jpg <image>Question:Is the animal in the shade? Short answer:no<|endofchunk|>\n",
            "COCO_train2014_000000229107.jpg <image>Question:How many logs? Short answer:6<|endofchunk|>\n",
            "COCO_train2014_000000229107.jpg <image>Question:Is this a young or a mature zebra? Short answer:young<|endofchunk|>\n",
            "COCO_train2014_000000229107.jpg <image>Question:Is this animal free or in captivity? Short answer:in captivity<|endofchunk|>\n"
        ]


# visual_mode = "gold"
visual_mode = "no_images"

if visual_mode == "no_images":
    demo_image_urls = []
    demo_text = [t.split("<image>")[1] for t in demo_extracted]
    query_text = f"Question:{test_question} Short answer:"
else:
    demo_image_urls = [base_train_url + d.split(" ")[0] for d in demo_extracted]
    demo_text = ["<image>" + t.split("<image>")[1] for t in demo_extracted]
    query_text = f"Question:{test_question} Short answer:"


vision_x = prepare_vision_x(demo_image_urls, test_image, image_processor, device=device)
input_ids, attention_masks = prepare_lang_x(demo_text, query_text, tokenizer, device=device, visual_mode=visual_mode)
generate(vision_x, input_ids, attention_masks, model, tokenizer)


