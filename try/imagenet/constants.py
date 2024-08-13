# -*- coding: utf-8 -*-

"""TODO."""

import logging

logger = logging.getLogger(__name__)

DATA_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/datasets/imagenet"

NEEDED_32_CLASSES = [
    "ambulance",
    "bell pepper",
    "fox squirrel",
    "basketball",
    "grasshopper",
    "tarantula",
    "centipede",
    "accordion",
    "saxophone",
    "porcupine",
    "hot dog",
    "rugby ball",
    "parachute",
    "baboon",
    "vulture",
    "bow tie",
    "Rottweiler",
    "mantis",
    "lion",
    "cucumber",
    "broccoli",
    "flamingo",
    "jellyfish",
    "ant",
    "broom",
    "bison",
    "mushroom",
    "American egret",
    "school bus",
    "African chameleon",
    "ladybug",
    "volcano",
]

NEEDED_8_CLASSES = NEEDED_32_CLASSES[:8]
NEEDED_16_CLASSES = NEEDED_32_CLASSES[:16]

NEEDED_32_CLASSES_TO_NUMBER = {
    "vulture": "n01616318",
    "African chameleon": "n01694178",
    "tarantula": "n01774750",
    "centipede": "n01784675",
    "jellyfish": "n01910747",
    "flamingo": "n02007558",
    "American egret": "n02009912",
    "Rottweiler": "n02106550",
    "lion": "n02129165",
    "ladybug": "n02165456",
    "ant": "n02219486",
    "grasshopper": "n02226429",
    "mantis": "n02236044",
    "porcupine": "n02346627",
    "fox squirrel": "n02356798",
    "bison": "n02410509",
    "baboon": "n02486410",
    "accordion": "n02672831",
    "ambulance": "n02701002",
    "basketball": "n02802426",
    "bow tie": "n02883205",
    "broom": "n02906734",
    "parachute": "n03888257",
    "rugby ball": "n04118538",
    "saxophone": "n04141076",
    "school bus": "n04146614",
    "hot dog": "n07697537",
    "broccoli": "n07714990",
    "cucumber": "n07718472",
    "bell pepper": "n07720875",
    "mushroom": "n07734744",
    "volcano": "n09472597",
}

NEEDED_32_NUMBER_TO_CLASSES = {
    v:k for k,v in NEEDED_32_CLASSES_TO_NUMBER.items()
}

NEEDED_8_CLASSES_TO_NUMBER = {
    k:v for k,v in NEEDED_32_CLASSES_TO_NUMBER.items() if k in NEEDED_8_CLASSES
}

NEEDED_16_CLASSES_TO_NUMBER = {
    k:v for k,v in NEEDED_32_CLASSES_TO_NUMBER.items() if k in NEEDED_16_CLASSES
}

NOVEL_8_CLASSES = [
    "cheeseburger",
    "candle",
    "monarch",
    "goldfinch",
    "hermit crab",
    "banana",
    "drake",
    "canoe",
]

NOVEL_8_CLASSES_TO_NUMBER = {
    "cheeseburger": "n07697313",
    "candle": "n02948072",
    "monarch": "n02279972",
    "goldfinch": "n01531178",
    "hermit crab": "n01986214",
    "banana": "n07753592",
    "drake": "n01847000",
    "canoe": "n02951358",
}


NUMBER_TO_ORDER_NUMBER = {}
ORDER_NUMBER_TO_NUMBER = {}
NUMBER_TO_NAME_FILE = f"LOC_synset_mapping.txt"
NUMBER_TO_NAME = {}
NEEDED_NAME_TO_NUMBER = {}
with open(NUMBER_TO_NAME_FILE, "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        number = line.split()[0]
        name = " ".join(line.split()[1:])
        name = list(name.split(","))
        name = [n.strip() for n in name]
        # print(name)
        NUMBER_TO_NAME[number] = name
        NUMBER_TO_ORDER_NUMBER[number] = i
        ORDER_NUMBER_TO_NUMBER[i] = number


# print(NUMBER_TO_NAME[list(NUMBER_TO_NAME.keys())[0]])
for number in NUMBER_TO_NAME.keys():
    for name in NEEDED_32_CLASSES:
        if name in NUMBER_TO_NAME[number]:
            NEEDED_NAME_TO_NUMBER[name] = number

# print(NEEDED_NAME_TO_NUMBER)
for k, v in NEEDED_NAME_TO_NUMBER.items():
    # print(
    #     f"class name: {k}, class number: {v}, original class name {NUMBER_TO_NAME[v]}"
    # )
    assert k in NUMBER_TO_NAME[v]

for k, v in NEEDED_32_CLASSES_TO_NUMBER.items():
    # print(
    #     f"class name: {k}, class number: {v}, original class name {NUMBER_TO_NAME[v]}"
    # )
    assert k in NUMBER_TO_NAME[v]

NEEDED_NUMBER_8_CLASSES = [NEEDED_NAME_TO_NUMBER[name] for name in NEEDED_8_CLASSES]
NEEDED_NUMBER_16_CLASSES = [NEEDED_NAME_TO_NUMBER[name] for name in NEEDED_16_CLASSES]
NEEDED_NUMBER_32_CLASSES = [NEEDED_NAME_TO_NUMBER[name] for name in NEEDED_32_CLASSES]

NEEDED_ORDER_NUMBER_8_CLASSES = [NUMBER_TO_ORDER_NUMBER[number] for number in NEEDED_NUMBER_8_CLASSES]
NEEDED_ORDER_NUMBER_16_CLASSES = [NUMBER_TO_ORDER_NUMBER[number] for number in NEEDED_NUMBER_16_CLASSES]
NEEDED_ORDER_NUMBER_32_CLASSES = [NUMBER_TO_ORDER_NUMBER[number] for number in NEEDED_NUMBER_32_CLASSES]

NEEDED_ORDER_NUMBER_NOVEL_8_CLASSES = [NUMBER_TO_ORDER_NUMBER[NOVEL_8_CLASSES_TO_NUMBER[name]] for name in NOVEL_8_CLASSES]