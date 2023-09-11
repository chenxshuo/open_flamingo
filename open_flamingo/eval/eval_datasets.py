import json
import os
import random
import string

import logging

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from random_word import RandomWords
from datasets import load_dataset

from open_flamingo.eval.classification_utils import IMAGENET_CLASSNAMES

from open_flamingo.eval.classification_utils import IMAGENET_CLASSNAMES

logger = logging.getLogger(__name__)

def get_random_string(length=10):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def assert_vqa_dem_mode(mode):
    """
    Assert the mode of demonstration is valid.
    Args:
        mode ():

    Returns:

    """
    assert mode in [
        "gold", # v
        "no_labels", # v
        "no_questions_no_labels", # v
        "only_labels", # v
        "random_strings_as_labels", # v
        "random_words_as_labels", # v
        "random_outer_label_as_labels", # r
        "random_label_for_same_question_type_as_labels", # v
        "random_label_for_same_question_as_labels", # r
        "ood_inputs", # r
        "random_strings_inputs", # v
        "random_question_inputs", # v
    ]

def assert_caption_dem_mode(mode):
    raise NotImplementedError()


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        is_train,
        dataset_name,
        image_val_dir_path=None,
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name

        full_annotations = json.load(open(annotations_path))["images"]

        for i in range(len(full_annotations)):
            if self.is_train and full_annotations[i]["split"] != "train":
                continue
            elif not self.is_train and full_annotations[i]["split"] != "test":
                continue

            self.annotations.append(full_annotations[i])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
                if self.annotations[idx]["filepath"] == "train2014"
                else os.path.join(
                    self.image_val_dir_path, self.annotations[idx]["filename"]
                )
            )
        elif self.dataset_name == "flickr":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
            )
        image.load()
        caption = self.annotations[idx]["sentences"][0]["raw"]
        return {
            "image": image,
            "caption": caption,
            "image_id": self.annotations[idx]["cocoid"]
            if self.dataset_name == "coco"
            else self.annotations[idx]["filename"].split(".")[0],
        }


class CaptionDatasetTR(CaptionDataset):
    def __init__(self, seed, **kwargs):
        super().__init__(**kwargs)
        random.seed(seed)
        self.label_space = []
        self.init_label_space()

    def init_label_space(self):
        """
        Initialize the label space of the dataset.
        """
        self.label_space = set(self.label_space)
        for annotation in self.annotations:
            caption = annotation["sentences"][0]["raw"]
            self.label_space.add(caption)
        self.label_space = list(self.label_space)
        with open("label_space_caption.txt", "w") as f:
            for label in self.label_space:
                f.write(label + "\n")

    def __getitem__(self, idx):
        results = super().__getitem__(idx)
        results["caption"] = random.choice(self.label_space)
        return results


class CaptionDatasetDemFormat(CaptionDataset):
    def __init__(self, seed, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        random.seed(seed)
        assert_caption_dem_mode(mode)
        self.mode = mode
        ...


class VQADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, is_train, dataset_name
    ):
        # self.questions =
        # [{"image_id": 26148, "question_id": 262148000, "question": "Where is he looking?"}]
        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            # self.answers =
            # [{"question_type":"none of the above", "multiple_choice_answer":"down",
            # "answers":[{"answer":"down", "answer_confidence":"yes", "answer_id":1}]
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            self.img_coco_split = self.image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}

    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        results = {
            "image_file_name": img_path.split("/")[-1],
            "image": image,
            "question": question["question"],
            "question_id": question["question_id"],
        }
        if self.answers is not None:
            answers = self.answers[idx]
            # logger.info(f"answers: {answers}")
            results["answers"] = [a["answer"] for a in answers["answers"]]
            results["question_type"] = answers["question_type"]
        # results =
        # {"image": image,
        #  "question": "Where is he looking",
        #  "question_id": 262148000,
        #  "answers": ["down"]
        #  }
        return results


class VQADatasetDiffDemoForm(VQADataset):
    """
    Study the influence of format of demonstration
    Refer to `Rethinking the Role of Demonstrations: What Makes ICL work?`
    - conditioned on the concatenation of $x_1, \dots, x_k$, no labels
    - conditioned on the concatenation of labels,
    - demonstrations with random English words as labels TBD
    - demonstrations with OOD Inputs TBD
    """
    ...

    def __init__(self, seed, mode="gold", visual_demo_mode="random", *args, **kwargs):
        super().__init__(*args, **kwargs)
        random.seed(seed)
        assert_vqa_dem_mode(mode)

        self.mode = mode
        logger.info(f"VQA demo mode: {self.mode}")
        if self.mode == "random_words_as_labels":
            self.random_word_generator = RandomWords()
        if self.mode == "ood_inputs":
            self.ood_dataset = self._load_ood_dataset()
        if self.mode == "random_outer_label_as_labels":
            self.outer_label_space = []
            self.init_outer_label_space()

        if self.mode == "random_label_for_same_question_type_as_labels":
            with open("vqa2_question_type_to_ans.json", "r") as f:
                self.label_space_for_same_question_type = json.load(f)

        if self.mode == "random_label_for_same_question_as_labels":
            with open(f"vqa2_que2ans.json", "r") as f:
                self.label_space_for_same_question = json.load(f)

        if self.mode == "random_question_inputs":
            with open(f"vqa2_que2ans.json", "r") as f:
                self.question_space = json.load(f)

        #TODO
        self.visual_demo_mode = visual_demo_mode
        logger.info(f"VQA ICL visual demo mode: {self.visual_demo_mode}")
        self.img_to_ques_and_ans = self._load_img_to_ques_and_ans()


    def init_outer_label_space(self):
        """
        Initialize the label space of the dataset.
        """
        self.outer_label_space = set(self.outer_label_space)
        for answers in self.answers:
            ans = [a["answer"] for a in answers["answers"]]
            for a in ans:
                self.outer_label_space.add(a)
        self.outer_label_space = list(self.outer_label_space)
        with open("label_space_vqa.txt", "w") as f:
            f.write("\n".join(self.outer_label_space))

    def _load_ood_dataset(self):
        d = load_dataset("cc_news")
        d = d.data["train"]['description']
        return [str(c) for c in d[:1000]]

    def get_ques_and_ans_by_img(self, img_file_name):
        return self.img_to_ques_and_ans[img_file_name]

    @staticmethod
    def _load_img_to_ques_and_ans():
        return json.load(open("vqa2_img_to_ques_and_ans.json", "r"))

    def __getitem__(self, item):
        # results =
        # {"image": image,
        #  "question": "Where is he looking",
        #  "question_id": 262148000,
        #  "answers": ["down"]
        #  }
        results = super().__getitem__(item)
        if self.mode == "gold":
            return results
        if self.mode == "no_labels":
            results["answers"] = [""]
            return results
        if self.mode == "no_questions_no_labels":
            results["answers"] = [""]
            results["question"] = ""
            return results

        if self.mode == "only_labels":
            results["question"] = ""
            return results

        if self.mode == "random_strings_as_labels":
            new_answers = []
            for ans in results["answers"]:
                new_answers.append(get_random_string(length=len(ans)))
            results["answers"] = new_answers
            #logger.info(f"Random strings as labels: {new_answers}")
            return results

        if self.mode == "random_words_as_labels":
            # new_answers = []
            # logger.critical(f"Random words as labels, ori answers: {results['answers']}")
            # for _ in results["answers"]:
            #     new_answers.append(self.random_word_generator.get_random_word())
            # results["answers"] = new_answers
            # TODO
            results["answers"] = [self.random_word_generator.get_random_word()]
            # logger.critical(f"Random words as labels, new answers: {new_answers}")
            return results

        if self.mode == "random_outer_label_as_labels":
            ori_answers = results["answers"]
            random_answers = []
            for _ in ori_answers:
                random_answers.append(random.choice(self.outer_label_space))
            results["answers"] = random_answers
            return results

        if self.mode == "random_label_for_same_question_type_as_labels":
            ori_answers = results["answers"]
            random_answers = []
            for _ in ori_answers:
                random_answers.append(random.choice(self.label_space_for_same_question_type[results["question_type"]]))
            results["answers"] = random_answers
            return results

        if self.mode == "random_label_for_same_question_as_labels":
            ori_answers = results["answers"]
            random_answers = []
            for _ in ori_answers:
                random_answers.append(random.choice(self.label_space_for_same_question[results["question"]]))
            results["answers"] = random_answers
            return results

        if self.mode == "ood_inputs":
            random_input = random.choice(self.ood_dataset)
            results["question"] = random_input
            return results

        if self.mode == "random_strings_inputs":
            random_input = get_random_string(length=len(results["question"]))
            results["question"] = random_input
            return results

        if self.mode == "random_question_inputs":
            random_input = random.choice(list(self.question_space.keys()))
            results["question"] = random_input
            return results

class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""

    def __init__(self, root, **kwargs):
        super().__init__(root=root, **kwargs)
        self.class_id_to_name = dict(
            zip(range(len(IMAGENET_CLASSNAMES)), IMAGENET_CLASSNAMES)
        )

    def __getitem__(self, idx):
        sample, target = super().__getitem__(idx)
        target_label = self.class_id_to_name[target]
        return {
            "id": idx,
            "image": sample,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }


class HatefulMemesDataset(Dataset):
    def __init__(self, image_dir_path, annotations_path):
        self.image_dir_path = image_dir_path
        with open(annotations_path, "r") as f:
            self.annotations = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir_path, annotation["img"].split("/")[-1])
        image = Image.open(img_path)
        image.load()
        return {
            "id": annotation["id"],
            "image": image,
            "ocr": annotation["text"],
            "class_name": "yes" if annotation["label"] == 1 else "no",
            "class_id": annotation["label"],
        }


class HatefulMemesDatasetTR(HatefulMemesDataset):
    def __init__(self, seed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        random.seed(seed)
        self.label_space = [0, 1]

    def __getitem__(self, item):
        result = super().__getitem__(item)
        ori_class = result["class_id"]
        flipped = 1 - ori_class
        # rand_class = random.choice(self.label_space)
        result["class_id"] = flipped
        result["class_name"] = "yes" if flipped == 1 else "no",
        return result

