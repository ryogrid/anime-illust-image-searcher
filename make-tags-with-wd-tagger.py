import os
import numpy as np
import onnxruntime as rt
from huggingface_hub import hf_hub_download
from PIL import Image
import pandas as pd
import argparse
import traceback, sys
import re

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]

VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"
MODEL_FILE_NAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# extensions of image files to be processed
EXTENSIONS = ['.png', '.jpg', '.jpeg', ".PNG", ".JPG", ".JPEG"]

def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
    for Multi-label Classification. In 11th International Symposium, IDA 2012
    (pp. 172-183).
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh

def load_labels(dataframe) -> list[str]:
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

def print_traceback():
    tb = traceback.extract_tb(sys.exc_info()[2])
    trace = traceback.format_list(tb)
    print('---- traceback ----')
    for line in trace:
        if '~^~' in line:
            print(line.rstrip())
        else:
            text = re.sub(r'\n\s*', ' ', line.rstrip())
            print(text)
    print('-------------------')

# list up files and filter by extension
def list_files_recursive(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if any(file_path.endswith(ext) for ext in EXTENSIONS):
                file_list.append(file_path)
    return file_list

class Predictor:
    def __init__(self):
        self.model_target_size = None
        self.last_loaded_repo = None
        self.tagger_model_path = None
        self.tagger_model = None
        self.tag_names = None
        self.rating_indexes = None
        self.general_indexes = None
        self.character_indexes = None

    def prepare_image(self, image):
        target_size = self.model_target_size

        if image.mode in ('RGBA', 'LA'):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        else:
            image = image.convert("RGB")

        # Pad image to square        
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def load_model(self):
        if self.tagger_model is not None:
            return
        
        self.tagger_model_path = hf_hub_download(repo_id=VIT_MODEL_DSV3_REPO, filename=MODEL_FILE_NAME)

        self.tagger_model = rt.InferenceSession(self.tagger_model_path, providers=['CPUExecutionProvider'])
        _, height, _, _ = self.tagger_model.get_inputs()[0].shape
        
        self.model_target_size = height

        csv_path = hf_hub_download(
            VIT_MODEL_DSV3_REPO,
            LABEL_FILENAME,
        )
        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

    def predict(
        self,
        image,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    ):
        image = self.prepare_image(image)

        input_name = self.tagger_model.get_inputs()[0].name
        label_name = self.tagger_model.get_outputs()[0].name
        preds = self.tagger_model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # # First 4 labels are actually ratings: pick one with argmax
        # ratings_names = [labels[i] for i in self.rating_indexes]
        # rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = [x.replace(' ', '_') for x in sorted_general_strings]
        sorted_general_strings = (
            ",".join(sorted_general_strings).replace("(", "\(").replace(")", "\)")
        )

        ret_string = sorted_general_strings

        if len(character_res) > 0:
            sorted_character_strings = sorted(
                character_res.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            sorted_character_strings = [x[0] for x in sorted_character_strings]
            sorted_character_strings = [x.replace(' ', '_') for x in sorted_character_strings]
            sorted_character_strings = (
                ",".join(sorted_character_strings).replace("(", "\(").replace(")", "\)")
            )
            ret_string += "," + sorted_character_strings

        return ret_string

    # write a line to file
    def write_to_file(self, csv_line):    
        self.f.write(csv_line + '\n')
        self.f.flush()

    # root function
    def process_directory(self, directory):
        file_list = list_files_recursive(directory)
        print(f'{len(file_list)} files found')

        # file for tagged results
        self.f = open('tags-wd-tagger.txt', 'a', encoding='utf-8')

        self.load_model()

        cnt = 0
        # process each image file
        for file_path in file_list:
            try:
                img = Image.open(file_path)
                results_in_csv_format = self.predict(img, 0.3, True, 0.3, True)
                
                # write result to file
                self.write_to_file(file_path + ',' + results_in_csv_format)

                if cnt % 100 == 0:
                    print(f'{cnt} files processed')

                cnt += 1
            except Exception as e:
                error_class = type(e)
                error_description = str(e)
                err_msg = '%s: %s' % (error_class, error_description)
                print(err_msg)
                print_traceback()
                pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', nargs=1, required=True, help='tagging target directory path')
    args = parser.parse_args()

    predictor = Predictor()
    predictor.process_directory(args.dir[0])

if __name__ == "__main__":
    main()