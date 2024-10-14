import os, time
import numpy as np
import onnxruntime as rt
from huggingface_hub import hf_hub_download
from PIL import Image
import pandas as pd
import argparse
import traceback, sys
import re
from typing import List, Tuple, Dict, Any, Optional, Callable, Protocol

kaomojis: List[str] = [
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

VIT_MODEL_DSV3_REPO: str = "SmilingWolf/wd-vit-tagger-v3"
MODEL_FILE_NAME: str = "model.onnx"
LABEL_FILENAME: str = "selected_tags.csv"

EXTENSIONS: List[str] = ['.png', '.jpg', '.jpeg', ".PNG", ".JPG", ".JPEG"]

def mcut_threshold(probs: np.ndarray) -> float:
    sorted_probs: np.ndarray = probs[probs.argsort()[::-1]]
    difs: np.ndarray = sorted_probs[:-1] - sorted_probs[1:]
    t: int = difs.argmax()
    thresh: float = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh

def load_labels(dataframe: pd.DataFrame) -> Tuple[List[str], List[int], List[int], List[int]]:
    name_series: pd.Series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names: List[str] = name_series.tolist()

    rating_indexes: List[int] = list(np.where(dataframe["category"] == 9)[0])
    general_indexes: List[int] = list(np.where(dataframe["category"] == 0)[0])
    character_indexes: List[int] = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

def print_traceback() -> None:
    tb: traceback.StackSummary = traceback.extract_tb(sys.exc_info()[2])
    trace: List[str] = traceback.format_list(tb)
    print('---- traceback ----')
    for line in trace:
        if '~^~' in line:
            print(line.rstrip())
        else:
            text: str = re.sub(r'\n\s*', ' ', line.rstrip())
            print(text)
    print('-------------------')

def list_files_recursive(directory: str) -> List[str]:
    file_list: List[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path: str = os.path.join(root, file)
            if any(file_path.endswith(ext) for ext in EXTENSIONS):
                file_list.append(file_path)
    return file_list

class Predictor:
    def __init__(self) -> None:
        self.model_target_size: Optional[int] = None
        self.last_loaded_repo: Optional[str] = None
        self.tagger_model_path: Optional[str] = None
        self.tagger_model: Optional[rt.InferenceSession] = None
        self.tag_names: Optional[List[str]] = None
        self.rating_indexes: Optional[List[int]] = None
        self.general_indexes: Optional[List[int]] = None
        self.character_indexes: Optional[List[int]] = None

    def prepare_image(self, image: Image.Image) -> np.ndarray:
        target_size: int = self.model_target_size

        if image.mode in ('RGBA', 'LA'):
            background: Image.Image = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        else:
            image = image.convert("RGB")

        image_shape: Tuple[int, int] = image.size
        max_dim: int = max(image_shape)
        pad_left: int = (max_dim - image_shape[0]) // 2
        pad_top: int = (max_dim - image_shape[1]) // 2

        padded_image: Image.Image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        image_array: np.ndarray = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def load_model(self) -> None:
        if self.tagger_model is not None:
            return

        self.tagger_model_path = hf_hub_download(repo_id=VIT_MODEL_DSV3_REPO, filename=MODEL_FILE_NAME)
        self.tagger_model = rt.InferenceSession(self.tagger_model_path, providers=['CPUExecutionProvider'])
        _, height, _, _ = self.tagger_model.get_inputs()[0].shape

        self.model_target_size = height

        csv_path: str = hf_hub_download(
            VIT_MODEL_DSV3_REPO,
            LABEL_FILENAME,
        )
        tags_df: pd.DataFrame = pd.read_csv(csv_path)
        sep_tags: Tuple[List[str], List[int], List[int], List[int]] = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

    def predict(
        self,
        image: Image.Image,
        general_thresh: float,
        general_mcut_enabled: bool,
        character_thresh: float,
        character_mcut_enabled: bool,
    ) -> str:
        image: np.ndarray = self.prepare_image(image)

        input_name: str = self.tagger_model.get_inputs()[0].name
        label_name: str = self.tagger_model.get_outputs()[0].name
        preds: np.ndarray = self.tagger_model.run([label_name], {input_name: image})[0]

        labels: List[Tuple[str, float]] = list(zip(self.tag_names, preds[0].astype(float)))

        general_names: List[Tuple[str, float]] = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs: np.ndarray = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res: Dict[str, float] = {x[0]: x[1] for x in general_names if x[1] > general_thresh}

        character_names: List[Tuple[str, float]] = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs: np.ndarray = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res: Dict[str, float] = {x[0]: x[1] for x in character_names if x[1] > character_thresh}

        sorted_general_strings: List[str] = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = [x.replace(' ', '_') for x in sorted_general_strings]
        sorted_general_strings = (
            ",".join(sorted_general_strings).replace("(", "\(").replace(")", "\)")
        )

        ret_string: str = sorted_general_strings

        if len(character_res) > 0:
            sorted_character_strings: List[str] = sorted(
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

    def write_to_file(self, csv_line: str) -> None:
        self.f.write(csv_line + '\n')
        self.f.flush()

    def process_directory(self, directory: str) -> None:
        file_list: List[str] = list_files_recursive(directory)
        print(f'{len(file_list)} files found')

        self.f = open('tags-wd-tagger.txt', 'a', encoding='utf-8')

        self.load_model()

        start: float = time.perf_counter()
        cnt: int = 0
        for file_path in file_list:
            try:
                img: Image.Image = Image.open(file_path)
                results_in_csv_format: str = self.predict(img, 0.3, True, 0.3, True)

                self.write_to_file(file_path + ',' + results_in_csv_format)

                if cnt % 100 == 0:
                    now: float = time.perf_counter()
                    print(f'{cnt} files processed')
                    diff: float = now - start
                    print('{:.2f} seconds elapsed'.format(diff))
                    if cnt > 0:
                        time_per_file: float = diff / cnt
                        print('{:.4f} seconds per file'.format(time_per_file))
                    print("", flush=True)

                cnt += 1
            except Exception as e:
                error_class: type = type(e)
                error_description: str = str(e)
                err_msg: str = '%s: %s' % (error_class, error_description)
                print(err_msg)
                print_traceback()
                pass

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--dir', nargs=1, required=True, help='tagging target directory path')
    args: argparse.Namespace = parser.parse_args()

    predictor: Predictor = Predictor()
    predictor.process_directory(args.dir[0])

if __name__ == "__main__":
    main()