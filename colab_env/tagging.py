import os, time
import pandas as pd
import argparse
import traceback, sys
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable, Protocol

import numpy as np
from humanfriendly.terminal import output
from numpy import signedinteger
from PIL import Image
import timm
from timm.data import create_transform, resolve_data_config
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

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

TAGGER_VIT_MODEL_REPO: str = "SmilingWolf/wd-eva02-large-tagger-v3"

EXTENSIONS: List[str] = ['.png', '.jpg', '.jpeg', ".PNG", ".JPG", ".JPEG"]

BATCH_SIZE: int = 10
PROGRESS_INTERVAL: int = 100

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for apple silicon
if torch.backends.mps.is_available():
    torch_device = torch.device("mps")

def mcut_threshold(probs: np.ndarray) -> float:
    sorted_probs: np.ndarray = probs[probs.argsort()[::-1]]
    difs: np.ndarray = sorted_probs[:-1] - sorted_probs[1:]
    t: signedinteger[Any] = difs.argmax()
    thresh: float = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh

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


class Predictor:
    def __init__(self) -> None:
        # self.model_target_size: Optional[int] = None
        self.last_loaded_repo: Optional[str] = None
        self.tagger_model: Optional[nn.Module] = None
        self.tag_names: Optional[List[str]] = None
        self.rating_index: Optional[List[int]] = None
        self.general_index: Optional[List[int]] = None
        self.character_index: Optional[List[int]] = None
        self.transform: Optional[Callable] = None

    def list_files_recursive(self, tarfile_path: str) -> List[str]:
        file_list: List[str] = []
        for root, _, files in os.walk(tarfile_path):
            for file in files:
                file_path: str = os.path.join(root, file)
                if any(file_path.endswith(ext) for ext in EXTENSIONS):
                    file_list.append(file_path)
        return file_list

    def prepare_image(self, image: Image.Image) -> Image.Image:
        #target_size: int = self.model_target_size

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

        return padded_image

    def load_labels_hf(
            self,
            repo_id: str,
            revision: Optional[str] = None,
            token: Optional[str] = None,
    ) -> None:
        try:
            csv_path = hf_hub_download(
                repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
            )
            csv_path = Path(csv_path).resolve()
        except HfHubHTTPError as e:
            raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

        df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
        self.rating_index = list(np.where(df["category"] == 9)[0])
        self.general_index = list(np.where(df["category"] == 0)[0])
        self.character_index = list(np.where(df["category"] == 4)[0])
        self.tag_names = df["name"].tolist()

    def load_model(self) -> None:
        if self.tagger_model is not None:
            return

        self.tagger_model = timm.create_model("hf-hub:" + TAGGER_VIT_MODEL_REPO).eval()
        state_dict = timm.models.load_state_dict_from_hf(TAGGER_VIT_MODEL_REPO)
        self.tagger_model.load_state_dict(state_dict)

        print("Loading tag list...")
        self.load_labels_hf(repo_id=TAGGER_VIT_MODEL_REPO)

        print("Creating data transform...")
        self.transform = create_transform(**resolve_data_config(self.tagger_model.pretrained_cfg, model=self.tagger_model))

    def predict(
            self,
            images: List[Image.Image],
            general_thresh: float,
            general_mcut_enabled: bool,
            character_thresh: float,
            character_mcut_enabled: bool,
    ) -> List[str]:
        inputs: List[Tensor] = []
        for img in images:
            img_tmp = self.prepare_image(img)
            # run the model's input transform to convert to tensor and rescale
            # input: Tensor = self.transform(img_tmp).unsqueeze(0)
            input: Tensor = self.transform(img_tmp)
            # NCHW image RGB to BGR
            # input = input[:, [2, 1, 0]]
            input = input[[2, 1, 0]]
            # if inputs is None:
            #     inputs = input
            # else:
            #     inputs = torch.cat((inputs, input), 0)
            inputs.append(input)
        batched_tensor = torch.stack(inputs, dim=0)

        print("Running inference...")
        with torch.inference_mode():
            # move model to GPU, if available
            model = self.tagger_model
            if torch_device.type != "cpu":
                model = self.tagger_model.to(torch_device)
                batched_tensor = batched_tensor.to(torch_device)
            # run the model
            outputs = model.forward(batched_tensor)
            # apply the final activation function (timm doesn't support doing this internally)
            outputs = F.sigmoid(outputs)
            # move inputs, outputs, and model back to to cpu if we were on GPU
            if torch_device.type != "cpu":
                outputs = outputs.to("cpu")

        print("Processing results...")
        preds = outputs.numpy()

        # print(preds)
        # exit(1)
        ret_strings: List[str] = []
        for idx in range(0, len(images)):
            labels: List[Tuple[str, float]] = list(zip(self.tag_names, preds[idx].astype(float)))

            general_names: List[Tuple[str, float]] = [labels[i] for i in self.general_index]

            if general_mcut_enabled:
                general_probs: np.ndarray = np.array([x[1] for x in general_names])
                general_thresh = mcut_threshold(general_probs)

            general_res: Dict[str, float] = {x[0]: x[1] for x in general_names if x[1] > general_thresh}

            character_names: List[Tuple[str, float]] = [labels[i] for i in self.character_index]

            if character_mcut_enabled:
                character_probs: np.ndarray = np.array([x[1] for x in character_names])
                character_thresh = mcut_threshold(character_probs)
                character_thresh = max(0.15, character_thresh)

            character_res: Dict[str, float] = {x[0]: x[1] for x in character_names if x[1] > character_thresh}

            sorted_general_strings: List[Tuple[str, float]] = sorted(
                general_res.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            sorted_general_strings_str: List[str] = [x[0] for x in sorted_general_strings]
            sorted_general_strings_str = [x.replace(' ', '_') for x in sorted_general_strings_str]
            ret_string: str = (
                ",".join(sorted_general_strings_str)
            )

            if len(character_res) > 0:
                sorted_character_strings: List[Tuple[str, float]] = sorted(
                    character_res.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                sorted_character_strings_str: List[str] = [x[0] for x in sorted_character_strings]
                sorted_character_strings_str = [x.replace(' ', '_') for x in sorted_character_strings_str]
                ret_string += ",".join(sorted_character_strings_str)

            ret_strings.append(ret_string)

        return ret_strings

    def write_to_file(self, csv_line: str) -> None:
        self.f.write(csv_line + '\n')
        self.f.flush()

    def process_directory(self, tarfile_path: str) -> None:
        file_list: List[str] = self.list_files_recursive(tarfile_path)
        print(f'{len(file_list)} files found')

        self.f = open('tags-wd-tagger.txt', 'w', encoding='utf-8')

        self.load_model()

        imgs: List[Image.Image] = []
        fpathes: List[str] = []
        start: float = time.perf_counter()
        last_cnt: int = 0
        cnt: int = 0
        for file_path in file_list:
            try:
                img: Optional[Image.Image] = None
                try:
                    img: Image.Image = Image.open(file_path)
                except Exception as e:
                    if img is not None:
                        img.close()
                    print(f"Failed to open image: {file_path}")
                    continue
                # img.load()
                imgs.append(img)

                fpathes.append(file_path)

                if len(imgs) >= BATCH_SIZE:
                    results_in_csv_format: List[str] = self.predict(imgs, 0.3, True, 0.3, True)
                    for idx, line in enumerate(results_in_csv_format):
                        self.write_to_file(fpathes[idx] + ',' + line)
                    imgs = []
                    fpathes = []

                cnt += 1

                if cnt - last_cnt >= PROGRESS_INTERVAL:
                    now: float = time.perf_counter()
                    print(f'{cnt} files processed')
                    diff: float = now - start
                    print('{:.2f} seconds elapsed'.format(diff))
                    if cnt > 0:
                        time_per_file: float = diff / cnt
                        print('{:.4f} seconds per file'.format(time_per_file))
                    print("", flush=True)
                    last_cnt = cnt

            except Exception as e:
                if img is not None:
                    img.close()
                error_class: type = type(e)
                error_description: str = str(e)
                err_msg: str = '%s: %s' % (error_class, error_description)
                print(err_msg)
                print_traceback()
                continue


#def main(arg_str: str) -> None:
def main(arg_str: List[str]) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--dir', nargs=1, required=True, help='tagging target directory path')
    args: argparse.Namespace = parser.parse_args(arg_str)
    predictor: Predictor = Predictor()
    # predictor.process_directory(arg_str)
    predictor.process_directory(args.dir[0])

#main('/content/freepik')
main(sys.argv[1:])