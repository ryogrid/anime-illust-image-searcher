import concurrent.futures

import os, time

import pandas as pd
import argparse
import traceback, sys
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable, Protocol

import numpy as np
from numpy import signedinteger
from PIL import Image
import timm
from timm.data import create_transform, resolve_data_config
import torch
from torch import Tensor, nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import concurrent.futures

# (venv) PS C:\Data\work\anime-illust-image-searcher-for-trial> python .\utility\make_tensor_files.py
# --dir H:\somedir\images\241101_1 H:\somedir\images\241101_2 H:\somedir\images\241101_3 H:\somedir\images\241101_4
# --dirbase H:\somedir\images\
# --dest H:\images\
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

WORKER_NUM: int = 8
RESIZE_TARGET_SIZE: int = 448

def mcut_threshold(probs: np.ndarray) -> float:
    sorted_probs: np.ndarray = probs[probs.argsort()[::-1]]
    difs: np.ndarray = sorted_probs[:-1] - sorted_probs[1:]
    t: signedinteger[Any] = difs.argmax()
    thresh: float = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh

def list_files_recursive(directory: str) -> List[str]:
    file_list: List[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path: str = os.path.join(root, file)
            if any(file_path.endswith(ext) for ext in EXTENSIONS):
                file_list.append(file_path)
    return file_list

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
        self.args: Optional[argparse.Namespace] = None

    def list_files_recursive(self, dir_path: str) -> List[str]:
        file_list: List[str] = []
        for root, _, files in os.walk(dir_path):
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
            # copy image to avoid error at convert method call
            image = image.copy()
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

    def save_tensor(self, tensor: Tensor, file_path: str) -> None:
        save_path:str = file_path.replace(self.args.dirbase[0], self.args.dest[0])
        try:
            path_dir = os.path.dirname(save_path)
            if not os.path.exists(path_dir):
                # create directory if not exists
                os.makedirs(path_dir)
            torch.save(tensor, save_path)
        except Exception as e:
            error_class: type = type(e)
            error_description: str = str(e)
            err_msg: str = '%s: %s' % (error_class, error_description)
            print(err_msg)
            print_traceback()

    def gen_image_tensor(self, file_path: str) -> Tensor:
        img: Image.Image = None
        try:
            img = Image.open(file_path)
            img.load()
            img_tmp = self.prepare_image(img)
            # run the model's input transform to convert to tensor and rescale
            input: Tensor = self.transform(img_tmp)
            # NCHW image RGB to BGR
            input = input[[2, 1, 0]]
            return input
        except Exception as e:
            if img is not None:
                img.close()
            error_class: type = type(e)
            error_description: str = str(e)
            err_msg: str = '%s: %s' % (error_class, error_description)
            print(err_msg)
            return None

    def tensor_file_convert_th(self, file_path: str) -> bool:
        try:
            got_tensor: Tensor = self.gen_image_tensor(file_path)
            self.save_tensor(got_tensor, file_path)
            return True
        except Exception as e:
            print(f"Failed to convert image to tensor: {file_path}")
            print(e)
            return False

    def process_directory(self, dir_path: str) -> None:
        file_list: List[str] = self.list_files_recursive(dir_path)
        print(f'{len(file_list)} files found')

        self.load_model()

        start: float = time.perf_counter()
        last_cnt: int = 0
        cnt: int = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKER_NUM) as executor:
            # dispatch get Tensor task to processes
            future_to_path = {executor.submit(self.tensor_file_convert_th, file_path): file_path for file_path in file_list}
            #for file_path in file_list:
            for future in concurrent.futures.as_completed(future_to_path):
            # for file_path in file_list:
                try:
                    # result: Tensor | None = self.gen_image_tensor(file_path)
                    # if result is None:
                    #     print("Failed to convert image to tensor")
                    #     continue
                    # self.save_tensor(result, file_path)

                    result = future.result()
                    if result is False:
                        print("Failed to convert image to tensor")
                        continue

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
                    error_class: type = type(e)
                    error_description: str = str(e)
                    err_msg: str = '%s: %s' % (error_class, error_description)
                    print(err_msg)
                    print_traceback()
                    continue

def main(arg_str: list[str]) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--dir', nargs='*', required=True, help='convert target directory')
    parser.add_argument('--dest', nargs=1, required=True, help='tensor file ouput directory')
    parser.add_argument('--dirbase', nargs=1, required=True, help='ignored directory structure part')

    args: argparse.Namespace = parser.parse_args(arg_str)

    predictor: Predictor = Predictor()
    predictor.args = args
    for path in args.dir:
        predictor.process_directory(path)

if __name__ == "__main__":
    main(sys.argv[1:])
