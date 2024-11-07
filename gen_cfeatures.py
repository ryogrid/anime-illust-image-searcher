import datetime
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
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import concurrent.futures

import json
import os.path
from functools import lru_cache
from typing import Union, List

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, HfFileSystem

from imgutils.data import MultiImagesTyping, load_images, ImageTyping
from imgutils.utils import open_onnx_model

try:
    from typing import Literal
except (ModuleNotFoundError, ImportError):
    from typing_extensions import Literal

hf_fs = HfFileSystem()


_VALID_MODEL_NAMES = [
    os.path.basename(os.path.dirname(file)) for file in
    hf_fs.glob('deepghs/ccip_onnx/*/model.ckpt')
]
_DEFAULT_MODEL_NAMES = 'ccip-caformer-24-randaug-pruned'


EXTENSIONS: List[str] = ['.png', '.jpg', '.jpeg', ".PNG", ".JPG", ".JPEG"]

BATCH_SIZE: int = 10 # max size for M1 MBA GPU
PROGRESS_INTERVAL: int = 1000

WORKER_NUM: int = 8

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
        pass
        # self.last_loaded_repo: Optional[str] = None
        # self.tagger_model: Optional[nn.Module] = None
        # self.tag_names: Optional[List[str]] = None
        # self.rating_index: Optional[List[int]] = None
        # self.general_index: Optional[List[int]] = None
        # self.character_index: Optional[List[int]] = None
        # self.transform: Optional[Callable] = None

    def list_files_recursive(self, dir_path: str) -> List[str]:
        file_list: List[str] = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path: str = os.path.join(root, file)
                if any(file_path.endswith(ext) for ext in EXTENSIONS):
                    file_list.append(file_path)
        return file_list

    # def prepare_image(self, image: Image.Image) -> Image.Image:
    #     #target_size: int = self.model_target_size
    #
    #     if image.mode in ('RGBA', 'LA'):
    #         background: Image.Image = Image.new("RGB", image.size, (255, 255, 255))
    #         background.paste(image, mask=image.split()[-1])
    #         image = background
    #     else:
    #         # copy image to avoid error at convert method call
    #         image = image.copy()
    #         image = image.convert("RGB")
    #
    #     image_shape: Tuple[int, int] = image.size
    #     max_dim: int = max(image_shape)
    #     pad_left: int = (max_dim - image_shape[0]) // 2
    #     pad_top: int = (max_dim - image_shape[1]) // 2
    #
    #     padded_image: Image.Image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    #     padded_image.paste(image, (pad_left, pad_top))
    #
    #     return padded_image

    # def load_model(self) -> None:
        # if self.tagger_model is not None:
        #     return
        #
        # self.tagger_model = timm.create_model("hf-hub:" + TAGGER_VIT_MODEL_REPO).eval()
        # state_dict = timm.models.load_state_dict_from_hf(TAGGER_VIT_MODEL_REPO)
        # self.tagger_model.load_state_dict(state_dict)
        #
        # print("Loading tag list...")
        # self.load_labels_hf(repo_id=TAGGER_VIT_MODEL_REPO)
        #
        # print("Creating data transform...")
        # self.transform = create_transform(**resolve_data_config(self.tagger_model.pretrained_cfg, model=self.tagger_model))

    def write_to_file(self, csv_line: str) -> None:
        self.f.write(csv_line + '\n')

    # def gen_image_tensor(self, file_path: str) -> Tensor | None:
    #     img: Image.Image = None
    #     try:
    #       img = Image.open(file_path)
    #       img.load()
    #       img_tmp = self.prepare_image(img)
    #       # run the model's input transform to convert to tensor and rescale
    #       input: Tensor = self.transform(img_tmp)
    #       # NCHW image RGB to BGR
    #       input = input[[2, 1, 0]]
    #       return input
    #     except Exception as e:
    #       if img is not None:
    #         img.close()
    #       error_class: type = type(e)
    #       error_description: str = str(e)
    #       err_msg: str = '%s: %s' % (error_class, error_description)
    #       print(err_msg)
    #       return None

    def filter_files_by_date(self, file_list: List[str], added_date: datetime.date) -> List[str]:
        filtered_list: List[str] = []
        for file_path in file_list:
            stat = os.stat(file_path)
            ctime: datetime.date = datetime.date.fromtimestamp(stat.st_ctime)
            if ctime >= added_date:
                filtered_list.append(file_path)

        return filtered_list

    def _normalize(self, data, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
        mean, std = np.asarray(mean), np.asarray(std)
        return (data - mean[:, None, None]) / std[:, None, None]

    def _preprocess_image(self, image: Image.Image, size: int = 384):
        image = image.resize((size, size), resample=Image.BILINEAR)
        # noinspection PyTypeChecker
        data = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
        data = self._normalize(data)

        return data

    @lru_cache()
    def _open_feat_model(self, model):
        return open_onnx_model(hf_hub_download(
            f'deepghs/ccip_onnx',
            f'{model}/model_feat.onnx',
        ))

    @lru_cache()
    def _open_metric_model(self, model):
        return open_onnx_model(hf_hub_download(
            f'deepghs/ccip_onnx',
            f'{model}/model_metrics.onnx',
        ))

    @lru_cache()
    def _open_metrics(self, model):
        with open(hf_hub_download(f'deepghs/ccip_onnx', f'{model}/metrics.json'), 'r') as f:
            return json.load(f)

    @lru_cache()
    def _open_cluster_metrics(self, model):
        with open(hf_hub_download(f'deepghs/ccip_onnx', f'{model}/cluster.json'), 'r') as f:
            return json.load(f)

    def ccip_batch_extract_features(self, images: MultiImagesTyping, size: int = 384, model: str = _DEFAULT_MODEL_NAMES):
        """
        Extracts the feature vectors of multiple images using the specified model.
        :param images: The input images from which to extract the feature vectors.
        :type images: MultiImagesTyping
        :param size: The size of the input image to be used for feature extraction. (default: ``384``)
        :type size: int
        :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                      The available model names are: ``ccip-caformer-24-randaug-pruned``,
                      ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
        :type model: str
        :return: The feature vectors of the input images.
        :rtype: numpy.ndarray
        Examples::
            >>> from imgutils.metrics import ccip_batch_extract_features
            >>>
            >>> feat = ccip_batch_extract_features(['ccip/1.jpg', 'ccip/2.jpg', 'ccip/6.jpg'])
            >>> feat.shape, feat.dtype
            ((3, 768), dtype('float32'))
        """
        images = load_images(images, mode='RGB')
        data = np.stack([self._preprocess_image(item, size=size) for item in images]).astype(np.float32)
        output, = self._open_feat_model(model).run(['output'], {'input': data})
        return output

    def ccip_extract_feature(self, image: ImageTyping, size: int = 384, model: str = _DEFAULT_MODEL_NAMES):
        """
        Extracts the feature vector of the character from the given anime image.
        :param image: The anime image containing a single character.
        :type image: ImageTyping
        :param size: The size of the input image to be used for feature extraction. (default: ``384``)
        :type size: int
        :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                      The available model names are: ``ccip-caformer-24-randaug-pruned``,
                      ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
        :type model: str
        :return: The feature vector of the character.
        :rtype: numpy.ndarray
        Examples::
            >>> from imgutils.metrics import ccip_extract_feature
            >>>
            >>> feat = ccip_extract_feature('ccip/1.jpg')
            >>> feat.shape, feat.dtype
            ((768,), dtype('float32'))
        """
        return self.ccip_batch_extract_features([image], size, model)[0]

    _FeatureOrImage = Union[ImageTyping, np.ndarray]

    def _p_feature(self, x: _FeatureOrImage, size: int = 384, model: str = _DEFAULT_MODEL_NAMES):
        if isinstance(x, np.ndarray):  # if feature
            return x
        else:  # is image or path
            return self.ccip_extract_feature(x, size, model)

    def predict(
            self,
            tensors: List[np.ndarray],
    ) -> List[str]:
        pass
        # batched_tensor = torch.stack(tensors, dim=0)
        #
        # print("Running inference...")
        # with torch.inference_mode():
        #     # move model to GPU, if available
        #     model = self.tagger_model
        #     if torch_device.type != "cpu":
        #         model = self.tagger_model.to(torch_device)
        #         batched_tensor = batched_tensor.to(torch_device)
        #     # run the model
        #     outputs = model.forward(batched_tensor)
        #     # apply the final activation function (timm doesn't support doing this internally)
        #     outputs = F.sigmoid(outputs)
        #     # move inputs, outputs, and model back to to cpu if we were on GPU
        #     if torch_device.type != "cpu":
        #         outputs = outputs.to("cpu")
        #
        # print("Processing results...")
        # preds = outputs.numpy()

    def process_directory(self, dir_path: str, added_date: datetime.date | None = None) -> None:
        file_list: List[str] = self.list_files_recursive(dir_path)
        print(f'{len(file_list)} files found')

        # tag new images after specified date
        if added_date is not None:
            file_list = self.filter_files_by_date(file_list, added_date)
            print(f'{len(file_list)} files found after {added_date}')
            # backup tags-wd-tagger.txt with copying to tags-wd-tagger.txt.bak
            if os.path.exists('tags-wd-tagger.txt'):
                with open('tags-wd-tagger.txt', 'r', encoding='utf-8') as f:
                    with open('tags-wd-tagger.txt.bak', 'w', encoding='utf-8') as f_bak:
                        f_bak.write(f.read())
            else:
                print('tags-wd-tagger.txt not found')
                exit(1)

        self.f = open('tags-wd-tagger.txt', 'a', encoding='utf-8')

        self.load_model()

        tensors: List[Tensor] = []
        fpathes: List[str] = []
        start: float = time.perf_counter()
        last_cnt: int = 0
        cnt: int = 0
        failed_cnt: int = 0
        passed_idx: int = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKER_NUM) as executor:
            # dispatch get Tensor task to processes
            future_to_path = {executor.submit(self.gen_image_tensor, file_path): file_path for file_path in
                              file_list[0: BATCH_SIZE]}
            passed_idx += BATCH_SIZE
            while passed_idx < len(file_list):
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        tensor = future.result()
                        if tensor is None:
                            failed_cnt += 1
                            cnt -= 1
                            # continue

                        if tensor is not None:
                            tensors.append(tensor)
                            fpathes.append(path)

                        if len(tensors) >= BATCH_SIZE - failed_cnt:
                            # submit load Tensor tasks for next batch
                            end_idx = passed_idx + BATCH_SIZE
                            if end_idx > len(file_list):
                                end_idx = len(file_list)
                            future_to_path = {executor.submit(self.gen_image_tensor, file_path): file_path for file_path
                                              in file_list[passed_idx: end_idx]}
                            passed_idx = end_idx

                            # run inference
                            results_in_csv_format: List[str] = self.predict(tensors)
                            for idx, line in enumerate(results_in_csv_format):
                                self.write_to_file(fpathes[idx] + ',' + line)
                            tensors = []
                            fpathes = []
                            failed_cnt = 0

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
    parser.add_argument('--dir', nargs=1, required=True, help='tagging target directory path')
    # Note: when specified --after, create tags-wd-tagger.txt.bak file and update tags-wd-tagger.txt
    parser.add_argument('--after', nargs=1,
                        help='tagging new images after this date (mtime attribute). Format: YYYY-MM-DD')
    args: argparse.Namespace = parser.parse_args(arg_str)

    predictor: Predictor = Predictor()
    if args.after is not None:
        try:
            after_date: datetime.date = datetime.datetime.strptime(args.after[0], '%Y-%m-%d').date()
        except Exception as e:
            error_class: type = type(e)
            error_description: str = str(e)
            err_msg: str = '%s: %s' % (error_class, error_description)
            print(err_msg)
            print('Invalid date format. format is YYYY-MM-DD')
            exit(1)

        predictor.process_directory(args.dir[0], after_date)
    else:
        predictor.process_directory(args.dir[0])

if __name__ == "__main__":
    main(sys.argv[1:])
