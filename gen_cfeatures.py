# https://huggingface.co/spaces/deepghs/ccip/blob/f7d50a4f5dd3d4681984187308d70839ff0d3f5b/ccip.py

import datetime
import os, time
import shutil
from pathlib import Path

import argparse
import traceback, sys
import re
import concurrent.futures

import json
import os.path
from io import TextIOWrapper
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, HfFileSystem
from gensim.similarities import Similarity

try:
    from imgutils.data import load_images, ImageTyping
    from imgutils.utils import open_onnx_model
    from onnxruntime import InferenceSession
except (ModuleNotFoundError, ImportError):
    print('Please install the imgutils and onnxruntime package to use charactor feature extraction.')

try:
    from typing import Literal
except (ModuleNotFoundError, ImportError):
    try:
        from typing_extensions import Literal
    except (ModuleNotFoundError, ImportError):
        pass

hf_fs = HfFileSystem()

_VALID_MODEL_NAMES = [
    os.path.basename(os.path.dirname(file)) for file in
    hf_fs.glob('deepghs/ccip_onnx/*/model.ckpt')
]
_DEFAULT_MODEL_NAMES = 'ccip-caformer-24-randaug-pruned'


EXTENSIONS: List[str] = ['.png', '.jpg', '.jpeg', ".PNG", ".JPG", ".JPEG"]

BATCH_SIZE: int = 20
PROGRESS_INTERVAL: int = 100

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
        self.embed_model: Optional[InferenceSession] = None
        self.metric_model: Optional[InferenceSession] = None
        self.threshold: float = -1.0
        self.f: Optional[TextIOWrapper] = None
        self.cindex: Optional[Similarity] = None
        # self.tagger_model: Optional[nn.Module] = None


    def list_files_recursive(self, dir_path: str) -> List[str]:
        file_list: List[str] = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path: str = os.path.join(root, file)
                if any(file_path.endswith(ext) for ext in EXTENSIONS):
                    file_list.append(file_path)
        return file_list

    def write_to_file(self, csv_line: str) -> None:
        self.f.write(csv_line + '\n')

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

    def _open_feat_model(self, model, executor = 'CUDAExecutionProvider') -> InferenceSession:
        return open_onnx_model(hf_hub_download(
                f'deepghs/ccip_onnx',
                f'{model}/model_feat.onnx',
            ),
            mode = executor,
        )

    def _open_metrics(self, model):
        with open(hf_hub_download(f'deepghs/ccip_onnx', f'{model}/metrics.json'), 'r') as f:
            return json.load(f)

    def _open_metric_model(self, model, executor = 'CUDAExecutionProvider') -> InferenceSession:
        return open_onnx_model(hf_hub_download(
            f'deepghs/ccip_onnx',
            f'{model}/model_metrics.onnx',
            ),
            mode = executor,
        )

    #def ccip_batch_extract_features(self, images: MultiImagesTyping, size: int = 384, model: str = _DEFAULT_MODEL_NAMES):
    def ccip_batch_extract_features(self, images: List[np.ndarray], size: int = 384,
                                    model: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
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
        # images = load_images(images, mode='RGB')
        # data = np.stack([self._preprocess_image(item, size=size) for item in images]).astype(np.float32)
        data = np.stack(images).astype(np.float32)
        # output, = self._open_feat_model(model).run(['output'], {'input': data})
        output, = self.embed_model.run(['output'], {'input': data})
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

    def ccip_default_threshold(self, model: str = _DEFAULT_MODEL_NAMES) -> float:
        """
        Retrieves the default threshold value obtained from model metrics in the Hugging Face model repository.
        :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                      The available model names are: ``ccip-caformer-24-randaug-pruned``,
                      ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
        :type model: str
        :return: The default threshold value obtained from model metrics.
        :rtype: float
        Examples::
            >>> from imgutils.metrics import ccip_default_threshold
            >>>
            >>> ccip_default_threshold()
            0.17847511429108218
            >>> ccip_default_threshold('ccip-caformer-6-randaug-pruned_fp32')
            0.1951224011983088
            >>> ccip_default_threshold('ccip-caformer-5_fp32')
            0.18397327797685215
        """
        return self._open_metrics(model)['threshold']

    _FeatureOrImage = Union[ImageTyping, np.ndarray]

    def _p_feature(self, x: _FeatureOrImage, size: int = 384, model: str = _DEFAULT_MODEL_NAMES):
        if isinstance(x, np.ndarray):  # if feature
            return x
        else:  # is image or path
            return self.ccip_extract_feature(x, size, model)

    def ccip_difference(self, x: _FeatureOrImage, y: _FeatureOrImage,
                        size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> float:
        """
        Calculates the difference value between two anime characters based on their images or feature vectors.
        :param x: The image or feature vector of the first anime character.
        :type x: Union[ImageTyping, np.ndarray]
        :param y: The image or feature vector of the second anime character.
        :type y: Union[ImageTyping, np.ndarray]
        :param size: The size of the input image to be used for feature extraction. (default: ``384``)
        :type size: int
        :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                      The available model names are: ``ccip-caformer-24-randaug-pruned``,
                      ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
        :type model: str
        :return: The difference value between the two anime characters.
        :rtype: float
        Examples::
            >>> from imgutils.metrics import ccip_difference
            >>>
            >>> ccip_difference('ccip/1.jpg', 'ccip/2.jpg')  # same character
            0.16583099961280823
            >>>
            >>> # different characters
            >>> ccip_difference('ccip/1.jpg', 'ccip/6.jpg')
            0.42947039008140564
            >>> ccip_difference('ccip/1.jpg', 'ccip/7.jpg')
            0.4037521779537201
            >>> ccip_difference('ccip/2.jpg', 'ccip/6.jpg')
            0.4371533691883087
            >>> ccip_difference('ccip/2.jpg', 'ccip/7.jpg')
            0.40748104453086853
            >>> ccip_difference('ccip/6.jpg', 'ccip/7.jpg')
            0.392294704914093
        """
        return self.ccip_batch_differences([x, y], size, model)[0, 1].item()

    def ccip_batch_differences(self, images: List[_FeatureOrImage],
                               size: int = 384, model: str = _DEFAULT_MODEL_NAMES) -> np.ndarray:
        """
        Calculates the pairwise differences between a given list of images or feature vectors representing anime characters.
        :param images: The list of images or feature vectors representing anime characters.
        :type images: List[Union[ImageTyping, np.ndarray]]
        :param size: The size of the input image to be used for feature extraction. (default: ``384``)
        :type size: int
        :param model: The name of the model to use for feature extraction. (default: ``ccip-caformer-24-randaug-pruned``)
                      The available model names are: ``ccip-caformer-24-randaug-pruned``,
                      ``ccip-caformer-6-randaug-pruned_fp32``, ``ccip-caformer-5_fp32``.
        :type model: str
        :return: The matrix of pairwise differences between the given images or feature vectors.
        :rtype: np.ndarray
        Examples::
            >>> from imgutils.metrics import ccip_batch_differences
            >>>
            >>> ccip_batch_differences(['ccip/1.jpg', 'ccip/2.jpg', 'ccip/6.jpg', 'ccip/7.jpg'])
            array([[6.5350548e-08, 1.6583106e-01, 4.2947042e-01, 4.0375218e-01],
                   [1.6583106e-01, 9.8025822e-08, 4.3715334e-01, 4.0748104e-01],
                   [4.2947042e-01, 4.3715334e-01, 3.2675274e-08, 3.9229470e-01],
                   [4.0375218e-01, 4.0748104e-01, 3.9229470e-01, 6.5350548e-08]],
                  dtype=float32)
        """
        input_ = np.stack([self._p_feature(img, size, model) for img in images]).astype(np.float32)
        output, = self.metric_model.run(['output'], {'input': input_})
        return output

    def predict(
            self,
            images: List[np.ndarray],
    ) -> np.ndarray:
        print("Running inference...")
        ret = self.ccip_batch_extract_features(images)
        print("Processing results...")
        return ret

    def gen_image_ndarray(self, file_path) -> np.ndarray | None:
        try:
            img: Image.Image = load_images([file_path], mode='RGB')[0]
            ret_arr: np.ndarray = self._preprocess_image(img, size=384)
            return ret_arr
        except Exception as e:
            error_class: type = type(e)
            error_description: str = str(e)
            err_msg: str = '%s: %s' % (error_class, error_description)
            print(err_msg)
            return None

    def get_image_feature(self, file_path: str) -> np.ndarray:
        if self.cindex is None:
            self.cindex = Similarity.load('charactor-featues-idx')
            self.threshold = self.ccip_default_threshold(_DEFAULT_MODEL_NAMES) / 1.5

        img: np.ndarray = self.gen_image_ndarray(file_path)
        return self.predict([img])[0]

    def write_vecs_to_index(self, vecs: np.ndarray) -> bool:
        for vec in vecs:
            if self.cindex is None:
                self.cindex = Similarity('charactor-featues-idx', [vec], num_features=768)
            else:
                id_and_vals: List[int, float] = [(ii, val) for ii, val in enumerate(vec)]
                self.cindex.add_documents([id_and_vals])
                #self.cindex.add_documents([vec])


    def process_directory(self, dir_path: str, added_date: datetime.date | None = None) -> None:
        file_list: List[str] = self.list_files_recursive(dir_path)
        print(f'{len(file_list)} files found')

        # Filter files by date if specified
        if added_date is not None:
            file_list = self.filter_files_by_date(file_list, added_date)
            print(f'{len(file_list)} files found after {added_date}')
            
            # Create backup directory with timestamp
            
            # backup_dir = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            # os.makedirs(backup_dir, exist_ok=True)
            #
            # # Backup existing index files
            # for file in Path('.').glob('charactor-featues-idx*'):
            #     shutil.copy2(file, Path(backup_dir) / file.name)
            #     print(f'Backed up {file} to {backup_dir}')

            #self.cindex = Similarity.load('charactor-featues-idx')
            self.cindex = Similarity.load('charactor-featues-idx')
            self.threshold = self.ccip_default_threshold(_DEFAULT_MODEL_NAMES) / 1.5

        self.embed_model = self._open_feat_model(_DEFAULT_MODEL_NAMES)
        self.threshold = self.ccip_default_threshold(_DEFAULT_MODEL_NAMES)
        self.f = open('charactor-featues-idx.csv', 'a', encoding='utf-8')

        ndarrs: List[np.ndarray] = []
        fpathes: List[str] = []
        start: float = time.perf_counter()
        last_cnt: int = 0
        cnt: int = 0
        failed_cnt: int = 0
        passed_idx: int = 0
        future_to_vec: dict[concurrent.futures.Future[np.ndarray], bool] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor_vec_write:
            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKER_NUM) as executor:
                # dispatch get Tensor task to processes
                future_to_path = {executor.submit(self.gen_image_ndarray, file_path): file_path for file_path in
                                  file_list[0: BATCH_SIZE]}
                passed_idx += BATCH_SIZE
                while passed_idx < len(file_list):
                    for future in concurrent.futures.as_completed(future_to_path):
                        path = future_to_path[future]
                        try:
                            ndarr = future.result()
                            if ndarr is None:
                                failed_cnt += 1
                                cnt -= 1
                                # continue

                            if ndarr is not None:
                                ndarrs.append(ndarr)
                                fpathes.append(path)

                            if len(ndarrs) >= BATCH_SIZE - failed_cnt:
                                # submit load Tensor tasks for next batch
                                end_idx = passed_idx + BATCH_SIZE
                                if end_idx > len(file_list):
                                    end_idx = len(file_list)
                                future_to_path = {executor.submit(self.gen_image_ndarray, file_path): file_path for file_path
                                                  in file_list[passed_idx: end_idx]}
                                passed_idx = end_idx

                                # run inference
                                # dimension of results: (batch_size, 768)
                                results: np.ndarray = self.predict(ndarrs)
                                for idx in range(0, len(results)):
                                    self.write_to_file(fpathes[idx])
                                # submit write to index tasks to another thread
                                future_to_vec[executor_vec_write.submit(self.write_vecs_to_index, results)] = True
                                #self.write_vecs_to_index(results)
                                # for idx, line in enumerate(results_in_csv_format):
                                #     self.write_to_file(fpathes[idx] + ',' + line)
                                # for arr in results:
                                #     print(arr.astype(float))
                                ndarrs = []
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
                                #self.cindex.save()

                        except Exception as e:
                            error_class: type = type(e)
                            error_description: str = str(e)
                            err_msg: str = '%s: %s' % (error_class, error_description)
                            print(err_msg)
                            print_traceback()
                            continue

        # wait for all tasks to be finished
        for future in concurrent.futures.as_completed(future_to_vec):
            try:
                future.result()
            except Exception as e:
                error_class: type = type(e)
                error_description: str = str(e)
                err_msg: str = '%s: %s' % (error_class, error_description)
                print(err_msg)
                print_traceback()
                continue
        #self.cindex.save('charactor-featues-idx')
        self.cindex.save()

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
