import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, itertools
import traceback, sys, re, time
from typing import List, Optional, Any, Protocol, Callable

from PIL import Image
import faiss
import open_clip, torch

CLIP_MODEL_REPO: str = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
EMBED_DIM: int = 1024
IMAGE_SIZE: int = 224
INDEX_FNAME: str = 'clip-index'
INDEX_FPATHES_FNAME: str = 'clip-index-fpathes.txt'
BATCH_SIZE: int = 10

# extensions of image files to be processed
EXTENSIONS: List[str] = ['.png', '.jpg', '.jpeg', ".PNG", ".JPG", ".JPEG"]

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

# list up files and filter by extension
def list_files_recursive(directory: str) -> List[str]:
    file_list: List[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path: str = os.path.join(root, file)
            if any(file_path.endswith(ext) for ext in EXTENSIONS):
                file_list.append(file_path)
    return file_list

class ImageEncodable(Protocol):
    encode_image: Callable[[Image.Image], torch.Tensor]


class Predictor:
    def __init__(self) -> None:
        self.model_target_size: Optional[int] = None
        self.last_loaded_repo: Optional[str] = None
        self.clip_model: Optional[ImageEncodable] = None
        self.preprocess: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.index: Optional[faiss.IndexFlatIP] = None

    def load_model(self) -> None:
        if self.clip_model is not None:
            return

        self.model_target_size = IMAGE_SIZE

        self.clip_model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:' + CLIP_MODEL_REPO, device='cpu')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:' + CLIP_MODEL_REPO)

    def get_feature_vectors(self, images: List[Any]) -> List[torch.Tensor]:
        if self.clip_model is None:
            print("self.clip_model is None")
            exit(1)

        ret_vectors: List[torch.Tensor] = []
        with torch.no_grad():
            for image in images:
                image_features: torch.Tensor = self.clip_model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                ret_vectors.append(image_features)
        return ret_vectors

    # write lines to file
    def write_to_file(self, lines: List[str]) -> None:
        for line in lines:
            self.f.write(line + '\n')
        self.f.flush()

    def create_vector_index(self) -> None:
        # Create vector index with Faiss
        self.index = faiss.IndexFlatIP(EMBED_DIM)

    def add_vectors_to_index(self, feature_vectors: List[torch.Tensor]) -> None:
        if self.index is None:
            print("self.index is None")
            exit(1)

        feature_vectors_for_faiss: Any = torch.cat(feature_vectors).detach().numpy()
        self.index.add(feature_vectors_for_faiss)
        faiss.write_index(self.index, INDEX_FNAME)

    # root function
    def process_directory(self, directory: str) -> None:
        file_list: List[str] = list_files_recursive(directory)
        print(f'{len(file_list)} files found')

        # file for vectorized image filepathes
        self.f = open(INDEX_FPATHES_FNAME, 'w', encoding='utf-8')
        self.load_model()

        feature_vectors_all: List[torch.Tensor] = []

        idx: int = 0
        cnt: int = 0
        last_cnt: int = 0
        start: float = time.perf_counter()
        # process each image file in batch
        while True:
            path_batch: List[str] = list(itertools.islice(file_list, idx, idx + BATCH_SIZE))

            if len(path_batch) == 0:
                break
            idx += BATCH_SIZE

            images: List[Any] = []
            indexed_file_pathes: List[str] = []
            for file_path in path_batch:
                try:
                    img: Image.Image = Image.open(file_path)
                    if self.preprocess is None:
                        print("preprocess is None")
                        exit(1)

                    img = self.preprocess(img).unsqueeze(0)
                    images.append(img)
                    indexed_file_pathes.append(file_path)
                except Exception as e:
                    error_class: type = type(e)
                    error_description: str = str(e)
                    err_msg: str = '%s: %s' % (error_class, error_description)
                    print(err_msg)
                    print_traceback()
                    pass

            feature_vectors: List[torch.Tensor] = self.get_feature_vectors(images)
            feature_vectors_all.extend(feature_vectors)

            # Write vectorized image filepathes to file each time
            self.write_to_file(indexed_file_pathes)

            cnt += len(feature_vectors)

            if cnt - last_cnt >= 100:
                now: float = time.perf_counter()
                print(f'{cnt} files processed')
                diff: float = now - start
                print('{:.2f} seconds elapsed'.format(diff))
                time_per_file: float = diff / cnt
                print('{:.4f} seconds per file'.format(time_per_file))
                last_cnt = cnt
                print("", flush=True)

        # Create vector index
        self.create_vector_index()
        # Add vectors to index and save to file
        self.add_vectors_to_index(feature_vectors_all)

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--dir', nargs=1, required=True, help='image files directory path')
    args: argparse.Namespace = parser.parse_args()

    predictor: Predictor = Predictor()
    predictor.process_directory(args.dir[0])

if __name__ == "__main__":
    main()