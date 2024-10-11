import os
import argparse, itertools
import traceback, sys, re

import numpy as np
from PIL import Image
import faiss
import open_clip, torch

CLIP_MODEL_REPO = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
EMBED_DIM = 1024
IMAGE_SIZE = 224
INDEX_FNAME = 'clip-index'
INDEX_FPATHES_FNAME = 'clip-index-fpathes.txt'
BATCH_SIZE = 50

# extensions of image files to be processed
EXTENSIONS = ['.png', '.jpg', '.jpeg', ".PNG", ".JPG", ".JPEG"]

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
        self.clip_model = None
        self.preprocess = None
        self.tokenizer = None

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
        if self.clip_model is not None:
            return
        
        self.model_target_size = IMAGE_SIZE

        self.clip_model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    def get_feature_vectors(
        self,
        images, # already prepared images
    ):
        cated_images = torch.cat(images, dim=0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            return self.clip_model.get_image_features(cated_images)

    # write lines to file
    def write_to_file(self, lines):    
        for line in lines:
            self.f.write(line + '\n')
        self.f.flush()

    def create_vector_index(self, feature_vectors):
        # Create vector index with Faiss
        quantizer = faiss.IndexFlatL2(EMBED_DIM)
        index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, 5)        
        feature_vectors_for_faiss = torch.cat(feature_vectors).detach().numpy()
        index.train(feature_vectors_for_faiss)
        index.add(feature_vectors_for_faiss)
        faiss.write_index(index, INDEX_FNAME)

    # root function
    def process_directory(self, directory):
        file_list = list_files_recursive(directory)
        print(f'{len(file_list)} files found')

        # file for vectorized image filepathes
        self.f = open('clip-index-fpathes.txt', 'w', encoding='utf-8')
        self.load_model()

        feature_vectors_all = []
        
        idx = 0
        cnt = 0
        last_cnt = 0
        # process each image file in batch
        while True:
            path_batch = list(itertools.islice(file_list, idx, idx + BATCH_SIZE))

            if len(path_batch) == 0:
                break
            idx += BATCH_SIZE

            images = []
            indexed_file_pathes = []
            for file_path in path_batch:
                try:
                    img = Image.open(file_path)
                    img = self.prepare_image(img)
                    img = self.preprocess(img).unsqueeze(0)
                    images.append(img)
                    indexed_file_pathes.append(file_path)
                except Exception as e:
                    error_class = type(e)
                    error_description = str(e)
                    err_msg = '%s: %s' % (error_class, error_description)
                    print(err_msg)
                    print_traceback()
                    pass            

            feature_vectors = self.get_feature_vectors(images)
            feature_vectors_all.append(feature_vectors)
            
            # Write vectorized image filepathes to file each time
            self.write_to_file(indexed_file_pathes)

            if cnt - last_cnt >= 100:
                print(f'{cnt} files processed')
                last_cnt = cnt

            cnt += len(feature_vectors)

        self.create_vector_index(feature_vectors_all)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', nargs=1, required=True, help='image files directory path')
    args = parser.parse_args()

    predictor = Predictor()
    predictor.process_directory(args.dir[0])

if __name__ == "__main__":
    main()