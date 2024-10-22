import os, time
from PIL import Image
import argparse
import sys
from typing import List, Tuple

EXTENSIONS: List[str] = ['.png', '.jpg', '.jpeg', ".PNG", ".JPG", ".JPEG"]
RESEZE_TARGET_SIZE: int = 448

def list_files_recursive(directory: str) -> List[str]:
    file_list: List[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path: str = os.path.join(root, file)
            if any(file_path.endswith(ext) for ext in EXTENSIONS):
                file_list.append(file_path)
    return file_list

def resize_image(image: Image.Image, target_size: int) -> Image.Image:
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

    return padded_image

def main(arg_str: list[str]) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--dir', nargs=1, required=True, help='resize target directory (ATTENTION: file is overwritten)')
    args: argparse.Namespace = parser.parse_args(arg_str)

    dir_path: str = args.dir[0]
    file_list: List[str] = list_files_recursive(dir_path)
    cnt = 0
    start = time.perf_counter()
    for file_path in file_list:
        try:
            image: Image.Image = Image.open(file_path)
            resized_image: Image.Image = resize_image(image, RESEZE_TARGET_SIZE)
            resized_image.save(file_path)
        except Exception as e:
            print(f"Failed to resize image: {file_path}")
            print(f"Remove: {file_path}")
            os.remove(file_path)

        cnt += 1

        if cnt % 100 == 0:
            end = time.perf_counter()
            print(f"Processed {cnt} images.")
            print(f"Elapsed time: {end - start:.2f} sec")
            print(f"throughput: {cnt / (end - start):.2f} images/sec")

if __name__ == "__main__":
    main(sys.argv[1:])
