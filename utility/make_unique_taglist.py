# script for covering bug of tagger script

import pandas as pd
import argparse
import traceback, sys
import re
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

TAGGER_VIT_MODEL_REPO: str = "SmilingWolf/wd-eva02-large-tagger-v3"

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

class Replacer:
    def __init__(self) -> None:
        self.tag_names: Optional[List[str]] = None
        self.character_index: Optional[List[int]] = None

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
        self.character_index = list(np.where(df["category"] == 4)[0])
        self.tag_names = df["name"].tolist()

    # collect the last element of list and save a collected list to a new csv file
    def replace_and_etc(
            self,
            file_path: str,
    ) -> None:
        self.load_labels_hf(repo_id=TAGGER_VIT_MODEL_REPO)

        taggs_dict: Dict[str, bool] = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                row: List[str] = line.strip().split(",")
                row[-1] = row[-1].replace("\\", "")
                for tag in row:
                    taggs_dict[tag] = True

        all_tags: List[str] = list(taggs_dict.keys())
        conved_and_char_tags_dict: Dict[str, bool] = {}

        # make replaced dict
        character_tags: List[str] = [self.tag_names[i] for i in self.character_index]
        for elem in all_tags:
            for tag in character_tags:
                if elem.endswith(tag):
                    conved_and_char_tags_dict[elem.replace(tag, '')] = True
                    conved_and_char_tags_dict[tag] = True
                    break

        # write cheet sheet
        result_list: List[str] = list(conved_and_char_tags_dict.keys())
        result_list.sort()

        cheet_sheet_fpath: str = file_path.split('.')[0] + '_unique_tags.csv'
        with open(cheet_sheet_fpath, 'w', encoding='utf-8') as cheet_sheet_f:
            for tag in result_list:
                try:
                    cheet_sheet_f.write(tag + '\n')
                except Exception as e:
                    print(f'error: {tag}')
                    continue

# python --tags tags-wd-tagger.txt
def main(arg_str: List[str]) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--tags', nargs=1, required=True, help='analyze target tags file path')
    args: argparse.Namespace = parser.parse_args(arg_str)
    replacer: Replacer = Replacer()
    replacer.replace_and_etc(args.tags[0])

if __name__ == '__main__':
    main(sys.argv[1:])