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

def sort_and_uniq(tags: List[str]) -> List[str]:
    tags = list(set(tags))
    tags.sort()
    return tags

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


    def write_to_file(self, csv_line: str) -> None:
        self.f.write(csv_line + '\n')
        self.f.flush()

    # collect the last element of list and save a collected list to a new csv file
    def replace_and_etc(
            self,
            file_path: str,
    ) -> None:
        self.load_labels_hf(repo_id=TAGGER_VIT_MODEL_REPO)

        self.f = open(file_path.split('.')[0] + '_replaced.csv', 'w', encoding='utf-8')

        tagged_info_list: List[List[str]] = []
        cheet_sheet_fpath: str = file_path.split('.')[0] + '_tag_chet_sheet.csv'
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                row: List[str] = line.strip().split(",")
                row[-1] = row[-1].replace("\\", "")
                tagged_info_list.append(row)

        last_elems: List[str] = [x[-1].replace("\\", "") for x in tagged_info_list]
        last_elems = sort_and_uniq(last_elems)

        # make dict for replacing
        character_res: dict = {}
        character_tags: List[str] = [self.tag_names[i] for i in self.character_index]
        for elem in last_elems:
            for tag in character_tags:
                if elem.endswith(tag):
                    if len(elem) > len(tag):
                        character_res[elem] = elem.replace(tag, "," + tag)
                    else:
                        # last element is collected tag
                        character_res[elem] = tag
                    break

        # replace and write to file
        deduplicate_dict: Dict = {}
        for tags in tagged_info_list:
            if tags[-1] in character_res:
                tags[-1] = character_res[tags[-1]]
            if tags[0] in deduplicate_dict:
                continue
            else:
                deduplicate_dict[tags[0]] = True
                self.write_to_file(','.join(tags))

        # write cheet sheet
        contained_character_tags: List[str] = [ value for value in character_res.values() ]
        tmp_list: List[str] = []
        for idx in range(len(contained_character_tags)):
            splited = contained_character_tags[idx].split(',')
            if len(splited) > 1:
                tmp_list.append(splited[1])
            else:
                tmp_list.append(contained_character_tags[idx])
        contained_character_tags = tmp_list
        contained_character_tags = sort_and_uniq(contained_character_tags)
        with open(cheet_sheet_fpath, 'w', encoding='utf-8') as cheet_sheet_f:
            for tag in contained_character_tags:
                cheet_sheet_f.write(tag + '\n')


def main(arg_str: List[str]) -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--csv', nargs=1, required=True, help='replace target csv file path')
    args: argparse.Namespace = parser.parse_args(arg_str)
    replacer: Replacer = Replacer()
    replacer.replace_and_etc(args.csv[0])

if __name__ == '__main__':
    main(sys.argv[1:])