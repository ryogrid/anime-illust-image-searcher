# -- coding: utf-8 --

from typing import Dict, List

def main() -> None:
    tag_map: Dict[str, bool] = {}
    with open('tags-wd-tagger.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tags: List[str] = line.strip().split(',')
            tags = tags[1:-1]
            for tag in tags:
                tag_map[tag] = True
    print(f'{len(tag_map)} unique tags found')

if __name__ == '__main__':
    main()
