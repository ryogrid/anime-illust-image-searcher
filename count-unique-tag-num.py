# -- coding: utf-8 --

tag_map = {}
with open('tags-wd-tagger.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tags = line.strip().split(',')
        tags = tags[1:-1]
        for tag in tags:
            tag_map[tag] = True
print(f'{len(tag_map)} unique tags found')