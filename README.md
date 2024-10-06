# local-illust-image-searcher
## About
- Local illustration image file search engine with ML technique
  - Can be used for photos. but flexible photo search is offered Google photo or etc :)
- Search capabilities of cloud photo album services is poor for some reason
- So, I wrote simple scripts

## Method
- Image file tagging with Visual Transformar (ViT) and Latent Semantic Indexing (LSI)
- LSI is used for covering tagging presision
  - You can find image files using tags which is difficult for tagging because search index is applyed LSI

## Usage
- pip install -r requirements.py
- python make-tags-with-wd-tagger.py
  - this takes quite a while...
  - about 1 file/s at middle spec desktop PC (GPU is not used)
- python count-unique-tag-num.py
  - => for deciding appropriate dimension scale fitting your data

## Information related to copylight
- I Used [this code](https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py) as reference wheh implmenting tagger script
- ["WD ViT Tagger v3" model](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) is used for image file tagging
- **I thank to great works of SmilingWolf**
