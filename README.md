# local-illust-image-searcher
## What's this?
- Local illustration image file search engine with ML technique
  - Can be used for photos. but flexible photo search is offered by Google Photos or etc :)
- Search capabilities of cloud photo album services are poor for some reason
- So, I wrote simple scripts

## Method
- Image file tagging with Visual Transformar (ViT) and Latent Semantic Indexing (LSI)
- LSI is used for covering tagging presision
  - You can find image files using tags which is difficult for tagging because search index is applyed LSI

## Usage
- $ pip install -r requirements.txt
- $ python make-tags-with-wd-tagger.py --dir "IMAGE FILES CONTAINED DIR PATH"
  - The script searches directory structure recursively :)
  - This takes quite a while...
    - About 1 file/s at middle spec desktop PC (GPU is not used)
      - AMD Ryzen 7 5700X 8-Core Processor 4.50 GHz
  - Image files and its tags are saved to tags-wd-tagger.txt
- $ python count-unique-tag-num.py
  - => for deciding appropriate dimension scale fitting your data
- $ python gen-lsi-model.py
  - **Please edit [num_topics paramater](https://github.com/ryogrid/local-illust-image-searcher/blob/main/gen-lsi-model.py#L51) before execution**
  - I think about 80% of unique tags which is counted with count-unique-tag-num.py is better
    - EX: unique tags count is 1000 -> 0.8 * 1000 -> 800 num_topics (dimension)
  - This takes quite a while...
    - Take several sec only for 1000 files and reduction from 800 to 700 dimension case
    - But, in 330k files and from 8000 to 5000 dimension case, about 1 hour is taken
      - files are not for demo :)
    - LSI processing: dimension reduction about 800 dimenstion to 700 dimension  
- $ streamlit run web-ui-image-search-lsi.py
  - Search app is opend on your web brower

## Screenshot
- I used about 1000 image files collected from [Irasutoya](https://www.irasutoya.com/) which offers free image materials
  - Note: image materials of Irasutoya have restrictions for commercial purposes

![image](https://github.com/user-attachments/assets/3e3a6dce-b3aa-491f-8727-52282821ac7f)
![image](https://github.com/user-attachments/assets/3e66dd5e-7e68-4c68-9d29-884d08ae1e18)


## Information related to copylight
- I used [this code](https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py) as reference wheh implmenting tagger script
- ["WD ViT Tagger v3" model](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) is used for image file tagging
- **I thank to great works of SmilingWolf**
