# Anime Style Illustration Specific Image Search App with ViT Tagger x LSI
## What's This?
- Anime Style Illustration Specific Image Search App with ML Technique
  - can be used for photos. but flexible photo search is offered by Google Photos or etc :)
- Search capabilities of cloud photo album services towards illustration image files are poor for some reason
- So, I wrote simple scripts

## Method
- Search Images Matching with Query Texts on Latent Semantic Representation Vector Space
  - Vectors are generated with embedding model: Visual Transformar (ViT) Tagger x Latent Semantic Indexing (LSI)
  - Scores which is calculated with [bm25](https://en.wikipedia.org/wiki/Okapi_BM25) is used in combination
  - Internal relanking method is also used
    - Assumption: Users make queries better asymptotically according to top search results and find appropriate queries eventually
    - If you wan to know detail of the method, please read webui.py :)
- LSI is mainly used for Covering Tagging Presision
  - Simple search logic can be implemented with BM25 only
  - But, you can use tags to search which are difficult for tagging because the index data which is composed of vectors is applyed LSI
    - implemented with Gensim lib
- ( Web UI is implemented with StreamLit )

## Usage
- (collect working confirmed environment)
  - (Windows 11 Pro 64bit 23H2)
  - Python 3.10.4
  - pip 22.0.4
- $ pip install -r requirements.txt 
- $ python tagging.py --dir "IMAGE FILES CONTAINED DIR PATH"
  - The script searches directory structure recursively :)
  - This takes quite a while...
    - About 1.7 sec/file at middle spec desktop PC (GPU is not used)
      - AMD Ryzen 7 5700X 8-Core Processor 4.50 GHz
    - You may speed up with editing the script to use CUDAExecutionProvider, CoreMLExecutionProvider and etc :)
      - Plese see [here](https://onnxruntime.ai/docs/execution-providers/)
      - Performance key is processing speed of ONNX Runtime at your machine :)
  - Image files and tags of these are saved to tags-wd-tagger.txt
- $ python counttag.py
  - => for deciding appropriate dimension scale fitting your data
  - unique tag count is shown
- $ python genmodel.py --dim MODEL_DIMENSION
  - MODEL_DIMENSION is integer which specify dimension of latent sementic representation
    - Dimension after applying LSI
  - I think that 80% of unique tags which is counted with counttag.py is better
    - EX: unique tags count is 1000 -> 0.8 * 1000 -> 800 (dimension)
  - This takes quite a while...
    - LSI processing: dimension reduction with [Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition)
    - Take several secs only for 1000 files and reduction from 800 to 700 dimension case (case of demo on later section)
    - But, in 340k files and from 7500 to 6000 dimension case, about 3.5 hour are taken
      - files are not for demo :)
- $ streamlit run webui.py
  - Search app is opend on your web browser

## Usage (Binary Package of Windows at Release Page)
- Same with above except that you need not to execute python and execution path (current path) is little bit different :)
- First, unzip package and launch command prompt or PowerShell :)
- $ cd anime-illust-image-searcher-pkg
- $ .\cmd_run\cmd_run.exe tagging --dir "IMAGE FILES CONTAINED DIR PATH"
- $ .\cmd_run\cmd_run.exe counttag
  - Confirm unique tag num
- $ .\cmd_run\cmd_run.exe genmodel --dim MODEL_DIMENSION
  - Same with above :)
- $ .\run_webui.exe
  - Search app is opend on your web browser!

## Tips (Attention)
- Words (tags) which were not apeeared at tagging are not usable on query
  - Solution
    - Search words you want to use from taggs-wd-tagger.txt with grep, editor or something for existance checking
    - If exist, there is no problem. If not, you should think similar words and search it in same manner :)
- **Specifying Eath Tag Weight (format -> TAG:WEIGHT, WEIGHT shoud be integer)**
  - Example
    - "girl:3 dragon"
    - "girl:2 boy:3"
    - "girl dragon:2 boy:-3"
      - **Negative weight also can be specified!**
- **Search Result Exporting feature**
  - You can export file paths list which is hitted at search
  - Pressing 'Export' button saves the list as text file to path Web UI executed at
  - File name is query text with timestamp and contents is line break delimited
    - Some viewer tools such as [Irfan View](https://www.irfanview.com/) can load image files with passing a text file contains path list :)
    - Irfan View can slideshow also. It's nice :)
  - At Windows, charactor code is sjis. At other OSes, charactor code is utf-8
- Character code of file pathes 
  - If file path contains charactors which can't be converted to Unicode or utf-8, scripts may ouput error message at processing the file
  - But, it doesn't mean that your script usage is wrong. Though these files is ignored or not displayed at Web UI :|
    - This is problem of current implentation. When you use scripts on Windows and charactor code of directory/file names isn't utf-8, the problem may occur

## Information Related to Copyrights
- I used [this code](https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py) as reference wheh implmenting tagger script
- ["WD EVA02-Large Tagger v3" model](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) is used for image file tagging
- **I thank to great works of SmilingWolf**

## TODO
- [ ] <del>Search on latent representation generated by CLIP model</del>
  - **This method was alredy tried but precition was not good because current public available CLIP models are not fitting for anime style illust :|**
    - If CLIP models which are fine tuned with anime style illust images are available, this method is better than current one
- [x] Weight specifying to tags like prompt format of Stable Diffusion Web UI
  - Current implemenataion uses all tags faialy. But there is many cases that users want to emphasize specific tags and can't get appropriate results without that!
- [ ] Fix a bug: some type of tags on tags-wd-tagger.txt can't be used on query 
- [ ] Incremental index updating at image files increasing
- [ ] Similar image search with specifying a image file 
- [x] Exporting found files list feature
  - In text file. Once you get list, many other tools and viewer you like can be used :)
- [x] Making binary package of this app which doesn't need python environment building


## Screenshots of Demo
- I used about 1000 image files collected from [Irasutoya](https://www.irasutoya.com/) which offers free image materials as search target example
  - Note: image materials of Irasutoya have restrictions at commercial purposes use
- Partial tagging result: [./tagging_example.txt](/tagging_example.txt)
  - Generation script was executed in Windows
  - File paths in linked file have been partially masked 


- Search "standing"
  - ![image](https://github.com/user-attachments/assets/6e324a1e-ae49-40de-9dbd-d040e153b837)
- Search "standing animal"
  - ![image](https://github.com/user-attachments/assets/cd2862e3-e7e2-42fe-b830-705c778e10b8)
- Image info page
  - ![image](https://github.com/user-attachments/assets/78898162-ac6d-4fdf-b806-798f2f52a8d5)
- Slideshow feature
  - Auto slide in 5 sec period (roop)
  - ![image](https://github.com/user-attachments/assets/ea42336f-6b59-402b-b19e-f10610e4b200)
  



