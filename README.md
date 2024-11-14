# Anime Style Illustration Specific Image Search App with ViT Tagger x BM25/Doc2Vec
![image](https://github.com/user-attachments/assets/3b95b3b4-db6d-483f-8bd1-8d2203c16792)

## What's This?
- Anime Style Illustration Specific Image Search App with ML Technique
  - can be used for photos. but flexible photo search is offered by Google Photos or etc :)
- Search capabilities of cloud photo album services towards illustration image files are poor for some reason
- So, I wrote simple scripts

## Method
- Search Images Matching with Query Texts on Latent Semantic Representation Vector Space and with BM25
  - Vectors are generated with embedding model: Tagger Using Visual Transformar (ViT) Internally x Doc2Vec
  - Scores which is calculated with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) is used in combination
  - Internal re-ranking method is also introduced
    - Assumption: Users make queries better asymptotically according to top search results and find appropriate queries eventually
    - If you wan to know detail of the method, please read webui.py :)
- Doc2Vec is Mainly Used for Covering Tagging Presision
  - Simple search logic can be implemented with BM25 only
  - But, you can use tags to search which are difficult for tagging because the index data which is composed of vectors generated with Doc2Vec model
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
    - You may speed up with setup libraries and drivers for using GPU :)
      - Plese see [here](https://pytorch.org/get-started/previous-versions/#v241)
        - Current pytorch version of this repo is v2.4.1
      - You should install pytorch package supporting CUDA matching wich CUDA library on your machine additionaly. And cuDNN library matching wich the CUDA library should be installed also :)
        - Example of pytorch install command line: $ pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
        - If your graphic board is not supported by CUDA 12.1.x library, you should change version of torchXXXXX packages
  - Pathes and tags of image files are saved to tags-wd-tagger.txt
- $ python genmodel.py
  - This takes quite a while...
- $ streamlit run webui.py
  - Search app is opend on your web browser

## Index Data Updating
- When you get and store new image files, you should update index data for adding the files to be hitted at search on webui.py
- Procedure
  - 1 Backup all files genarated by scripts on this repo!
    - Model files on your home directory is exception :)
  - 2 $ python --dir "IMAGE FILES CONTAINED DIR PATH" **--after "YYYY-MM-DD"**
    - Param of --dir doesn't have to be changed
    - Adding --after option is needed. Please specify date after last index data creation or update
      - Tagging target is filtered by specified date: added date (cdate attribute) <= YYYY-MM-DD
  - 3 $ python genmodel.py --update
  - Thats's all!

## Usage (Binary Package of Windows at Release Page)
- Same with above except that you need not to execute python and execution path (current path) is little bit different :)
- First, unzip package and launch command prompt or PowerShell :)
- $ cd anime-illust-image-searcher-pkg
- $ .\cmd_run\cmd_run.exe tagging --dir "IMAGE FILES CONTAINED DIR PATH"
- $ .\cmd_run\cmd_run.exe genmodel
  - Same with above :)
- $ .\run_webui.exe
  - Search app is opend on your web browser!

## Tips (Attention)
- Words (tags) which were not apeeared at tagging are not usable on query
  - Solution
    - Search words you want to use from taggs-wd-tagger.txt with grep, editor or something for existance checking
    - If exist, there is no problem. If not, you should think similar words and search it in same manner :)
- **Specifying Eath Tag Weight (format -> TAG:WEIGHT, WEIGHT shoud be integer)**
  - Examples
    - "girl:3 dragon"
    - "girl:2 boy:3"
  - **Exclude tag marking**
    - **Weight specification which starts with '-' indicates that images tagged it should be excluded**
    - **ex: "girl boy:-3"**
      - **Images tagged 'boy' are removed from results. Numerical weight value is ignored but can't be omitted :)** 
  - **Required tag marking**
    - **Weight specification which starts with '+' indicates the tag is required**
    - **ex: "girl:+3 dragon"**
      - **Images not tagged 'girl' are removed from results**
      - **Weight value is NOT ignored at calculation of scores**
- **Search Result Exporting feature**
  - You can export file paths list which is hitted at search
  - Pressing 'Export' button saves the list as text file to path Web UI executed at
  - File name is query text with timestamp and contents is line break delimited
    - Some viewer tools such as [Irfan View](https://www.irfanview.com/) can load image files with passing a text file contains path list :)
    - Irfan View can slideshow also. It's nice :)
  - At Windows, charactor code is sjis. At other OSes, charactor code is utf-8
- Character code of file pathes 
  - If file path contains characters which can't be converted to Unicode or utf-8, scripts may ouput error message at processing the file
  - But, it doesn't mean that your script usage is wrong. Though these files is ignored or not displayed at Web UI :|
    - This is problem of current implentation. When you use scripts on Windows and charactor code of directory/file names isn't utf-8, the problem may occur

## Information Related to Copyrights
- Tagger
  - [this code](https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py) was used as reference wheh implmenting tagger script
  - ["WD EVA02-Large Tagger v3" model](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3) is used for image file tagging
  - **I thank to great works of SmilingWolf**
- Character visual similarity calculation
  - [this code](https://huggingface.co/spaces/deepghs/ccip/blob/f7d50a4f5dd3d4681984187308d70839ff0d3f5b/ccip.py) was used as reference when implemnting model execution
  - [Quantized CCIP(Contrastive Anime Character Image Pre-Training) model](https://huggingface.co/deepghs/ccip_onnx) is used
    - **(This metrics based reranking mode exists. But usage is not wrote here yet...)**  
  - **I thank to great works of deepghs community members**

## For Busy People
- **Tagging using Google Colab env !**
  - 1 Make preprocessed data with [utility/make_tensor_files.py](./utility/make_tensor_files.py)
  - 2 Zip the output dir
  - 3 Upload zipped file to Google Drive
  - 4 Use Google Colab env like [this](https://github.com/ryogrid/ryogridJupyterNotebooks/blob/master/tagging_colab-241104-T4-with-Tensor-files.ipynb)
  - 5 Get tags-wd-tagger.txt and replace file pathes on it to be matched with your image files existing pathes :)
  - 6 Execute genmodel.py !

## TODO
- [ ] <del>Search on latent representation generated by CLIP model</del>
  - **This method was alredy tried but precition was not good because current public available CLIP models are not fitting for anime style illust :|**
    - If CLIP models which are fine tuned with anime style illust images are available, this method is better than current one
- [x] Weight specifying to tags like prompt format of Stable Diffusion Web UI
  - Current implemenataion uses all tags faialy. But there is many cases that users want to emphasize specific tags and can't get appropriate results without that!
- [x] Fix a bug: some type of tags on tags-wd-tagger.txt can't be used on query 
- [x] Incremental index updating at image files increasing
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
  



