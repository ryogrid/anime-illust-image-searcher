from gensim.models.lsimodel import LsiModel

from gensim.utils import simple_preprocess
from gensim.similarities import MatrixSimilarity
import pickle

import argparse
import streamlit as st

# $ streamlit run web_ui_image_search_lsi.py

ss = st.session_state
search_tags = ''
image_files_name_tags_arr = []
args = None
model = None
index = None
dictionary = None

NG_WORDS = ['language', 'english_text', 'pixcel_art']

def find_similar_documents(model, new_doc, topn=50):
    # vectorize the query
    query_bow = dictionary.doc2bow(simple_preprocess(new_doc))
    query_lsi = model[query_bow]

    # calculate similarities
    sims = index[query_lsi]

    # sort results, decide better result size and etc
    sims = sorted(enumerate(sims), key=lambda item: -item[1])    
    sims_filtered = [x for x in sims if x[1] > 0.1]
    if len(sims_filtered) > 30:
        sims = sims_filtered
    else:
        sims = sims[:30]

    ret_len = topn
    if ret_len > len(sims):
        ret_len = len(sims)    
    # for doc_position, score in sims[:ret_len]:
    #     print(f"Similarity of image {doc_position} : {score}\n{image_files_name_tags_arr[doc_position]}\n")
    return sims[:ret_len]

def is_include_ng_word(tags):
    for ng_word in NG_WORDS:
        if ng_word in tags:
            return True
    return False

def init_session_state(data=[]):
    global ss
    if 'data' not in ss:
        ss['data'] = []

    if len(data) > 0:
        ss['data'] = data
        ss['page_index'] = 0
        return

    if 'page_index' not in ss:
        ss['page_index'] = 0    

def update_index(session_key, num, max_val=None):
    global ss

    if max_val:
        if ss[session_key] < max_val - num:
            ss[session_key] += num
    else:
        if ss[session_key] >= num:
            ss[session_key] -= num

def convert_data_structure(file_pathes):
    pages = []
    rows = []
    cols = []

    for ii in range(len(file_pathes)):
        cols.append(file_pathes[ii])
        if len(cols) >= 6:
            rows.append(cols)
            cols = []
        if len(rows) >= 5:
            pages.append(rows)
            rows = []

    return pages

def display_images():
    global ss

    if len(ss['data']) > 0:
        for data_per_page in ss['data'][ss['page_index']]:
            for col_index, col_ph in enumerate(st.columns(6)):
                try:
                    col_ph.image(data_per_page[col_index])
                except Exception as e:
                    print(f'Error: {e}')
                    continue

def pagination():
    col1, col2, col3 = st.columns([1, 8, 1])
    col1.button('prev' * 1, on_click=update_index, args=('page_index', 1))
    col3.button('next' * 1, on_click=update_index, args=('page_index', 1, len(ss['data'])))
    col2.markdown(
        f'''
        <div style='text-align: center;'>
            {ss['page_index'] + 1} / {len(ss['data'])}
        </div>
        ''',
        unsafe_allow_html=True,
    )

def show_search_result():
    global image_files_name_tags_arr
    global args

    load_model()
    similar_docs = find_similar_documents(model, search_tags, topn=2000) 

    found_docs_pathes = []
    for doc_id, similarity in similar_docs:
        try:
            found_img_info_splited = image_files_name_tags_arr[doc_id].split(',')
            if is_include_ng_word(found_img_info_splited):
                continue
            print(f'Image ID: {doc_id}, Similarity: {similarity}, Tags: {image_files_name_tags_arr[doc_id]}')
            found_fpath = found_img_info_splited[0]
            if args.rep:
                found_fpath = found_fpath.replace(args.rep[0], args.rep[1])
            found_docs_pathes.append(found_fpath)
        except Exception as e:
            print(f'Error: {e}')
            continue    
    
    pages = convert_data_structure(found_docs_pathes)
    init_session_state(pages)

def load_model():
    global model
    global image_files_name_tags_arr
    global index
    global dictionary

    # load index text file to show image
    tag_file_path = 'tags-wd-tagger_lsi_idx.csv'
    image_files_name_tags_arr = []
    with open(tag_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            image_files_name_tags_arr.append(line.strip())

    # load model and etc
    model = LsiModel.load("lsi_model")
    index = MatrixSimilarity.load("lsi_index")
    dictionary = pickle.load(open("lsi_dictionary", "rb"))

def main():
    global search_tags
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', nargs=2, required=False, help='replace the string in file path to one you want')
    args = parser.parse_args()

    init_session_state()

    # input form
    search_tags = st.text_input('search tags', value='')
    st.button('Search', on_click=show_search_result)

    display_images()
    pagination()

main()