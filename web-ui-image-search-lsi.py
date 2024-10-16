from gensim.models.lsimodel import LsiModel
from gensim.utils import simple_preprocess
from gensim.similarities import MatrixSimilarity
from streamlit.runtime.state import SessionStateProxy
import pickle

import argparse
import streamlit as st
import time
from typing import List, Tuple, Dict, Any, Optional, Callable, Protocol

# $ streamlit run web-ui-image-search-lsi.py

ss: SessionStateProxy = st.session_state
search_tags: str = ''
image_files_name_tags_arr: List[str] = []
model: Optional[LsiModel] = None
index: Optional[MatrixSimilarity] = None
dictionary: Optional[Any] = None

NG_WORDS: List[str] = ['language', 'english_text', 'pixcel_art']

class Arguments(Protocol):
    rep: List[str]

args: Optional[Arguments] = None

def find_similar_documents(model: LsiModel, new_doc: str, topn: int = 50) -> List[Tuple[int, float]]:
    query_bow: List[Tuple[int, int]] = dictionary.doc2bow(simple_preprocess(new_doc))
    query_lsi: List[Tuple[int, float]] = model[query_bow]

    sims: List[Tuple[int, float]] = index[query_lsi]

    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    sims = [x for x in sims if x[1] > 0.1]

    ret_len: int = topn
    if ret_len > len(sims):
        ret_len = len(sims)
    return sims[:ret_len]

def is_include_ng_word(tags: List[str]) -> bool:
    for ng_word in NG_WORDS:
        if ng_word in tags:
            return True
    return False

def init_session_state(data: List[Any] = []) -> None:
    global ss
    if 'data' not in ss:
        ss['data'] = []
        ss['last_search_tags'] = ''
    if 'selected_image_info' not in ss:
        ss['selected_image_info'] = None
    if len(data) > 0:
        ss['data'] = data
        ss['page_index'] = 0
        return

    if 'page_index' not in ss:
        ss['page_index'] = 0

def update_index(session_key: str, num: int, max_val: Optional[int] = None) -> None:
    global ss

    if max_val:
        if ss[session_key] < max_val - num:
            ss[session_key] += num
            st.rerun()
    else:
        if ss[session_key] >= -num:
            ss[session_key] += num
            st.rerun()

def convert_data_structure(image_info_list: List[Dict[str, Any]]) -> List[List[List[Dict[str, Any]]]]:
    pages: List[List[List[Dict[str, Any]]]] = []
    rows: List[List[Dict[str, Any]]] = []
    cols: List[Dict[str, Any]] = []

    for ii in range(len(image_info_list)):
        cols.append(image_info_list[ii])
        if len(cols) >= 5:
            rows.append(cols)
            cols = []
        if len(rows) >= 5:
            pages.append(rows)
            rows = []

    if cols:
        rows.append(cols)
    if rows:
        pages.append(rows)

    return pages

def get_all_images() -> List[str]:
    images: List[str] = []
    for page in ss['data']:
        for row in page:
            for image_info in row:
                images.append(image_info['file_path'])
    return images

def slideshow() -> None:
    images: List[str] = get_all_images()
    if len(images) == 0:
        st.write("No images to display in slideshow.")
        ss['slideshow_active'] = False
        st.rerun()
    if 'slideshow_index' not in ss:
        ss['slideshow_index'] = 0
    cols: Any = st.columns([1])

    try:
        cols[0].image(images[ss['slideshow_index']], use_column_width=True)
    except Exception as e:
        print(f'Error: {e}')
        ss['slideshow_index'] = (ss['slideshow_index'] + 1) % len(images)
        st.rerun()

    if st.button('Stop'):
        ss['slideshow_active'] = False
        ss['slideshow_index'] = 0
        ss['text_input'] = ss['last_search_tags']
    else:
        time.sleep(5)
        ss['slideshow_index'] = (ss['slideshow_index'] + 1) % len(images)
    st.rerun()

def is_now_slideshow() -> bool:
    return 'slideshow_active' in ss and ss['slideshow_active']

def display_images() -> None:
    global ss

    if 'data' in ss and len(ss['data']) > 0:
        cols: Any = st.columns([10])
        with cols[0]:
            if st.button('Slideshow'):
                ss['slideshow_active'] = True
                ss['slideshow_index'] = 0
                st.rerun()

            for data_per_page in ss['data'][ss['page_index']]:
                cols = st.columns(5)
                for col_index, col_ph in enumerate(cols):
                    try:
                        image_info: Dict[str, Any] = data_per_page[col_index]
                        key: str = f"img_{ss['page_index']}_{image_info['doc_id']}_{col_index}"
                        if col_ph.button('info', key=key):
                            ss['selected_image_info'] = image_info
                            st.rerun()
                        col_ph.image(image_info['file_path'], use_column_width=True)
                    except Exception as e:
                        print(f'Error: {e}')
                        continue
            pagination()

def pagination() -> None:
    col1, col2, col3 = st.columns([2, 8, 2])
    if col1.button('Prev'):
        update_index('page_index', -1)
    if col3.button('Next'):
        update_index('page_index', 1, len(ss['data']))
    col2.markdown(
        f'''
        <div style='text-align: center;'>
            {ss['page_index'] + 1} / {len(ss['data'])}
        </div>
        ''',
        unsafe_allow_html=True,
    )

def display_selected_image() -> None:
    global ss
    image_info: Dict[str, Any] = ss['selected_image_info']
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image(image_info['file_path'], use_column_width=True)
    with col2:
        st.write("Matching Score:")
        st.write("{:.2f}%".format(image_info['similarity'] * 100))
        st.write("File Path:")
        st.code(image_info['file_path'])
        st.write("Tags:")
        st.write('  \n'.join(image_info['tags']))
    if st.button('Close'):
        ss['selected_image_info'] = None
        ss['text_input'] = ss['last_search_tags']
        st.rerun()

def show_search_result() -> None:
    global image_files_name_tags_arr
    global args

    load_model()
    similar_docs: List[Tuple[int, float]] = find_similar_documents(model, search_tags, topn=2000)

    found_docs_info: List[Dict[str, Any]] = []
    for doc_id, similarity in similar_docs:
        try:
            found_img_info_splited: List[str] = image_files_name_tags_arr[doc_id].split(',')
            if is_include_ng_word(found_img_info_splited):
                continue
            found_fpath: str = found_img_info_splited[0]
            if args is not None and args.rep:
                found_fpath = found_fpath.replace(args.rep[0], args.rep[1])
            found_docs_info.append({
                'file_path': found_fpath,
                'doc_id': doc_id,
                'similarity': similarity,
                'tags': found_img_info_splited[1:]
            })
        except Exception as e:
            print(f'Error: {e}')
            continue

    pages: List[List[List[Dict[str, Any]]]] = convert_data_structure(found_docs_info)
    init_session_state(pages)

def load_model() -> None:
    global model
    global image_files_name_tags_arr
    global index
    global dictionary

    tag_file_path: str = 'tags-wd-tagger_lsi_idx.csv'
    image_files_name_tags_arr = []
    with open(tag_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            image_files_name_tags_arr.append(line.strip())

    model = LsiModel.load("lsi_model")
    index = MatrixSimilarity.load("lsi_index")
    dictionary = pickle.load(open("lsi_dictionary", "rb"))

def main() -> None:
    global search_tags
    global args
    global ss

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--rep', nargs=2, required=False, help='replace the string in file path to one you want')
    args = parser.parse_args()

    init_session_state()

    if is_now_slideshow():
        slideshow()
    else:
        if 'selected_image_info' in ss and ss['selected_image_info']:
            display_selected_image()
        else:
            search_tags = st.text_input('Enter search tags', value='', key='text_input')
            if search_tags and ss['last_search_tags'] != search_tags:
                ss['last_search_tags'] = search_tags
                show_search_result()
                st.rerun()
            display_images()

main()