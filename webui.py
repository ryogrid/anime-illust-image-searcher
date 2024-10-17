from gensim.models.lsimodel import LsiModel
from gensim.similarities import MatrixSimilarity
from numpy import ndarray
from streamlit.runtime.state import SessionStateProxy
import pickle

import numpy as np
import argparse
import streamlit as st
import time
from typing import List, Tuple, Dict, Any, Optional, Protocol

# $ streamlit run webui.py

ss: SessionStateProxy = st.session_state
search_tags: str = ''
image_files_name_tags_arr: List[str] = []
model: Optional[LsiModel] = None
index: Optional[MatrixSimilarity] = None
dictionary: Optional[Any] = None

SIMILARITY_THRESHOLD: float = 0.1

NG_WORDS: List[str] = ['language', 'english_text', 'pixcel_art']

class Arguments(Protocol):
    rep: List[str]

args: Optional[Arguments] = None

# sorted_scores: sorted_scores[N] >= sorted_scores[N+1]
def filter_searched_result(sorted_scores: List[Tuple[int, float]]) -> List[Tuple[int,float]]:
    # sorted_scores: Any = scores[scores.argsort()[:-1]]
    # difs: ndarray = sorted_scores[:-1] - sorted_scores[1:]
    scores: List[float] = [sorted_scores[i][1] for i in range(len(sorted_scores))]
    scores_ndarr: ndarray = np.array(scores)
    max_val = scores_ndarr.max()
    scores_ndarr = scores_ndarr / max_val
    idxes_ndarr = np.where(scores_ndarr > SIMILARITY_THRESHOLD)

    return [(sorted_scores[idx][0], sorted_scores[idx][1] / float(max_val)) for idx in idxes_ndarr[0]]

# # sorted_scores: sorted_scores[N] >= sorted_scores[N+1]
# def mcut_threshold(sorted_scores: List[Tuple[int, float]]) -> float:
#     """
#     Maximum Cut Thresholding (MCut)
#     Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
#     for Multi-label Classification. In 11th International Symposium, IDA 2012
#     (pp. 172-183).
#     """
#     # sorted_scores: Any = scores[scores.argsort()[:-1]]
#     # difs: ndarray = sorted_scores[:-1] - sorted_scores[1:]
#     difs: List[float] = [sorted_scores[i + 1][1] - sorted_scores[i][1] for i in range(len(sorted_scores) - 1)]
#     tmp_list : List[float] = []
#     # Replace 0 with -inf (same image files exist case)
#     for idx, val in enumerate(difs):
#         if val == 0:
#             tmp_list.append(-np.inf)
#         else:
#             tmp_list.append(val)
#     difs_ndarr: ndarray = np.array(difs)
#
#     t: signedinteger = difs_ndarr.argmax()
#     thresh: float = (sorted_scores[t][1] + sorted_scores[t + 1][1]) / 2
#
#     # score should be >= thresh
#     return thresh

def normalize_and_apply_weight_lsi(query_bow: List[Tuple[int, int]], new_doc: str) -> List[Tuple[int, float]]:
    tags: List[str] = new_doc.split(" ")

    # parse tag:weight format
    is_exist_negative_weight: bool = False
    tag_and_weight_list: List[Tuple[str, int]] = []
    # all_weight: int = 0
    for tag in tags:
        tag_splited: List[str] = tag.split(":")
        if len(tag_splited) == 2:
            # replace is for specific type of tags
            tag_elem: str = tag_splited[0].replace('\\(', '(').replace('\\)', ')')
            tag_and_weight_list.append((tag_elem.replace('(', '\\(').replace(')', '\\)'), int(tag_splited[1])))
            # all_weight += int(tag_splited[1])
        else:
            # replace is for specific type of tags
            tag_elem: str = tag_splited[0].replace('\\(', '(').replace('\\)', ')')
            tag_and_weight_list.append((tag_elem.replace('(', '\\(').replace(')', '\\)'), 1))
            # all_weight += 1

    query_bow_local: List[Tuple[int, int]] = []
    # apply weight to query_bow
    for tag, weight in tag_and_weight_list:
        tag_id: int = dictionary.token2id[tag]
        for ii, _ in enumerate(query_bow):
            if query_bow[ii][0] == tag_id:
                if weight >= 1:
                    query_bow_local.append((query_bow[ii][0], query_bow[ii][1]*weight))
                elif weight < 0:
                    # ignore this elem weight here
                    query_bow_local.append((query_bow[ii][0], 0))
                    is_exist_negative_weight = True

                break

    query_lsi: List[Tuple[int, float]] = model[query_bow_local]
    # query_lsi: List[Tuple[int, float]] = model.__getitem__(query_bow_local, scaled=True)

    # reset
    query_bow_local = []

    if is_exist_negative_weight:
        for tag, weight in tag_and_weight_list:
            tag_id: int = dictionary.token2id[tag]
            for ii, _ in enumerate(query_bow):
                if query_bow[ii][0] == tag_id:
                    if weight >= 1:
                        query_bow_local.append((query_bow[ii][0], 0))
                    elif weight < 0:
                        # negative weighted tags value is changed to positive and multiplied by weight
                        query_bow_local.append((query_bow[ii][0], -1*weight))

                    break

        query_lsi_neg: List[Tuple[int, float]] = model[query_bow_local]
        # query_lsi_neg: List[Tuple[int, float]] = model.__getitem__(query_bow_local, scaled=True)

        # query_lsi - query_lsi_neg
        query_lsi_tmp: List[Tuple[int, float]] = []
        for ii, _ in query_lsi:
            query_lsi_tmp.append((query_lsi[ii][0], query_lsi[ii][1] - query_lsi_neg[ii][1]))
        query_lsi = query_lsi_tmp

    # # normalize query with tag num
    # if all_weight > 0:
    #     query_lsi = [(tag_id, tag_value / all_weight) for tag_id, tag_value in query_lsi]
    return query_lsi

def find_similar_documents(new_doc: str, topn: int = 50) -> List[Tuple[int, float]]:
    # when getting bow presentaton, weight description is removed
    # because without it, weighted tag is not found in the dictionary
    splited_doc = [x.split(":")[0] for x in new_doc.split(' ')]
    query_bow: List[Tuple[int, int]] = dictionary.doc2bow(splited_doc)

    query_lsi = normalize_and_apply_weight_lsi(query_bow, new_doc)
    #query_lsi: List[Tuple[int, float]] = model[query_bow]

    sims: List[Tuple[int, float]] = index[query_lsi]

    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # sims = [x for x in sims if x[1] > 0.01]

    # thresh = mcut_threshold(sims)
    # sims = [x for x in sims if x[1] >= thresh]

    sims = filter_searched_result(sims)

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
        # to Last
        if num == max_val:
            ss[session_key] = max_val - 1
            st.rerun()
        # Next
        if ss[session_key] < max_val - num:
            ss[session_key] += num
            st.rerun()
    else:
        # to Top
        if num == 0:
            ss[session_key] = 0
            st.rerun()
        # Prev
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
    col1, col2, col3, col4, col5 = st.columns([2, 2, 8, 2, 2])
    if col1.button('Top'):
        update_index('page_index', 0)
    if col2.button('Prev'):
        update_index('page_index', -1)
    if col4.button('Next'):
        update_index('page_index', 1, len(ss['data']))
    if col5.button('Last'):
        update_index('page_index', len(ss['data']), len(ss['data']))
    col3.markdown(
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
    similar_docs: List[Tuple[int, float]] = find_similar_documents(search_tags, topn=2000)

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