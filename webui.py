import math
import sys

from gensim import corpora
from gensim.models import Doc2Vec
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
model: Optional[Doc2Vec] = None
index: Optional[MatrixSimilarity] = None
dictionary: Optional[corpora.Dictionary] = None

NG_WORDS: List[str] = ['language', 'english_text', 'pixcel_art']

class Arguments(Protocol):
    rep: List[str]

args: Optional[Arguments] = None

# BM25 global variables
bm25_corpus: List[Dict[int, int]] = []
bm25_doc_lengths: Optional[ndarray] = None
bm25_avgdl: float = 0
bm25_idf: Dict[int, float] = {}
bm25_D: int = 0  # Total number of documents
BM25_WEIGHT: float = 0.5  # BM25 weight (modifiable)
DOC2VEC_WEIGHT: float = 0.5  # Doc2Vec weight (modifiable)

# weight at reranking
ORIGINAL_SCORE_WEIGHT: float = 0.7
RERANKED_SCORE_WEIGHT: float = 0.3

DIFF_FILTER_THRESH = 1e-6 #0.000001

# sorted_scores: sorted_scores[N] >= sorted_scores[N+1]
def filter_searched_result(sorted_scores: List[Tuple[int, float]]) -> List[Tuple[int,float]]:
    scores: List[float] = [sorted_scores[i][1] for i in range(len(sorted_scores))]
    scores_ndarr: ndarray = np.array(scores)
    diff_arr: ndarray = scores_ndarr[:-1] - scores_ndarr[1:]
    # find the index of the first diff value element that is less than the threshold
    diff_arr = np.where(diff_arr == 0, np.inf, diff_arr)
    # look second point for reliability
    t: float = np.where(diff_arr < DIFF_FILTER_THRESH)[0][1]
    max_val = scores_ndarr.max()

    return [(sorted_scores[idx][0], sorted_scores[idx][1] / float(max_val)) for idx in range(int(t))]

def normalize_and_apply_weight_doc2vec(new_doc: str) -> List[Tuple[int, float]]:
    tags: List[str] = new_doc.split(" ")

    # parse tag:weight format
    tag_and_weight_list: List[Tuple[str, int]] = []
    all_weight: int = 0
    for tag in tags:
        tag_splited: List[str] = tag.split(":")
        if len(tag_splited) >= 2 and tag_splited[-1].isdigit():
            # replace is for specific type of tags
            tag_elem: str = ':'.join(tag_splited[0:len(tag_splited) - 1]).replace('\(', '(').replace('\)', ')')
            tag_and_weight_list.append((tag_elem.replace('(', '\(').replace(')', '\)'), int(tag_splited[-1])))
            all_weight += int(tag_splited[-1])
        else:
            # replace is for specific type of tags
            tag_elem: str = ':'.join(tag_splited[0:len(tag_splited)]).replace('\(', '(').replace('\)', ')')
            tag_and_weight_list.append((tag_elem.replace('(', '\(').replace(')', '\)'), 1))
            all_weight += 1

    if all_weight == 0:
        all_weight = 1

    got_vector: ndarray = np.zeros(len(model.dv[0]))
    for tag, weight in tag_and_weight_list:
        tmp_vec: ndarray = model.infer_vector([tag])
        tmp_vec = tmp_vec / np.linalg.norm(tmp_vec)
        got_vector += weight * tmp_vec
    got_vector = got_vector / all_weight
    norm: float = np.linalg.norm(got_vector)

    if math.isinf(norm) or norm == 0:
        norm = 1.0

    got_vector = got_vector / norm

    return [(ii, val) for ii, val in enumerate(got_vector)]

def compute_bm25_scores(query_terms: List[str] = [], query_weights: Optional[Dict[int, float]] = None) -> ndarray:
    global bm25_corpus
    global bm25_doc_lengths
    global bm25_avgdl
    global bm25_idf
    global bm25_D

    k1 = 1.5  # BM25 parameter
    b = 0.75  # BM25 parameter

    # Convert query terms to term IDs
    if query_weights is not None:
        query_term_ids: List[int] = list(query_weights.keys())
    else:
        query_term_ids: List[int] = [dictionary.token2id.get(term, None) for term in query_terms if term in dictionary.token2id]
        query_term_ids = [term_id for term_id in query_term_ids if term_id is not None]

    # Initialize scores
    scores = np.zeros(bm25_D)

    for term_id in query_term_ids:
        idf: float = bm25_idf.get(term_id, 0)
        # Collect term frequencies for this term across all documents
        tfs: ndarray = np.array([doc.get(term_id, 0) for doc in bm25_corpus])
        dl = bm25_doc_lengths
        denom = tfs + k1 * (1 - b + b * (dl / bm25_avgdl))
        numer = tfs * (k1 + 1)
        score = idf * (numer / denom)

        # Apply query term weight
        if query_weights is not None:
            weight = query_weights.get(term_id, 1.0)
        else:
            weight = 1.0

        if weight < 0:
            # Exclude documents containing the term by setting the score to -inf
            exclude_doc_ids: List[int] = []
            for doc_idx, doc in enumerate(bm25_corpus):
                if term_id in doc:
                    exclude_doc_ids.append(doc_idx)
            scores[exclude_doc_ids] = -np.inf
        else:
            scores += weight * score

    return scores


def is_include_ng_word(tags: List[str]) -> bool:
    for ng_word in NG_WORDS:
        if ng_word in tags:
            return True
    return False

# return array([(doc_id, val), (doc_id, val), ...])
def get_embedded_vector_by_doc_id(doc_id: int) -> List[Tuple[int, float]]:
    tags: List[str] = image_files_name_tags_arr[doc_id - 1].split(',')[1:]
    #doc_bow: List[Tuple[int, int]] = dictionary.doc2bow(tags)
    embed_vec:ndarray = model.infer_vector(tags)
    doc_doc2vec: List[Tuple[int, float]] = [(ii, val) for ii, val in enumerate(embed_vec)]
    return doc_doc2vec

def find_similar_documents(new_doc: str, topn: int = 50) -> List[Tuple[int, float]]:
    # get embed vector using Doc2Vec model
    vec_doc2vec: List[Tuple[int, float]] = normalize_and_apply_weight_doc2vec(new_doc)

    # Existing similarity scores using Dod2Vec model
    sims_doc2vec: ndarray = index[vec_doc2vec]

    splited_term = [x for x in new_doc.split(' ')]
    query_term_and_weight: Dict[int, float] = {}
    for term in splited_term:
        term_splited: List[str] = term.split(':')
        if len(term_splited) >= 2 and term_splited[-1].isdigit():
            query_term_and_weight[dictionary.token2id[':'.join(term_splited[0:len(term_splited) - 1])]] = int(term_splited[-1])
        else:
            query_term_and_weight[dictionary.token2id[':'.join(term_splited[0:len(term_splited)])]] = 1

    # BM25 scores
    bm25_scores = compute_bm25_scores(query_weights=query_term_and_weight)

    # Normalize scores
    if sims_doc2vec.max() > 0:
        sims_doc2vec = sims_doc2vec / sims_doc2vec.max()
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # Combine scores
    final_scores = BM25_WEIGHT * bm25_scores + DOC2VEC_WEIGHT * sims_doc2vec

    # Get top documents
    sims: List[Tuple[int, float]] = list(enumerate(final_scores))
    sims = sorted(sims, key=lambda item: -item[1])

    if len(sims) > 10:
        # Perform rescoring
        top10_sims = sims[:10]  # Top 10 documents
        top10_doc_ids: List[int] = [doc_id for doc_id, _ in top10_sims]
        top10_doc_ids_set = set(top10_doc_ids)
        top10_doc_vectors: List[List[Tuple[int, float]]] = [get_embedded_vector_by_doc_id(doc_id + 1) for doc_id in top10_doc_ids]
        weighted_mean_vec: ndarray = np.average(top10_doc_vectors, axis=0, weights=[score for _, score in top10_sims])
        weighted_mean_vec = weighted_mean_vec / np.linalg.norm(weighted_mean_vec)
        weighted_mean_vec_with_docid: List[Tuple[int, float]] = [(round(docid), val) for docid, val in weighted_mean_vec.tolist()]

        reranked_scores: ndarray = index[weighted_mean_vec_with_docid]

        # ensenble original score and rescored score
        reranked_final_scores = ORIGINAL_SCORE_WEIGHT * final_scores + RERANKED_SCORE_WEIGHT * reranked_scores

        if reranked_final_scores.max() > 0:
            reranked_final_scores = reranked_final_scores / reranked_final_scores.max()

        # Convert reranked scores to list
        reranked_sims_list = list(enumerate(reranked_final_scores))

        # make top 10 documents execluded list
        reranked_sims_list = [item for item in reranked_sims_list if item[0] not in top10_doc_ids_set]

        # Set the scores of top 10 documents to 1
        final_sims = []
        for idx, (doc_id, _) in enumerate(top10_sims):
            final_sims.append((doc_id, 1.0))

        # Add remaining documents
        final_sims.extend(reranked_sims_list)

        # Define custom sorting key
        def sorting_key(item):
            doc_id, score = item
            if doc_id in top10_doc_ids_set:
                idx = top10_doc_ids.index(doc_id)
                return (-2, idx)  # Maintain order for top 10
            else:
                return (-1, -score)  # Sort remaining by score

        # Sort
        final_sims = sorted(final_sims, key=sorting_key)

        # Apply threshold filtering
        final_sims = filter_searched_result(final_sims)

        # Return results
        ret_len = topn
        if ret_len > len(final_sims):
            ret_len = len(final_sims)
        return final_sims[:ret_len]

    else:
        # Apply threshold filtering
        sims = filter_searched_result(sims)
        ret_len: int = topn
        if ret_len > len(sims):
            ret_len = len(sims)
        return sims[:ret_len]


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

def export_result_to_file() -> None:
    if sys.platform == 'win32':
        encoding = 'shift_jis'
    else:
        encoding = 'utf-8'

    output_file_path: str = f"{search_tags.replace(' ', '_').replace(':', '_') }_{int(time.time())}.txt"

    with open(output_file_path, 'w', encoding=encoding) as f:
        for page in ss['data']:
            for row in page:
                for image_info in row:
                    try:
                        f.write(f"{image_info['file_path']}\n")
                    except Exception as e:
                        print(f'Error: {e}')
                        continue

def display_images() -> None:
    global ss

    if 'data' in ss and len(ss['data']) > 0:
        cols: Any = st.columns([10])
        with cols[0]:
            if st.button('Slideshow'):
                ss['slideshow_active'] = True
                ss['slideshow_index'] = 0
                st.rerun()
            if st.button('Export'):
                export_result_to_file()
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
        try:
            st.image(image_info['file_path'], use_column_width=True)
        except Exception as e:
            print(f'Error: {e}')
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
    similar_docs: List[Tuple[int, float]] = find_similar_documents(search_tags, topn=800)

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
    # BM25 variables
    global bm25_corpus
    global bm25_doc_lengths
    global bm25_avgdl
    global bm25_idf
    global bm25_D

    tag_file_path: str = 'tags-wd-tagger_doc2vec_idx.csv'
    image_files_name_tags_arr = []
    with open(tag_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            image_files_name_tags_arr.append(line.strip())

    model = Doc2Vec.load("doc2vec_model")

    index = MatrixSimilarity.load("doc2vec_index")
    dictionary = pickle.load(open("doc2vec_dictionary", "rb"))

    if 'bm25_corpus' in ss:
        bm25_corpus = ss['bm25_corpus']
        bm25_doc_lengths = ss['bm25_doc_lengths']
        bm25_avgdl = ss['bm25_avgdl']
        bm25_idf = ss['bm25_idf']
        bm25_D = ss['bm25_D']
    else:
        ss['bm25_corpus'] = pickle.load(open("bm25_corpus", "rb"))
        ss['bm25_doc_lengths'] = pickle.load(open("bm25_doc_lengths", "rb"))
        ss['bm25_avgdl'] = pickle.load(open("bm25_avgdl", "rb"))
        ss['bm25_idf'] = pickle.load(open("bm25_idf", "rb"))
        ss['bm25_D'] = pickle.load(open("bm25_D", "rb"))
        bm25_corpus = ss['bm25_corpus']
        bm25_doc_lengths = ss['bm25_doc_lengths']
        bm25_avgdl = ss['bm25_avgdl']
        bm25_idf = ss['bm25_idf']
        bm25_D = ss['bm25_D']

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