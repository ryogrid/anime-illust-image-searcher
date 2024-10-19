import sys

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

# BM25 global variables
bm25_corpus: List[Dict[int, int]] = []
bm25_doc_lengths: Optional[ndarray] = None
bm25_avgdl: float = 0
bm25_idf: Dict[int, float] = {}
bm25_D: int = 0  # Total number of documents
BM25_WEIGHT: float = 0.5  # BM25 weight (modifiable)
LSI_WEIGHT: float = 0.5  # LSI weight (modifiable)

# sorted_scores: sorted_scores[N] >= sorted_scores[N+1]
def filter_searched_result(sorted_scores: List[Tuple[int, float]]) -> List[Tuple[int,float]]:
    scores: List[float] = [sorted_scores[i][1] for i in range(len(sorted_scores))]
    scores_ndarr: ndarray = np.array(scores)
    max_val = scores_ndarr.max()
    scores_ndarr = scores_ndarr / max_val
    idxes_ndarr = np.where(scores_ndarr > SIMILARITY_THRESHOLD)

    return [(sorted_scores[idx][0], sorted_scores[idx][1] / float(max_val)) for idx in idxes_ndarr[0]]

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
            tag_elem: str = tag_splited[0].replace('\(', '(').replace('\)', ')')
            tag_and_weight_list.append((tag_elem.replace('(', '\(').replace(')', '\)'), int(tag_splited[1])))
            # all_weight += int(tag_splited[1])
        else:
            # replace is for specific type of tags
            tag_elem: str = tag_splited[0].replace('\(', '(').replace('\)', ')')
            tag_and_weight_list.append((tag_elem.replace('(', '\(').replace(')', '\)'), 1))


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

        # query_lsi - query_lsi_neg
        query_lsi_tmp: List[Tuple[int, float]] = []
        for ii in range(len(query_lsi)):
            index, value = query_lsi[ii]
            neg_value = query_lsi_neg[ii][1] if ii < len(query_lsi_neg) else 0
            query_lsi_tmp.append((index, value - neg_value))
        query_lsi = query_lsi_tmp

    return query_lsi

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
        query_term_ids = list(query_weights.keys())
    else:
        query_term_ids = [dictionary.token2id.get(term, None) for term in query_terms if term in dictionary.token2id]
        query_term_ids = [term_id for term_id in query_term_ids if term_id is not None]

    # Initialize scores
    scores = np.zeros(bm25_D)

    for term_id in query_term_ids:
        idf = bm25_idf.get(term_id, 0)
        # Collect term frequencies for this term across all documents
        tfs = np.array([doc.get(term_id, 0) for doc in bm25_corpus])
        dl = bm25_doc_lengths
        denom = tfs + k1 * (1 - b + b * (dl / bm25_avgdl))
        numer = tfs * (k1 + 1)
        score = idf * (numer / denom)

        # Apply query term weight
        if query_weights is not None:
            weight = query_weights.get(term_id, 1.0)
        else:
            weight = 1.0

        scores += weight * score

    return scores


def is_include_ng_word(tags: List[str]) -> bool:
    for ng_word in NG_WORDS:
        if ng_word in tags:
            return True
    return False

def find_similar_documents(new_doc: str, topn: int = 50) -> List[Tuple[int, float]]:
    # Remove weight descriptions to get BoW representation
    splited_doc = [x.split(":")[0] for x in new_doc.split(' ')]
    query_bow: List[Tuple[int, int]] = dictionary.doc2bow(splited_doc)

    query_lsi = normalize_and_apply_weight_lsi(query_bow, new_doc)

    # Existing similarity scores using LSI
    sims_lsi: ndarray = index[query_lsi]

    # BM25 scores
    bm25_scores = compute_bm25_scores(splited_doc)

    # Normalize scores
    if sims_lsi.max() > 0:
        sims_lsi = sims_lsi / sims_lsi.max()
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()

    # Combine scores
    final_scores = BM25_WEIGHT * bm25_scores + LSI_WEIGHT * sims_lsi

    # Get top documents
    sims = list(enumerate(final_scores))
    sims = sorted(sims, key=lambda item: -item[1])

    if len(sims) > 10:
        # Perform rescoring
        top10_sims = sims[:10]  # Top 10 documents
        top10_doc_ids = [doc_id for doc_id, _ in top10_sims]
        top10_doc_ids_set = set(top10_doc_ids)

        # Calculate weighted average LSI vector and weighted average BoW for BM25
        total_weight = sum(score for _, score in top10_sims)
        weighted_sum_lsi = np.zeros(model.num_topics)
        weighted_bow = {}

        for (doc_id, score) in top10_sims:
            # Retrieve document tags
            tokens = image_files_name_tags_arr[doc_id].split(',')
            tags = tokens[1:]  # Exclude file path

            # Convert to BoW
            doc_bow = dictionary.doc2bow(tags)

            # Update weighted BoW for BM25
            for term_id, freq in doc_bow:
                weighted_bow[term_id] = weighted_bow.get(term_id, 0) + freq * score

            # Get LSI vector
            doc_lsi = model[doc_bow]

            # Convert sparse vector to dense vector
            doc_dense = np.zeros(model.num_topics)
            for idx, val in doc_lsi:
                doc_dense[idx] = val

            # Weighted addition
            weighted_sum_lsi += doc_dense * score

        # Compute average LSI vector
        weighted_avg_lsi = weighted_sum_lsi / total_weight

        # Convert average LSI vector to sparse format
        weighted_avg_lsi_sparse = [(idx, val) for idx, val in enumerate(weighted_avg_lsi) if val != 0]

        # Rescore using LSI index
        rescored_sims_lsi = index[weighted_avg_lsi_sparse]

        # Prepare weighted query terms and weights for BM25
        # Normalize weighted_bow by total_weight
        query_term_weights = {term_id: freq / total_weight for term_id, freq in weighted_bow.items()}

        # Compute BM25 scores for weighted query
        rescored_bm25_scores = compute_bm25_scores(query_weights=query_term_weights)

        # Normalize rescored scores
        rescored_sims_lsi = np.array(rescored_sims_lsi)
        if rescored_sims_lsi.max() > 0:
            rescored_sims_lsi = rescored_sims_lsi / rescored_sims_lsi.max()
        if rescored_bm25_scores.max() > 0:
            rescored_bm25_scores = rescored_bm25_scores / rescored_bm25_scores.max()

        # Combine rescored LSI and BM25 scores
        rescored_final_scores = BM25_WEIGHT * rescored_bm25_scores + LSI_WEIGHT * rescored_sims_lsi

        # Convert rescored scores to list
        rescored_sims_list = list(enumerate(rescored_final_scores))

        # Exclude top 10 documents
        rescored_sims_list = [item for item in rescored_sims_list if item[0] not in top10_doc_ids_set]

        # Set the scores of top 10 documents to 1
        final_sims = []
        for idx, (doc_id, _) in enumerate(top10_sims):
            final_sims.append((doc_id, 1.0))

        # Add remaining documents
        final_sims.extend(rescored_sims_list)

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

    # name convention: "{search_tags}" + "_" + {timestamp} + ".txt"
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

    tag_file_path: str = 'tags-wd-tagger_lsi_idx.csv'
    image_files_name_tags_arr = []
    with open(tag_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            image_files_name_tags_arr.append(line.strip())

    model = LsiModel.load("lsi_model")
    index = MatrixSimilarity.load("lsi_index")
    dictionary = pickle.load(open("lsi_dictionary", "rb"))

    if 'bm25_corpus' in ss:
        bm25_corpus = ss['bm25_corpus']
        bm25_doc_lengths = ss['bm25_doc_lengths']
        bm25_avgdl = ss['bm25_avgdl']
        bm25_idf = ss['bm25_idf']
        bm25_D = ss['bm25_D']
        return

    # Build BM25 index
    bm25_corpus = []
    doc_lengths = []
    term_doc_freq = {}
    bm25_D = len(image_files_name_tags_arr)

    for line in image_files_name_tags_arr:
        tokens = line.strip().split(',')
        tags = tokens[1:]  # Assuming the first token is file path

        # Convert tags to term IDs
        term_ids = [dictionary.token2id.get(tag, None) for tag in tags if tag in dictionary.token2id]
        # Remove None values
        term_ids = [term_id for term_id in term_ids if term_id is not None]

        # Build term frequency dictionary for the document
        term_freq = {}
        for term_id in term_ids:
            term_freq[term_id] = term_freq.get(term_id, 0) + 1

        bm25_corpus.append(term_freq)
        doc_lengths.append(len(term_ids))

        # Update document frequency for terms
        for term_id in term_freq.keys():
            term_doc_freq[term_id] = term_doc_freq.get(term_id, 0) + 1

    bm25_doc_lengths = np.array(doc_lengths)
    bm25_avgdl = np.mean(bm25_doc_lengths)

    # Compute IDF for each term
    bm25_idf = {}
    for term_id, df in term_doc_freq.items():
        idf = np.log(1 + (bm25_D - df + 0.5) / (df + 0.5))
        bm25_idf[term_id] = idf

    ss['bm25_corpus'] = bm25_corpus
    ss['bm25_doc_lengths'] = bm25_doc_lengths
    ss['bm25_avgdl'] = bm25_avgdl
    ss['bm25_idf'] = bm25_idf
    ss['bm25_D'] = bm25_D


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
