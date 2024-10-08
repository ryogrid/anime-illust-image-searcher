from gensim.models.lsimodel import LsiModel
from gensim.utils import simple_preprocess
from gensim.similarities import MatrixSimilarity
import pickle

import argparse
import streamlit as st

# $ streamlit run web-ui-image-search-lsi.py

ss = st.session_state
search_tags = ''
image_files_name_tags_arr = []
args = None
model = None
index = None
dictionary = None

NG_WORDS = ['language', 'english_text', 'pixcel_art']

def find_similar_documents(model, new_doc, topn=50):
    # Vectorize the query
    query_bow = dictionary.doc2bow(simple_preprocess(new_doc))
    query_lsi = model[query_bow]

    # Calculate similarities
    sims = index[query_lsi]

    # Sort results
    sims = sorted(enumerate(sims), key=lambda item: -item[1])    
    sims_filtered = [x for x in sims if x[1] > 0.1]
    if len(sims_filtered) > 30:
        sims = sims_filtered
    else:
        sims = sims[:30]

    ret_len = topn
    if ret_len > len(sims):
        ret_len = len(sims)    
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
    if 'selected_image_info' not in ss:
        ss['selected_image_info'] = None
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
            st.rerun()
    else:
        if ss[session_key] >= -num:
            ss[session_key] += num
            st.rerun()

def convert_data_structure(image_info_list):
    pages = []
    rows = []
    cols = []

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

def display_images():
    global ss

    if len(ss['data']) > 0:
        for data_per_page in ss['data'][ss['page_index']]:
            cols = st.columns(5)
            for col_index, col_ph in enumerate(cols):
                try:
                    image_info = data_per_page[col_index]
                    key = f"img_{ss['page_index']}_{image_info['doc_id']}_{col_index}"
                    # Make the image clickable
                    if col_ph.button('info', key=key):
                        ss['selected_image_info'] = image_info
                        st.rerun()
                    col_ph.image(image_info['file_path'], use_column_width=True)
                except Exception as e:
                    print(f'Error: {e}')
                    continue

def pagination():
    col1, col2, col3 = st.columns([1, 8, 1])
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

def display_selected_image():
    global ss
    image_info = ss['selected_image_info']
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image(image_info['file_path'], use_column_width=True)
    with col2:
        st.write("File Path:")
        st.code(image_info['file_path'])
        st.write("Tags:")
        st.write('  \n'.join(image_info['tags']))
    if st.button('Close'):
        ss['selected_image_info'] = None
        st.rerun()

def show_search_result():
    global image_files_name_tags_arr
    global args

    load_model()
    similar_docs = find_similar_documents(model, search_tags, topn=2000) 

    found_docs_info = []
    for doc_id, similarity in similar_docs:
        try:
            found_img_info_splited = image_files_name_tags_arr[doc_id].split(',')
            if is_include_ng_word(found_img_info_splited):
                continue
            print(f'Image ID: {doc_id}, Similarity: {similarity}, Tags: {image_files_name_tags_arr[doc_id]}')
            found_fpath = found_img_info_splited[0]
            if args.rep:
                found_fpath = found_fpath.replace(args.rep[0], args.rep[1])
            # Collect image info
            found_docs_info.append({
                'file_path': found_fpath,
                'doc_id': doc_id,
                # Assuming tags start from index 1
                'tags': found_img_info_splited[1:]
            })
        except Exception as e:
            print(f'Error: {e}')
            continue    

    pages = convert_data_structure(found_docs_info)
    init_session_state(pages)

def load_model():
    global model
    global image_files_name_tags_arr
    global index
    global dictionary

    # Load index text file to show images
    tag_file_path = 'tags-wd-tagger_lsi_idx.csv'
    image_files_name_tags_arr = []
    with open(tag_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            image_files_name_tags_arr.append(line.strip())

    # Load model and others
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

    # Input form
    search_tags = st.text_input('Enter search tags', value='')
    if st.button('Search'): 
        show_search_result()
        st.rerun()

    if 'selected_image_info' in ss and ss['selected_image_info']:
        display_selected_image()
    else:
        display_images()
        pagination()

main()
