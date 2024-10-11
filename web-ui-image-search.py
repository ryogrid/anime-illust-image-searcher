import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import open_clip
import faiss

import argparse
import streamlit as st
import time

# $ streamlit run web-ui-image-search.py

ss = st.session_state
search_tags = ''
indexed_file_pathes = []

args = None

clip_model = None
tokenizer = None
index = None

CLIP_MODEL_REPO = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'

INDEX_FNAME = 'clip-index'
INDEX_FPATHES_FNAME = 'clip-index-fpathes.txt'

def mcut_threshold(scores):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
    for Multi-label Classification. In 11th International Symposium, IDA 2012
    (pp. 172-183).
    """
    sorted_scores = scores[scores.argsort()[:-1]]
    difs = sorted_scores[:-1] - sorted_scores[1:]
    t = difs.argmax()
    thresh = (sorted_scores[t] + sorted_scores[t + 1]) / 2
    return thresh

def find_similar_documents(query, topn=50):
    query_tok = tokenizer(query)
    query_vec = clip_model.encode_text(query_tok)
    query_vec /= query_vec.norm(dim=-1, keepdim=True)
    query_vec = query_vec.detach().numpy()
    topn = len(indexed_file_pathes) if topn > len(indexed_file_pathes) else topn
    result_sims, result_idxes = index.search(query_vec, topn)

    thresh = mcut_threshold(result_sims[0])
    print(f'Threshold: {thresh}')

    result_sims = result_sims[0].tolist()
    result_idxes = result_idxes[0].tolist()

    pairs = [(idx - 1, sim) for idx, sim in zip(result_idxes, result_sims) if idx > 0 and sim > thresh]
    pairs = sorted(pairs, key=lambda x: -1 * x[1])

    ret_len = topn
    if ret_len > len(pairs):
        ret_len = len(pairs)
    return pairs[:ret_len]

def init_session_state(data=[]):
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

def get_all_images():
    images = []
    for page in ss['data']:
        for row in page:
            for image_info in row:
                images.append(image_info['file_path'])
    return images

def slideshow():
    images = get_all_images()
    if len(images) == 0:
        st.write("No images to display in slideshow.")
        ss['slideshow_active'] = False
        st.rerun()
    if 'slideshow_index' not in ss:
        ss['slideshow_index'] = 0
    cols = st.columns([1])

    try:
        cols[0].image(images[ss['slideshow_index']], use_column_width=True)
    except Exception as e:
        print(f'Error: {e}')
        ss['slideshow_index'] = (ss['slideshow_index'] + 1) % len(images)
        st.rerun()
        return
    # Provide a 'Stop Slideshow' button
    if st.button('Stop'):
        ss['slideshow_active'] = False
        ss['slideshow_index'] = 0
        ss['text_input'] = ss['last_search_tags']
    else:
        # Wait for 5 seconds
        time.sleep(5)
        # Update the index
        ss['slideshow_index'] = (ss['slideshow_index'] + 1) % len(images)    
    st.rerun()

def is_now_slideshow():
    return 'slideshow_active' in ss and ss['slideshow_active']

def display_images():
    global ss

    if 'data' in ss and len(ss['data']) > 0:
        # Add the 'Slideshow' button in the upper-left corner
        cols = st.columns([10])
        with cols[0]:
            if st.button('Slideshow'):
                ss['slideshow_active'] = True
                ss['slideshow_index'] = 0
                st.rerun()
                return

            for data_per_page in ss['data'][ss['page_index']]:                
                cols = st.columns(5)
                for col_index, col_ph in enumerate(cols):
                    try:
                        image_info = data_per_page[col_index]
                        key = f"img_{ss['page_index']}_{image_info['image_id']}_{col_index}"
                        # Make the image clickable
                        if col_ph.button('info', key=key):
                            ss['selected_image_info'] = image_info
                            st.rerun()
                        col_ph.image(image_info['file_path'], use_column_width=True)
                    except Exception as e:
                        print(f'Error: {e}')
                        continue
            pagination()


def pagination():
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

def display_selected_image():
    global ss
    image_info = ss['selected_image_info']
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image(image_info['file_path'], use_column_width=True)
    with col2:
        st.write("Matching Score:")
        st.write("{:.2f}".format(image_info['score'] * 100))
        st.write("File Path:")
        st.code(image_info['file_path'])
    if st.button('Close'):
        ss['selected_image_info'] = None
        ss['text_input'] = ss['last_search_tags']
        st.rerun()

def show_search_result():
    global image_files_name_tags_arr
    global args

    load_model()
    find_result = find_similar_documents(search_tags, topn=2000) 

    found_docs_info = []
    for image_id, score in find_result:
        try:
            found_fpath = indexed_file_pathes[image_id]
            print(f'Image ID: {image_id}, Score: {score}, Filepath: {found_fpath}')
            if args.rep:
                found_fpath = found_fpath.replace(args.rep[0], args.rep[1])
            # Collect image info
            found_docs_info.append({
                'file_path': found_fpath,
                'image_id': image_id,
                'score': score,
            })
        except Exception as e:
            print(f'Error: {e}')
            continue    

    pages = convert_data_structure(found_docs_info)
    init_session_state(pages)

def load_model():
    global clip_model, tokenizer
    global indexed_file_pathes
    global index

    # Load index text file to show images
    indexed_file_pathes = []
    with open(INDEX_FPATHES_FNAME, 'r', encoding='utf-8') as f:
        for line in f:
            indexed_file_pathes.append(line.strip())

    # Load model and others
    clip_model, _ = open_clip.create_model_from_pretrained('hf-hub:' + CLIP_MODEL_REPO)
    tokenizer = open_clip.get_tokenizer('hf-hub:' + CLIP_MODEL_REPO)    
    index = faiss.read_index(INDEX_FNAME)


def main():
    global search_tags
    global args
    global ss

    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', nargs=2, required=False, help='replace the string in file path to one you want')
    args = parser.parse_args()

    init_session_state()

    if is_now_slideshow():
        slideshow()
    else:        
        if 'selected_image_info' in ss and ss['selected_image_info']:
            display_selected_image()
        else:
            # Input form
            search_tags = st.text_input('Enter search tags', value='', key='text_input')            
            if search_tags and ss['last_search_tags'] != search_tags:
                ss['last_search_tags'] = search_tags
                show_search_result()
                st.rerun()            
            display_images()            

main()