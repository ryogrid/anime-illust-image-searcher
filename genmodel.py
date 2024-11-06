import argparse
import copy
import os
import sys

from gensim import corpora
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.similarities import Similarity
import pickle
from typing import List, Tuple
import logging
import numpy as np

TRAIN_EPOCHS = 100
VECTOR_LENGTH = 300

# python .\genmodel.py [ --update ]

# generate corpus for gensim and index text file for search tool
def read_documents_and_gen_idx_text(file_path: str) -> Tuple[List[List[str]],List[TaggedDocument]]:
    processed_docs: List[List[str]] = []
    tagged_docs: List[TaggedDocument] = []
    idx_text_fpath: str = file_path.split('.')[0] + '_doc2vec_idx.csv'
    with open(idx_text_fpath, 'w', encoding='utf-8') as idx_f:
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_id: int = 0
            for line in f:
                row: List[str] = line.strip().split(",")
                # remove file path element
                row = row[1:]

                # tokens: List[str] = simple_preprocess(tags_line.strip())
                tokens: List[str] = row
                # ignore simple_preprocess failure case and short tags image
                if tokens and len(tokens) >= 3:
                    tagged_docs.append(TaggedDocument(tokens, [doc_id]))
                    processed_docs.append(tokens)
                    idx_f.write(line)
                    idx_f.flush()
                    doc_id += 1

    return processed_docs, tagged_docs

# read image files pathes from file
def read_documents(filename: str) -> List[str]:
    with open(filename, 'r', encoding='utf-8') as file:
        documents: List[str] = [line.strip() for line in file.readlines()]
    return documents

def gen_and_save_bm25_index(corpus: List[List[str]], dictionary: corpora.Dictionary) -> None:
    bm25_corpus = []
    doc_lengths = []
    term_doc_freq: dict[int, int] = {}
    bm25_D = len(corpus)

    for tags in corpus:
        # Convert tags to term IDs
        term_ids = [dictionary.token2id.get(tag, None) for tag in tags if tag in dictionary.token2id]
        # Remove None values
        term_ids = [term_id for term_id in term_ids if term_id is not None]

        # Build term frequency dictionary for the document
        term_freq: dict[int, int] = {}
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

    with open('bm25_corpus', 'wb') as f:
        pickle.dump(bm25_corpus, f)

    with open('bm25_idf', 'wb') as f:
        pickle.dump(bm25_idf, f)

    with open('bm25_avgdl', 'wb') as f:
        pickle.dump(bm25_avgdl, f)

    with open('bm25_D', 'wb') as f:
        pickle.dump(bm25_D, f)

    with open('bm25_doc_lengths', 'wb') as f:
        pickle.dump(bm25_doc_lengths, f)

    print('BM25 index generated')

def count_non_empty_lines(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # count only non-empty lines
                count += 1
    return count

def main(arg_str: list[str]) -> None:
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        format=format_str,
        level=logging.DEBUG
    )

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    #parser.add_argument('--dim', nargs=1, type=int, required=True, help='number of dimensions at LSI model')
    # Note: new files are diff between tags-wd-tagger.txt and tags-wd-tagger.txt.bak
    #       when specified --update, check mdate of tags-wd-tagger.txt.bak and bm25_D file for avoiding duplicated entries insertion
    parser.add_argument('--update', action='store_true', help='add new images to index')
    args: argparse.Namespace = parser.parse_args(arg_str)

    if args.update:
        if os.path.exists('tags-wd-tagger_doc2vec_idx.csv'):
            with open('tags-wd-tagger_doc2vec_idx.csv', 'r', encoding='utf-8') as f:
                with open('tags-wd-tagger_doc2vec_idx.csv.bak', 'w', encoding='utf-8') as f_bak:
                    f_bak.write(f.read())
        else:
            print('tags-wd-tagger_doc2vec_idx.csv not found')
            exit(1)

    tmp_tuple : [List[List[str]], List[TaggedDocument]] = read_documents_and_gen_idx_text('tags-wd-tagger.txt')
    processed_docs: List[List[str]] = tmp_tuple[0]
    processed_docs_for_bm25: List[List[str]] = copy.deepcopy(processed_docs)
    tagged_docs: List[TaggedDocument] = tmp_tuple[1]

    if args.update is None or args.update is False:
        exit(1)

    if args.update:
        with open('doc2vec_dictionary', 'rb') as f:
            dictionary: corpora.Dictionary = pickle.load(f)
        doc2vec_model: Doc2Vec = Doc2Vec.load("doc2vec_model")
        index: Similarity = Similarity.load("doc2vec_index")

        before_update_lines: int = count_non_empty_lines('tags-wd-tagger_doc2vec_idx.csv.bak')

        print(f'update index: {len(processed_docs) - before_update_lines} files')

        # adding vector for new images only
        processed_docs: List[List[str]] = processed_docs[before_update_lines:]
    else:
        # image file => doc_id
        dictionary: corpora.Dictionary = corpora.Dictionary(processed_docs)
        # remove frequent tags
        #dictionary.filter_n_most_frequent(500)

        with open('doc2vec_dictionary', 'wb') as f:
            pickle.dump(dictionary, f)

        # gen Doc2Vec model with specified number of dimensions
        doc2vec_model: Doc2Vec = Doc2Vec(vector_size=VECTOR_LENGTH, window=50, min_count=1, workers=1, dm=0)
        doc2vec_model.build_vocab(tagged_docs)
        doc2vec_model.train(tagged_docs, total_examples=doc2vec_model.corpus_count, epochs=TRAIN_EPOCHS)
        doc2vec_model.save("doc2vec_model")

        # similarity index
        index: Similarity = None

    # store each image infos to index
    for doc in processed_docs:
        embed_vec = doc2vec_model.infer_vector(doc)
        if index is None:
            index = Similarity('doc2vec_index', [embed_vec], num_features=VECTOR_LENGTH)
        else:
            index.add_documents([embed_vec])

    index.save("doc2vec_index")

    gen_and_save_bm25_index(processed_docs_for_bm25, dictionary)

if __name__ == "__main__":
    main(sys.argv[1:])
