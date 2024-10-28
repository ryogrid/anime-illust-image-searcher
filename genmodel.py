import argparse
import sys

import gensim.models.lsimodel
from gensim import corpora
from gensim.models import LsiModel
from gensim.similarities import Similarity, MatrixSimilarity
import pickle
from typing import List, Tuple
import logging
import numpy as np

# generate corpus for gensim and index text file for search tool
def read_documents_and_gen_idx_text(file_path: str) -> List[List[str]]:
    corpus_base: List[List[str]] = []
    idx_text_fpath: str = file_path.split('.')[0] + '_lsi_idx.csv'
    with open(idx_text_fpath, 'w', encoding='utf-8') as idx_f:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                row: List[str] = line.strip().split(",")
                # remove file path element
                row = row[1:]

                # tokens: List[str] = simple_preprocess(tags_line.strip())
                tokens: List[str] = row
                # ignore simple_preprocess failure case and short tags image
                if tokens and len(tokens) >= 3:
                    corpus_base.append(tokens)
                    idx_f.write(line)
                    idx_f.flush()

    return corpus_base

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



def main(arg_str: list[str]) -> None:
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        format=format_str,
        level=logging.DEBUG
    )

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--dim', nargs=1, type=int, required=True, help='number of dimensions at LSI model')
    args: argparse.Namespace = parser.parse_args(arg_str)

    processed_docs: List[List[str]] = read_documents_and_gen_idx_text('tags-wd-tagger.txt')

    # image file => doc_id
    dictionary: corpora.Dictionary = corpora.Dictionary(processed_docs)
    # remove frequent tags
    #dictionary.filter_n_most_frequent(500)

    """"
    with open('lsi_dictionary', 'wb') as f:
        pickle.dump(dictionary, f)

    corpus: List[List[Tuple[int, int]]] = [dictionary.doc2bow(doc) for doc in processed_docs]

    # gen LSI model with specified number of topics (dimensions)
    # ATTENTION: num_topics should be set to appropriate value!!!
    # lsi_model: LsiModel = LsiModel(corpus, id2word=dictionary, num_topics=800)
    # lsi_model: LsiModel = LsiModel(corpus, id2word=dictionary, num_topics=args.dim[0])

    svd_result: Tuple[np.ndarray, np.ndarray] = gensim.models.lsimodel.stochastic_svd(corpus, rank=args.dim[0], num_terms=len(dictionary))
    # 2D matrix
    lsi_model: np.ndarray = svd_result[0]


    lsi_model = lsi_model.T
    with open('lsi_model', 'wb') as f:
        pickle.dump(lsi_model, f)
    """

    with open('lsi_model', 'rb') as f:
        lsi_model = pickle.load(f)

    print(lsi_model.shape)

    # make similarity index
    index: Similarity = None

    for doc in processed_docs:
        value_vec: np.ndarray = np.zeros((len(dictionary)))
        bow: List[Tuple[int, float]] = dictionary.doc2bow(doc)
        for term_id, term_freq in bow:
            value_vec[term_id] = term_freq
        embedding = lsi_model @ value_vec
        embedding_vec = np.array(embedding)
        if index is None:
            index = Similarity("lsi_index", [embedding_vec], num_features=args.dim[0])
        else:
            index.add_documents([embedding_vec])

    index.save("lsi_index")

    gen_and_save_bm25_index(processed_docs, dictionary)

if __name__ == "__main__":
    main(sys.argv[1:])