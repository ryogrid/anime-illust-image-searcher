from gensim import corpora
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
from gensim.utils import simple_preprocess
import pickle

# generate corpus for gensim and index text file for search tool
def read_documents_and_gen_idx_text(file_path):
    corpus_base = []
    idx_text_fpath = file_path.split('.')[0] + '_lsi_idx.csv'
    with open(idx_text_fpath, 'w', encoding='utf-8') as idx_f:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = line.split(",")
                # remove file path element
                row = row[1:]
                # # remove last element
                # row = row[:-1]

                # join tags with space for gensim
                tags_line = ' '.join(row)                                
                tokens = simple_preprocess(tags_line.strip())
                # ignore simple_preprocess failure case and short tags image
                if tokens and len(tokens) >= 3:
                    corpus_base.append(tokens)
                    idx_f.write(line)
                    idx_f.flush()
    return corpus_base

# read image files pathes from file
def read_documents(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        documents = [line.strip() for line in file.readlines()]
    return documents

def main():
    processed_docs = read_documents_and_gen_idx_text('tags-wd-tagger.txt')

    # image file => doc_id
    dictionary = corpora.Dictionary(processed_docs)
    # remove frequent tags
    #dictionary.filter_n_most_frequent(500)

    with open('lsi_dictionary', 'wb') as f:
        pickle.dump(dictionary, f)

    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # gen LSI model with specified number of topics (dimensions)
    # ATTENTION: num_topics should be set to appropriate value!!!
    lsi_model = LsiModel(corpus, id2word=dictionary, num_topics=700)

    lsi_model.save("lsi_model")

    # make similarity index
    index = MatrixSimilarity(lsi_model[corpus])

    index.save("lsi_index")

if __name__ == "__main__":
    main()
