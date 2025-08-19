import os
import json
from pinecone_text.sparse import BM25Encoder

def create_corpus(corpus):
    folder_path = 'data/clean'

    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if len(filename.split('.')) == 2 and filename.split('.')[1] == 'json':
                file_path=folder_path+'/'+filename
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        for item in data:
                            corpus.append(item['text'])
                        
                except FileNotFoundError:
                    print(f"Error: {file_path} not found. Please ensure the file exists in the correct directory.")
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {file_path}. The file might be malformed.")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

bm25 = BM25Encoder(stem=False)
bm25_pralatih = 'model/bm25_params.json'
if(os.path.exists(bm25_pralatih)):
    # load BM25 params from json
    bm25.load(bm25_pralatih)
else:
    # create bm25 corpus for sparse vector
    bm25_corpus = []
    create_corpus(bm25_corpus)

    # fit corpus to bm25 model
    bm25.fit(bm25_corpus)

    # store BM25 params as json
    bm25.dump("model/bm25_params.json")

doc_sparse_vector = bm25.encode_documents("The brown fox is quick")
print(doc_sparse_vector)
print("====================")
doc_sparse_vector = bm25.encode_documents(["The brown fox is quick", "test test test test test"])
print(doc_sparse_vector)

list1 = [{1}, {2}, {3}]
list2 = [{4}, {5}, {6}]
list1.extend(list2)
print(list1)