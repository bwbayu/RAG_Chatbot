# utils/pinecone_setup.py
import json
import os
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from pinecone_text.sparse import BM25Encoder
from pinecone import ServerlessSpec
from get_embedding import get_dense_embeddings
from pathlib import Path

# load env
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HOST_PINECONE_DENSE = os.getenv('HOST_PINECONE_DENSE')
HOST_PINECONE_SPARSE = os.getenv('HOST_PINECONE_SPARSE')
NAMESPACE = os.getenv('NAMESPACE')
EMBED_DIM = int(os.getenv('EMBED_DIM')) if os.getenv('EMBED_DIM') else 1024

# config
pc = Pinecone(api_key=PINECONE_API_KEY)
index_dense = pc.Index(host=HOST_PINECONE_DENSE)
index_sparse = pc.Index(host=HOST_PINECONE_SPARSE)

def create_index():
    # create index or vector database for dense and sparse vector
    dense_index_name = "dense-cs-upi"
    sparse_index_name = "sparse-cs-upi"

    if not pc.has_index(dense_index_name):
        print("create dense index")
        pc.create_index(
            name=dense_index_name,
            vector_type="dense",
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
            deletion_protection="disabled"
        )

    if not pc.has_index(sparse_index_name):
        print("create sparse index")
        pc.create_index(
            name=sparse_index_name,
            vector_type="sparse",
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1"
            )
        )

def generate_embedding(path_files, bm25_model):
    try:
        with open(path_files, 'r') as file:
            data = json.load(file)
            dense_vectors = []
            sparse_vectors = []
            for item in data:
                # get dense embedding
                dense_item = {
                    "id": item['_id'], 
                    "values": get_dense_embeddings(item['text'], EMBED_DIM), 
                    "metadata": {key: value for key, value in item.items() if key not in {'_id'}}
                }
                if dense_item["values"] is not None:
                    dense_vectors.append(dense_item)
                # get sparse embedding
                sparse_vals = bm25_model.encode_documents([item["text"]])[0]
                if sparse_vals and sparse_vals.get("indices") and sparse_vals.get("values"):
                    sparse_item = {
                        "id": item["_id"],
                        "sparse_values": sparse_vals,
                        "metadata": {k: v for k, v in item.items() if k not in {"_id"}}
                    }
                    sparse_vectors.append(sparse_item)
            
            return dense_vectors, sparse_vectors
    except FileNotFoundError:
        print(f"Error: {path_files} not found. Please ensure the file exists in the correct directory.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path_files}. The file might be malformed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# read data
folder_path = 'data/final_id'

BASE_DIR = Path(__file__).resolve().parents[1]   # project root
BM25_PATH = BASE_DIR / "model" / "bm25_params.json"

def create_corpus(corpus, folder_path):
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if len(filename.split('.')) == 2 and filename.split('.')[1] == 'json':
                file_path=folder_path+'/'+filename
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        for item in data:
                            corpus.append(item['text'])
                        
                except FileNotFoundError:
                    print(f"Error: {file_path} not found. Please ensure the file exists in the correct directory.")
                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {file_path}. The file might be malformed.")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
    
    print("corpus created successfully")

# define bm25 model
def create_corpus_train_bm25_model(bm25, folder_path):
    # folder_path juga dibuat absolut
    folder_path = (BASE_DIR / folder_path).resolve()
    bm25_corpus = []
    create_corpus(bm25_corpus, str(folder_path))
    bm25.fit(bm25_corpus)
    bm25.dump(str(BM25_PATH))
    print("bm25 model successfully loaded")

if __name__ == "__main__":
    # create index (if not available)
    create_index()
    # create corpus and train bm25 model
    bm25 = BM25Encoder(stem=False)
    create_corpus_train_bm25_model(bm25, folder_path)
    print("load bm25 model done")

    # generate dense and sparse vector
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if len(filename.split('.')) == 2 and filename.split('.')[1] == 'json':
                file_path=folder_path+'/'+filename
                # insert data dense
                dense_vectors, sparse_vectors = generate_embedding(file_path, bm25_model=bm25)
                print("generate dense and sparse vector done for: ", filename)
                index_dense.upsert(
                    vectors=dense_vectors,
                    namespace=NAMESPACE
                )
                print("upsert dense vector successfully for: ", filename)
                # insert data sparse
                index_sparse.upsert(
                    namespace=NAMESPACE,
                    vectors=sparse_vectors
                )
                print("upsert sparse vector successfully for: ", filename)
