from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from main import get_dense_embeddings, get_sparse_embeddings
import os
from dotenv import load_dotenv
from collections import defaultdict
import json
import requests
import time
from datetime import datetime

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# load env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HOST_PINECONE_DENSE = os.getenv('HOST_PINECONE_DENSE')
HOST_PINECONE_SPARSE = os.getenv('HOST_PINECONE_SPARSE')
SILICONFLOW_URL_RERANK = os.getenv('SILICONFLOW_URL_RERANK')
SILICONFLOW_API_KEY = os.getenv('SILICONFLOW_API_KEY')
NAMESPACE = os.getenv('NAMESPACE')
EMBED_DIM = int(os.getenv('EMBED_DIM')) if os.getenv('EMBED_DIM') else None
TOP_K = 10

# config
pc = Pinecone(api_key=PINECONE_API_KEY)
index_dense = pc.Index(host=HOST_PINECONE_DENSE)
index_sparse = pc.Index(host=HOST_PINECONE_SPARSE)

# load bm25 model
bm25 = BM25Encoder(stem=False)
try:
    bm25.load("model/bm25_params.json")
    print("bm25 params loaded")
except Exception as e:
    print("WARN: gagal load bm25 params", e)

# List of types
TYPES = ['Berita', 'Fasilitas', 'Fasilitas Departemen Ilmu Komputer', 'Fasilitas Fakultas/FPMIPA', 'Fasilitas Universitas/UPI', 'KBK/Penjurusan', 'Mata Kuliah', 'Metode Pengajaran', 
         'Person', 'Program Info Ilmu Komputer', 'Program Info Pendidikan Ilmu Komputer', 'Proses Penilaian', 'Sasaran Program', 'Sasaran Program Ilmu Komputer', 
         'Sasaran Program Magister Pendidikan Ilmu Komputer', 'Overview', 'Tujuan', 'Visi dan Misi', 'Visi dan Misi Ilmu Komputer', 'Visi dan Misi Magister Pendidikan Ilmu Komputer', 
         'Visi dan Misi Pendidikan Ilmu Komputer', 'alumni', 'beasiswa', 'keahlian', 'keluarga mahasiswa komputer', 'mata kuliah magister pendidikan ilmu komputer', 
         'mata kuliah pendidikan ilmu komputer', 'metode pengajaran pendidikan ilmu komputer', 'metode penilaian pendidikan ilmu komputer', 'pelayanan administrasi', 
         'pendaftaran', 'penjaminan mutu', 'pertukaran mahasiswa', 'prestasi', 'program info magister pendidikan ilmu komputer', 'program info pendidikan ilmu komputer']

def classify_query(query, chat_history):
    template = """Klasifikasikan query berikut dan, jika ada, riwayat obrolan ke dalam satu atau lebih kategori dari list ini: {types}.
    Kembalikan daftar kategori yang relevan dalam format list of string (misalnya, ["KBK/Penjurusan", "Mata Kuliah"]).
    Jika ada yang bertanya terkait profil kualifikasi lulusan, capaian pembelajaran masukkan ke kategori ["ProgramInfo"].
    Jika ada yang bertanya terkait tujuan masukkan ke kategori ["Visi dan Misi"]
    Jika ada yang bertanya terkait Ilkom maka merujuk pada Ilmu Komputer dan pendilkom pada Pendidikan Ilmu Komputer
    Jika tidak ada kategori spesifik yang cocok atau Anda tidak yakin, kembalikan ["Other"].
    Hanya kembalikan daftar kategori dalam format list of string, tanpa penjelasan tambahan.

    Query: {query}

    Riwayat obrolan:
    {chat_history}
    """
    prompt = ChatPromptTemplate.from_template(template)
    # gpt-4.1-mini / gpt-4.1-nano / o4-mini
    model = ChatOpenAI(model_name="gpt-4.1-mini", max_retries=4, timeout=60)
    chain = prompt | model
    try:
        response = chain.invoke({"types": TYPES, "query": query, "chat_history": chat_history})
        classified_types = json.loads(response.content.strip())
        # Validate that all returned types are in TYPES
        classified_types = [t for t in classified_types if t in TYPES]
        # If no valid types or empty, return ["Other"]
        return classified_types if classified_types else ["Other"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse classify_query response: {response.content}, error: {str(e)}")
        return ["Other"]

def search_dense_index(text: str, filter_types=None):
    query_dense = get_dense_embeddings(text, EMBED_DIM)
    # using filter
    filter_query = {}
    if filter_types and filter_types != ["Other"]:
        filter_query = {"type": {"$in": filter_types}}
    
    dense_response = index_dense.query(
        namespace=NAMESPACE,
        vector= query_dense,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False,
        filter=filter_query if filter_query else None
    )
    
    matches = dense_response.get("matches", []) or []
    dense_results = []
    for item in matches:
        text = item['metadata'].get("text", '')
        dense_results.append({
            "id": item.get("id"),
            "similarity": item.get('score', 0.0),
            "text": text
        })
    
    # non-filter
    dense_response2 = index_dense.query(
        namespace=NAMESPACE,
        vector= query_dense,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )
    
    matches2 = dense_response2.get("matches", []) or []
    dense_results2 = []
    for item in matches2:
        text = item['metadata'].get("text", '')
        dense_results2.append({
            "id": item.get("id"),
            "similarity": item.get('score', 0.0),
            "text": text
        })
    
    return dense_results, dense_results2

def search_sparse_index(text: str, filter_types=None):
    query_sparse = get_sparse_embeddings(text=text, bm25_model=bm25, query_type='search')
    # filter
    filter_query = {}
    if filter_types and filter_types != ["Other"]:
        filter_query = {"type": {"$in": filter_types}}
    
    sparse_response = index_sparse.query(
        namespace=NAMESPACE,
        sparse_vector=query_sparse,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False,
        filter=filter_query if filter_query else None
    )
    
    matches = sparse_response.get("matches", []) or []
    sparse_results = []
    for item in matches:
        text = item['metadata'].get("text", '')
        sparse_results.append({
            "id": item.get("id"),
            "similarity": item.get('score', 0.0),
            "text": text
        })

    # non filter
    sparse_response2 = index_sparse.query(
        namespace=NAMESPACE,
        sparse_vector=query_sparse,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )
    
    matches2 = sparse_response2.get("matches", []) or []
    sparse_results2 = []
    for item in matches2:
        text = item['metadata'].get("text", '')
        sparse_results2.append({
            "id": item.get("id"),
            "similarity": item.get('score', 0.0),
            "text": text
        })
    
    return sparse_results, sparse_results2

"""
RRF score(d) = Σ 1/(k+rank(d)) where k is between 1-60 where d is document
"""
def rrf_fusion(dense_results, sparse_results, k=60, top_n=TOP_K):
    scores = defaultdict(float)

    # add rrf score from dense result
    for rank, res in enumerate(dense_results, 1):
        doc_id = res['id']
        scores[doc_id] += 1/(k + rank)

    # add rrf score from dense result
    for rank, res in enumerate(sparse_results, 1):
        doc_id = res['id']
        scores[doc_id] += 1/(k + rank)

    # sort by rrf score desc
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    all_results = {r['id']: r for r in dense_results + sparse_results}
    fused_results = [all_results[doc_id] for doc_id, _ in fused[:top_n]]

    return fused_results

def reranking_results(query, docs, fused_results, top_k=10):
    # Validate inputs
    if not query or not isinstance(query, str):
        print("Invalid query: Query must be a non-empty string")
        return fused_results  # Fallback to fused results
    if not docs or not all(isinstance(doc, str) and doc.strip() for doc in docs):
        print("Invalid documents: All documents must be non-empty strings")
        return fused_results  # Fallback to fused results
    
    payload = {
        "model": "Qwen/Qwen3-Reranker-8B",
        "query": query,
        "documents": docs,
        "top_n": top_k,
        "return_documents": False
    }

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(SILICONFLOW_URL_RERANK, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        reranked_data = response.json()
        reranked_results = reranked_data.get('results', [])
        
        # map original data after reranking
        final_results = []
        for res in reranked_results:
            index = res['index']
            relevance_score = res['relevance_score']
            original_result = fused_results[index]
            final_results.append({
                "id": original_result['id'],
                "similarity": relevance_score,
                "text": original_result['text']
            })
        
        return final_results
    else:
        print(f"Error in reranking: {response.status_code} - {response.text}")
        return fused_results

def context_generation(query, contexts, chat_history, streaming=True):
    context = "\n\n".join([data.get("text", "") for data in contexts])
    template = """Anda adalah asisten AI yang menjawab pertanyaan berdasarkan konteks yang diberikan dan, jika ada, riwayat obrolan.
    Gunakan hanya informasi yang relevan dengan pertanyaan. Abaikan konteks yang tidak relevan atau ambigu.
    Jika ada yang bertanya terkait Ilkom maka merujuk pada Ilmu Komputer dan pendilkom pada Pendidikan Ilmu Komputer.
    Jika memang tidak ada di konteks suruh user berikan pertanyaan yang lebih detail. Prioritaskan ini dibandingkan "Saya tidak tahu." 
    Jika jawaban tidak dapat ditentukan dari informasi yang diberikan, jawablah dengan: "Saya tidak tahu.".
    Jangan berikan penjelasan yang tidak ada di konteks.

    Konteks:
    {context}

    Pertanyaan:
    {query}

    Riwayat obrolan:
    {chat_history}

    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="gpt-4.1-mini", streaming=streaming, max_retries=4, timeout=60)
    chain = prompt | model
    if not streaming:
        response = chain.invoke({"context": context, "query": query, "chat_history": chat_history})
        return response.content
    else:
        return chain.stream({"context": context, "query": query, "chat_history": chat_history})

def RAG_pipeline(query, chat_history, streaming=True):
    print("start : ", datetime.now())
    classified_type = classify_query(query, chat_history)
    dense_results_f, dense_results_nf = search_dense_index(query, filter_types=classified_type)
    sparse_results_f, sparse_results_nf = search_sparse_index(query, filter_types=classified_type)
    # filter
    fused_results_f = rrf_fusion(dense_results_f, sparse_results_f)
    docs_f = [result['text'] for result in fused_results_f]
    # non filter
    fused_results_nf = rrf_fusion(dense_results_nf, sparse_results_nf)
    docs_nf = [result['text'] for result in fused_results_nf]
    # rerank from filter and non filter
    fused_results_f.extend(fused_results_nf)
    docs_f.extend(docs_nf)
    contexts = reranking_results(query, docs_f, fused_results_f)
    
    return context_generation(query, contexts, chat_history, streaming=streaming)