# src/rag_pipeline.py
import json
import os
from collections import defaultdict
from typing import List, Dict, Any

import asyncio, httpx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from pinecone.grpc import PineconeGRPC as Pinecone
from src.get_embedding import get_dense_embeddings, get_sparse_embeddings
from src.bm25_model import load_bm25_model

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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
TOP_K = 10
EMBED_DIM = int(os.getenv('EMBED_DIM')) if os.getenv('EMBED_DIM') else 1024

# config
pc = Pinecone(api_key=PINECONE_API_KEY)
index_dense = pc.Index(host=HOST_PINECONE_DENSE)
index_sparse = pc.Index(host=HOST_PINECONE_SPARSE)
bm25 = load_bm25_model()

TYPES = ['Berita', 'Fasilitas', 'Fasilitas Departemen Ilmu Komputer', 'Fasilitas Fakultas/FPMIPA', 'Fasilitas Universitas/UPI', 'KBK/Penjurusan', 'Mata Kuliah', 'Metode Pengajaran', 
         'Person', 'Program Info Ilmu Komputer', 'Program Info Pendidikan Ilmu Komputer', 'Proses Penilaian', 'Sasaran Program', 'Sasaran Program Ilmu Komputer', 
         'Sasaran Program Magister Pendidikan Ilmu Komputer', 'Overview', 'Tujuan', 'Visi dan Misi', 'Visi dan Misi Ilmu Komputer', 'Visi dan Misi Magister Pendidikan Ilmu Komputer', 
         'Visi dan Misi Pendidikan Ilmu Komputer', 'alumni', 'beasiswa', 'keahlian', 'keluarga mahasiswa komputer', 'mata kuliah magister pendidikan ilmu komputer', 
         'mata kuliah pendidikan ilmu komputer', 'metode pengajaran pendidikan ilmu komputer', 'metode penilaian pendidikan ilmu komputer', 'pelayanan administrasi', 
         'pendaftaran', 'penjaminan mutu', 'pertukaran mahasiswa', 'prestasi', 'program info magister pendidikan ilmu komputer', 'program info pendidikan ilmu komputer']

async def classify_query(query: str, chat_history) -> List[str]:
    template = """Klasifikasikan query berikut dan, jika ada, riwayat obrolan ke dalam satu atau lebih kategori dari list ini: {types}.
    Kembalikan daftar kategori yang relevan dalam format list of string (misalnya, ["KBK/Penjurusan", "Mata Kuliah"]).
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
    model = ChatOpenAI(model="gpt-4.1-mini", max_retries=4, timeout=60)
    chain = prompt | model
    try:
        response = await chain.ainvoke({"types": TYPES, "query": query, "chat_history": chat_history})
        classified_types = json.loads(response.content.strip())
        classified_types = [t for t in classified_types if t in TYPES]
        return classified_types if classified_types else ["Other"]
    except json.JSONDecodeError as e:
        print(f"Failed to parse classify_query response: {response.content}, error: {str(e)}")
        return ["Other"]
    
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=True)
async def _pinecone_query_dense(vector, filter_query):
    return await asyncio.to_thread(
        index_dense.query,
        namespace=NAMESPACE,
        vector=vector,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False,
        filter=filter_query or None,
    )
    

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=True)
async def _pinecone_query_sparse(sparse_vector, filter_query):
    return await asyncio.to_thread(
        index_sparse.query,
        namespace=NAMESPACE,
        sparse_vector=sparse_vector,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False,
        filter=filter_query or None,
    )

async def search_dense_index_async(text: str, filter_types=None):
    filter_query = {"type": {"$in": filter_types}} if filter_types and filter_types != ["Other"] else {}
    vec = await get_dense_embeddings(text, EMBED_DIM)
    dense_response = await _pinecone_query_dense(vec, filter_query)
    matches = dense_response.get("matches", []) or []
    return [{
        "id": item.get("id"),
        "similarity": item.get('score', 0.0),
        "text": item['metadata'].get("text", ''),
    } for item in matches]

async def search_sparse_index_async(text: str, filter_types=None):
    filter_query = {"type": {"$in": filter_types}} if filter_types and filter_types != ["Other"] else {}
    sp = get_sparse_embeddings(text=text, bm25_model=bm25, query_type='search')
    sparse_response = await _pinecone_query_sparse(sp, filter_query)
    matches = sparse_response.get("matches", []) or []
    return [{
        "id": item.get("id"),
        "similarity": item.get('score', 0.0),
        "text": item['metadata'].get("text", ''),
    } for item in matches]

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=True)
async def _call_reranker_async(url: str, headers: Dict[str, str], payload: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=15) as client:
        return await client.post(url, headers=headers, json=payload)

async def reranking_results_async(query: str, docs: List[str], fused_results: List[Dict[str, Any]]):
    if not query or not isinstance(query, str):
        print("Invalid query: Query must be a non-empty string")
        return fused_results
    if not docs or not all(isinstance(doc, str) and doc.strip() for doc in docs):
        print("Invalid documents: All documents must be non-empty strings")
        return fused_results

    payload = {
        "model": "Qwen/Qwen3-Reranker-8B",
        "query": query,
        "documents": docs,
        "return_documents": False
    }
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        resp = await _call_reranker_async(SILICONFLOW_URL_RERANK, headers, payload)
        if resp.status_code == 200:
            reranked_data = resp.json()
            reranked_results = reranked_data.get('results', [])
            final_results = []
            for res in reranked_results:
                idx = res['index']
                relevance_score = res['relevance_score']
                if 0 <= idx < len(fused_results):
                    o = fused_results[idx]
                    final_results.append({"id": o['id'], "similarity": relevance_score, "text": o['text']})
            # fallback jika kosong
            return final_results or fused_results
        else:
            print(f"Error in reranking: {resp.status_code} - {resp.text}")
            return fused_results
    except Exception as e:
        print(f"Reranker failed: {e}")
        return fused_results

async def context_generation_async(query: str, contexts: List[Dict[str, Any]], chat_history, streaming: bool = True):
    context = "\n\n".join([data.get("text", "") for data in contexts])
    template = """Anda adalah asisten AI yang menjawab pertanyaan berdasarkan konteks yang diberikan dan, jika ada, riwayat obrolan.
    Gunakan hanya informasi yang relevan dengan pertanyaan. Abaikan konteks yang tidak relevan atau ambigu.
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
    model = ChatOpenAI(model="gpt-4.1-mini", streaming=streaming, max_retries=4, timeout=60)
    chain = prompt | model

    inputs = {"context": context, "query": query, "chat_history": chat_history}

    if not streaming:
        resp = await chain.ainvoke(inputs)
        return resp.content
    else:
        return chain.astream(inputs)

async def RAG_pipeline_async(query: str, chat_history, streaming: bool = True):
    classified_type = await classify_query(query, chat_history)
    # create task to search data from pinecone simultaneously
    async with asyncio.TaskGroup() as tg:
        t_dense   = tg.create_task(search_dense_index_async(query, classified_type))
        t_sparse  = tg.create_task(search_sparse_index_async(query, classified_type))
        t_dense2  = tg.create_task(search_dense_index_async(query, ["Other"]))
        t_sparse2 = tg.create_task(search_sparse_index_async(query, ["Other"]))
    
    # get result
    dense_results   = t_dense.result()
    sparse_results  = t_sparse.result()
    dense_results2  = t_dense2.result()
    sparse_results2 = t_sparse2.result()

    # fusion filter and non filter result
    fused_results  = rrf_fusion(dense_results,  sparse_results)
    fused_results2 = rrf_fusion(dense_results2, sparse_results2)
    
    # rerank filtered result
    fused_results.extend(fused_results2)
    docs = [r['text'] for r in fused_results]
    contexts = await reranking_results_async(query, docs, fused_results)

    return await context_generation_async(query, contexts, chat_history, streaming=streaming)