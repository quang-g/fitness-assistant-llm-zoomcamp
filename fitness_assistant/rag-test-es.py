#!/usr/bin/env python
# coding: utf-8

# ## Ingestion

# In[1]:


import pandas as pd
import json
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch


# In[2]:


df = pd.read_csv('../data/fitness_exercises_500.csv')


# In[3]:


df.head()


# In[4]:


# import minsearch
model_name = 'multi-qa-MiniLM-L6-cos-v1'
model = SentenceTransformer(model_name)


# In[5]:


df.columns


# In[6]:


documents = df.to_dict(orient='records')


# In[7]:


documents[0]


# In[8]:


for doc in tqdm(documents):
    exercise_name = doc['exercise_name']
    instruction  = doc['instruction']
    ei = exercise_name + " " + instruction

    doc['exercise_name_vector'] = model.encode(exercise_name)
    doc['instruction_vector'] = model.encode(instruction)
    doc['ei_vector'] = model.encode(ei)


# In[10]:


es_client = Elasticsearch("http://localhost:9200")
index_name = "500_fitness_exercises"

dims = 384  # embedding dimensionality

body = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "normalizer": {
                "lc": {  # lowercase + ASCII fold for case-insensitive exact matching
                    "type": "custom",
                    "char_filter": [],
                    "filter": ["lowercase", "asciifolding"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # Full-text + exact (keyword) for names
            "exercise_name": {
                "type": "text",
                "fields": {
                    "keyword": {"type": "keyword", "normalizer": "lc"}
                }
            },

            # Facet-style attributes: exact match via keyword, plus optional .text for free-text search
            "type_of_activity": {
                "type": "keyword", "normalizer": "lc",
                "fields": {"text": {"type": "text"}}
            },
            "type_of_equipment": {
                "type": "keyword", "normalizer": "lc",
                "fields": {"text": {"type": "text"}}
            },
            "body_part": {"type": "keyword", "normalizer": "lc"},
            "type": {"type": "keyword", "normalizer": "lc"},

            # IMPORTANT: index as an ARRAY of strings, e.g. ["glutes","hamstrings"]
            "muscle_groups_activated": {"type": "keyword", "normalizer": "lc"},

            # Long free-text
            "instruction": {"type": "text"},

            # Vectors (kNN)
            "exercise_name_vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine"
            },
            "instruction_vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine"
            },
            "ei_vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

# Recreate index
es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=body)

print(f"Index '{index_name}' created.")


# In[11]:


for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)


# ## Retrieval Stage

# In[12]:


from langchain.embeddings import SentenceTransformerEmbeddings
from typing import Dict
from langchain_elasticsearch import ElasticsearchRetriever


# In[13]:


es_url = 'http://localhost:9200'


# In[14]:


from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
embeddings = HuggingFaceEmbeddings(model_name=model_name)


# In[15]:


def hybrid_query(search_query: str) -> Dict:
    vector = embeddings.embed_query(search_query)  # same embeddings as for indexing
    return {
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": search_query,
                        "type": "cross_fields",         # better for attribute-style queries
                        "operator": "and",              # tighten precision; try "or" if recall too low
                        "fields": [
                            "exercise_name^4",
                            "muscle_groups_activated^2",
                            "type_of_equipment^1.5",
                            "type_of_activity^1.3",
                            "body_part^1.2",
                            "type^1.2",
                            "instruction^1"
                        ]
                    }
                }
                # "filter": {
                #     "term": {
                #         "course": course
                #     }
                # }
            }
        },
        "knn": {
            "field": "ei_vector",
            "query_vector": vector,
            "k": 5,
            "num_candidates": 10000,
            "boost": 0.5,
            # "filter": {
            #     "term": {
            #         "course": course
            #     }
            # }
        },
        "size": 5
        # "rank": {"rrf": {}},
    }


hybrid_retriever = ElasticsearchRetriever.from_es_params(
    index_name=index_name,
    body_func=hybrid_query,
    content_field='instruction',
    url=es_url,
)


# In[16]:


query = 'give me leg exercises for hamstrings'


# In[17]:


query = 'What equipment do I need to perform the Push-Up Hold exercise?'


# In[18]:


hybrid_results = hybrid_retriever.invoke(query)


# In[20]:


for result in hybrid_results:
   print(result.metadata['_source']['exercise_name'], result.metadata['_source']['muscle_groups_activated'],result.page_content, result.metadata['_score'])


# In[21]:


print(query)


# ## Implementing RAG Flow

# In[22]:


from google import genai


# In[23]:


client = genai.Client()


# In[24]:


from typing import List, Dict, Optional

def hybrid_search(
    query: str,
    es=es_client,                          # Elasticsearch client instance
    index_name=index_name,
    embeddings=embeddings,                  # embedding model with .embed_query()
    size: int = 5,               # final results returned
    knn_k: int = 50,             # neighbors to consider in the vector branch
    num_candidates: int = 3000,  # ANN candidate pool size
    knn_boost: float = 0.7,      # how much the vector score contributes
    filters: Optional[Dict] = None,
    use_rrf: bool = False,
) -> List[Dict]:
    """
    Execute a hybrid (keyword + vector) query in Elasticsearch and return a list of sources
    shaped for the RAG context builder.
    """
    vector = embeddings.embed_query(query)

    # Build keyword filter(s), applied to both text and knn branches.
    # Example: filters = {"term": {"course": "strength"}} or a full bool of filters.
    es_bool_filter = []
    if filters:
        # Accept either {"term": {...}} or {"bool": {"filter": [...]}}
        if "bool" in filters and "filter" in filters["bool"]:
            es_bool_filter.extend(filters["bool"]["filter"])
        else:
            es_bool_filter.append(filters)

    body: Dict = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "type": "cross_fields",
                            "operator": "and",
                            "fields": [
                                "exercise_name^4",
                                "muscle_groups_activated^2",
                                "type_of_equipment^1.5",
                                "type_of_activity^1.3",
                                "body_part^1.2",
                                "type^1.2",
                                "instruction^1"
                            ]
                        }
                    }
                ],
                **({"filter": es_bool_filter} if es_bool_filter else {})
            }
        },
        "knn": {
            "field": "ei_vector",
            "query_vector": vector,
            "k": knn_k,
            "num_candidates": num_candidates,
            "boost": knn_boost,
            **({"filter": es_bool_filter} if es_bool_filter else {})
        },
        "size": size
    }

    if use_rrf:
        # RRF helps fuse ranks from text and vector more robustly than raw score mixing
        body["rank"] = {"rrf": {"rank_window_size": max(knn_k, size), "rank_constant": 60}}

    res = es.search(index=index_name, body=body)
    hits = res.get("hits", {}).get("hits", [])

    # Normalize results to the fields your entry_template expects.
    out: List[Dict] = []
    for h in hits:
        src = h.get("_source", {})
        out.append({
            "id":                     src.get("id", ""),
            "exercise_name":          src.get("exercise_name", ""),
            "type_of_activity":       src.get("type_of_activity", ""),
            "type_of_equipment":      src.get("type_of_equipment", ""),
            "body_part":              src.get("body_part", ""),
            "type":                   src.get("type", ""),
            "muscle_groups_activated":src.get("muscle_groups_activated", ""),
            "instruction":            src.get("instruction", "")
        })
    return out


# In[25]:


#We want the LLM to put the document content in the context of the answer
prompt_template = """
    You're a professional fitness assistant. Answer the QUESTION based only on the CONTEXT provided from the exercise & fitness database.  

    - Use only the facts from the CONTEXT when answering the QUESTION.  
    - If the CONTEXT does not contain the answer, respond with: NONE.  
    - Keep your answer clear, concise, and detail with instruction for fitness use.  

    QUESTION: {question}  

    CONTEXT:  
    {context}  
""".strip()

entry_template = """
exercise_name: {exercise_name}',
type_of_activity: {type_of_activity},
type_of_equipment: {type_of_equipment},
body_part: {body_part},
type: {type},
muscle_groups_activated: {muscle_groups_activated},
instruction: {instruction}
""".strip()

def build_prompt(query, search_results):    
    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


# In[26]:


hybrid_search(query)


# In[27]:


def llm(prompt):
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
    )

    return response.text


# In[28]:


def rag(query):
    search_results = hybrid_search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


# In[29]:


query = 'I want some core exercises that also help my back'

answer = rag(query)
print(answer)


# ## Retrieval Evalutation

# In[30]:


df_question = pd.read_csv('../data/ground-truth-retrieval.csv')


# In[31]:


df_question.head()


# In[32]:


ground_truth = df_question.to_dict(orient='records')


# In[33]:


ground_truth[0]


# In[34]:


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


# In[35]:


documents[0]


# In[36]:


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }


# In[37]:


from tqdm.auto import tqdm


# In[38]:


evaluate(ground_truth, lambda q: hybrid_search(q['question']))


# ## Finding the best parameters

# In[39]:


df_validation = df_question[:250]
df_test = df_question[250:]


# In[41]:


df_validation.head()


# In[68]:


from typing import Dict, List, Any, Optional, Tuple
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np
import time


# In[77]:


# =========================================================
# Metrics
# =========================================================
def reciprocal_rank(pred_ids: List[Any], relevant_id: Any) -> float:
    for i, pid in enumerate(pred_ids, start=1):
        if pid == relevant_id:
            return 1.0 / i
    return 0.0

def hit_at_k(pred_ids: List[Any], relevant_id: Any, k:int=5) -> float:
    return 1.0 if relevant_id in pred_ids[:k] else 0.0


# In[78]:


# =========================================================
# Field-boost builder
# =========================================================
def _fields_with_boosts(p: Dict) -> List[str]:
    """
    Build multi_match fields with tuned boosts.
    Expect keys: b_exercise_name, b_type_of_activity, b_type_of_equipment,
                 b_body_part, b_type, b_muscle_groups, b_instruction
    """
    return [
        f"exercise_name^{float(p['b_exercise_name'])}",
        f"muscle_groups_activated^{float(p['b_muscle_groups'])}",
        f"type_of_equipment^{float(p['b_type_of_equipment'])}",
        f"type_of_activity^{float(p['b_type_of_activity'])}",
        f"body_part^{float(p['b_body_part'])}",
        f"type^{float(p['b_type'])}",
        f"instruction^{float(p['b_instruction'])}",
    ]

# =========================================================
# Client-side RRF fusion
# =========================================================
def rrf_fuse(
    lists: List[List[Dict]],
    *,
    id_key: str = "id",
    rank_window_size: int = 50,
    rank_constant: int = 60
) -> List[Tuple[Any, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    Returns list of (doc_id, fused_score) sorted desc.
    """
    scores: Dict[Any, float] = {}
    for hits in lists:
        window = hits[:rank_window_size]
        for rank, item in enumerate(window, start=1):
            did = item.get(id_key)
            if did is None:
                continue
            scores[did] = scores.get(did, 0.0) + 1.0 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# In[79]:


# =========================================================
# Branch searches (text-only and knn-only)
# =========================================================
def run_text_search(
    query: str,
    es_client,
    index_name: str,
    operator: str,
    size: int,
    filters: Optional[Dict],
    id_field: str,
    fields: List[str]
) -> List[Dict]:
    es_bool_filter: List[Dict] = []
    if filters:
        if "bool" in filters and "filter" in filters["bool"]:
            es_bool_filter.extend(filters["bool"]["filter"])
        else:
            es_bool_filter.append(filters)

    body = {
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": query,
                        "type": "cross_fields",
                        "operator": operator,
                        "fields": fields
                    }
                }],
                **({"filter": es_bool_filter} if es_bool_filter else {})
            }
        },
        "size": int(size)
    }
    res = es_client.search(index=index_name, body=body)
    hits = res.get("hits", {}).get("hits", [])
    out: List[Dict] = []
    for h in hits:
        src = h.get("_source", {})
        out.append({
            "id": src.get(id_field, h.get("_id")),
            "exercise_name":           src.get("exercise_name", ""),
            "type_of_activity":        src.get("type_of_activity", ""),
            "type_of_equipment":       src.get("type_of_equipment", ""),
            "body_part":               src.get("body_part", ""),
            "type":                    src.get("type", ""),
            "muscle_groups_activated": src.get("muscle_groups_activated", ""),
            "instruction":             src.get("instruction", "")
        })
    return out

def run_knn_search(
    query: str,
    es_client,
    index_name: str,
    embeddings,
    knn_k: int,
    num_candidates: int,
    filters: Optional[Dict],
    id_field: str
) -> List[Dict]:
    vector = embeddings.embed_query(query)

    es_bool_filter: List[Dict] = []
    if filters:
        if "bool" in filters and "filter" in filters["bool"]:
            es_bool_filter.extend(filters["bool"]["filter"])
        else:
            es_bool_filter.append(filters)

    body = {
        "knn": {
            "field": "ei_vector",
            "query_vector": vector,
            "k": int(knn_k),
            "num_candidates": int(num_candidates),
            **({"filter": es_bool_filter} if es_bool_filter else {})
        }
    }
    res = es_client.search(index=index_name, body=body)
    hits = res.get("hits", {}).get("hits", [])
    out: List[Dict] = []
    for h in hits:
        src = h.get("_source", {})
        out.append({
            "id": src.get(id_field, h.get("_id")),
            "exercise_name":           src.get("exercise_name", ""),
            "type_of_activity":        src.get("type_of_activity", ""),
            "type_of_equipment":       src.get("type_of_equipment", ""),
            "body_part":               src.get("body_part", ""),
            "type":                    src.get("type", ""),
            "muscle_groups_activated": src.get("muscle_groups_activated", ""),
            "instruction":             src.get("instruction", "")
        })
    return out


# In[80]:


# =========================================================
# Hybrid search (client-side RRF optional) with field boosts
# =========================================================
def hybrid_search_with_ids(
    query: str,
    es_client,
    index_name: str,
    embeddings,
    size: int,
    knn_k: int,
    num_candidates: int,
    knn_boost: float,           # ignored in RRF mode (rank-based fusion)
    operator: str,
    use_rrf: bool,
    filters: Optional[Dict] = None,
    id_field: str = "id",
    # RRF params:
    rrf_rank_window_size: int = 50,
    rrf_rank_constant: int = 60,
    # Field boosts (pass via params)
    b_exercise_name: float = 4.0,
    b_type_of_activity: float = 1.3,
    b_type_of_equipment: float = 1.5,
    b_body_part: float = 1.2,
    b_type: float = 1.2,
    b_muscle_groups: float = 2.0,
    b_instruction: float = 1.0,
) -> List[Dict]:
    # Build field list with current boosts
    fields = _fields_with_boosts({
        "b_exercise_name": b_exercise_name,
        "b_type_of_activity": b_type_of_activity,
        "b_type_of_equipment": b_type_of_equipment,
        "b_body_part": b_body_part,
        "b_type": b_type,
        "b_muscle_groups": b_muscle_groups,
        "b_instruction": b_instruction,
    })

    if use_rrf:
        # --- Client-side RRF fusion ---
        text_hits  = run_text_search(query, es_client, index_name, operator, rrf_rank_window_size, filters, id_field, fields)
        knn_hits   = run_knn_search(query, es_client, index_name, embeddings, max(knn_k, rrf_rank_window_size), num_candidates, filters, id_field)
        fused = rrf_fuse([text_hits, knn_hits],
                         id_key="id",
                         rank_window_size=rrf_rank_window_size,
                         rank_constant=rrf_rank_constant)
        # Materialize top 'size' docs (prefer text_hits, then knn_hits for doc materialization)
        by_id: Dict[Any, Dict] = {}
        for d in text_hits + knn_hits:
            by_id.setdefault(d["id"], d)
        top_docs: List[Dict] = []
        for did, _score in fused[:size]:
            doc = by_id.get(did)
            if doc:
                top_docs.append(doc)
        return top_docs

    # --- Single-request hybrid (text + knn) with boosts when not using RRF ---
    vector = embeddings.embed_query(query)

    es_bool_filter: List[Dict] = []
    if filters:
        if "bool" in filters and "filter" in filters["bool"]:
            es_bool_filter.extend(filters["bool"]["filter"])
        else:
            es_bool_filter.append(filters)

    body: Dict = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "type": "cross_fields",
                            "operator": operator,
                            "fields": fields
                        }
                    }
                ],
                **({"filter": es_bool_filter} if es_bool_filter else {})
            }
        },
        "knn": {
            "field": "ei_vector",
            "query_vector": vector,
            "k": int(knn_k),
            "num_candidates": int(num_candidates),
            "boost": float(knn_boost),
            **({"filter": es_bool_filter} if es_bool_filter else {})
        },
        "size": int(size)
    }

    res = es_client.search(index=index_name, body=body)
    hits = res.get("hits", {}).get("hits", [])
    out: List[Dict] = []
    for h in hits:
        src = h.get("_source", {})
        out.append({
            "id": src.get(id_field, h.get("_id")),
            "exercise_name":           src.get("exercise_name", ""),
            "type_of_activity":        src.get("type_of_activity", ""),
            "type_of_equipment":       src.get("type_of_equipment", ""),
            "body_part":               src.get("body_part", ""),
            "type":                    src.get("type", ""),
            "muscle_groups_activated": src.get("muscle_groups_activated", ""),
            "instruction":             src.get("instruction", "")
        })
    return out


# In[81]:


# =========================================================
# Evaluation on a dataframe with columns ["id", "question"]
# =========================================================
def evaluate_params_on(
    df_eval: pd.DataFrame,
    params: Dict,
    *,
    es_client,
    index_name,
    embeddings,
    id_field="id"
) -> Dict:
    mrrs, hits, latencies = [], [], []
    k = int(params["size"])
    for _, row in df_eval.iterrows():
        q = row["question"]
        true_id = row["id"]
        t0 = time.time()
        results = hybrid_search_with_ids(
            query=q,
            es_client=es_client,
            index_name=index_name,
            embeddings=embeddings,
            size=int(params["size"]),
            knn_k=int(params["knn_k"]),
            num_candidates=int(params["num_candidates"]),
            knn_boost=float(params["knn_boost"]),
            operator=params["operator"],
            use_rrf=bool(params["use_rrf"]),
            filters=params.get("filters"),
            id_field=id_field,
            rrf_rank_window_size=int(params.get("rrf_rank_window_size", 50)),
            rrf_rank_constant=int(params.get("rrf_rank_constant", 60)),
            b_exercise_name=float(params["b_exercise_name"]),
            b_type_of_activity=float(params["b_type_of_activity"]),
            b_type_of_equipment=float(params["b_type_of_equipment"]),
            b_body_part=float(params["b_body_part"]),
            b_type=float(params["b_type"]),
            b_muscle_groups=float(params["b_muscle_groups"]),
            b_instruction=float(params["b_instruction"]),
        )
        latencies.append(time.time() - t0)
        pred_ids = [r["id"] for r in results]
        mrrs.append(reciprocal_rank(pred_ids, true_id))
        hits.append(hit_at_k(pred_ids, true_id, k=k))

    return {
        "MRR": float(np.mean(mrrs)) if mrrs else 0.0,
        f"Hit@{k}": float(np.mean(hits)) if hits else 0.0,
        "AvgLatencySec": float(np.mean(latencies)) if latencies else 0.0,
        "P95LatencySec": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "N": int(len(df_eval))
    }


# In[83]:


# =========================================================
# Objective
# =========================================================
def make_objective(df_eval, es_client, index_name, embeddings, id_field="id"):
    def objective(hparams):
        p = dict(hparams)

        # Ensure ints for integer-like params
        for k in ("size", "knn_k", "num_candidates", "rrf_rank_window_size", "rrf_rank_constant"):
            p[k] = int(p[k])

        # Cast any numpy scalars to float for boosts & continuous params
        for k in ("knn_boost", "b_exercise_name", "b_type_of_activity", "b_type_of_equipment",
                  "b_body_part", "b_type", "b_muscle_groups", "b_instruction"):
            p[k] = float(p[k])

        metrics = evaluate_params_on(
            df_eval,
            p,
            es_client=es_client,
            index_name=index_name,
            embeddings=embeddings,
            id_field=id_field
        )
        # Loss: prioritize MRR; lightly penalize high latency
        loss = (1.0 - metrics["MRR"]) + 0.05 * metrics["AvgLatencySec"]
        return {"loss": loss, "status": STATUS_OK, "metrics": metrics, "params": p}
    return objective


# In[84]:


# =========================================================
# Run tuning on df_validation
# =========================================================
objective = make_objective(df_validation, es_client, index_name, embeddings, id_field="id")

trials = Trials()
best = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=80,  # adjust as you like
    trials=trials,
    rstate=np.random.default_rng(42)
)

# Decode hp.choice fields from "best"
best_decoded = dict(best)
best_decoded["operator"] = ["and", "or"][best["operator"]]
best_decoded["use_rrf"]   = [False, True][best["use_rrf"]]

# Cast numerics
for k in ("size", "knn_k", "num_candidates", "rrf_rank_window_size", "rrf_rank_constant"):
    best_decoded[k] = int(best_decoded[k])
for k in ("knn_boost", "b_exercise_name", "b_type_of_activity", "b_type_of_equipment",
          "b_body_part", "b_type", "b_muscle_groups", "b_instruction"):
    best_decoded[k] = float(best_decoded[k])

print("Best params:", best_decoded)

best_trial = sorted(trials.results, key=lambda r: r["loss"])[0]
print("Validation metrics:", best_trial["metrics"])


# In[85]:


# =========================================================
# Final evaluation on df_test
# =========================================================
test_metrics = evaluate_params_on(
    df_test,
    best_decoded,
    es_client=es_client,
    index_name=index_name,
    embeddings=embeddings,
    id_field="id"
)
print("Test metrics:", test_metrics)


# In[91]:


def hybrid_search_with_ids_improved(
    query: str,
    es_client=es_client,
    index_name=index_name,
    embeddings=embeddings,
    *,
    id_field: str = "id",
    filters: Optional[Dict] = None,
    # You can override these, but they default to your best params
    size: int = 9,
    operator: str = "or",
    knn_k: int = 41,
    num_candidates: int = 1763,
    knn_boost: float = 0.8389895612929347,
    # Best field boosts
    b_exercise_name: float = 3.423888532659442,
    b_muscle_groups: float = 4.0786853724241805,
    b_type_of_equipment: float = 1.730475772103169,
    b_type_of_activity: float = 0.9459499781180558,
    b_body_part: float = 1.1642902810438727,
    b_type: float = 1.3527056181134047,
    b_instruction: float = 0.4735048221260146,
) -> List[Dict[str, Any]]:
    """
    Hybrid (keyword + vector) ES search using tuned best parameters.
    Returns a list of dicts shaped for your RAG context:
      {'id', 'exercise_name', 'type_of_activity', 'type_of_equipment',
       'body_part', 'type', 'muscle_groups_activated', 'instruction'}
    """
    # --- Build boosted fields list from best params ---
    fields = [
        f"exercise_name^{b_exercise_name}",
        f"muscle_groups_activated^{b_muscle_groups}",
        f"type_of_equipment^{b_type_of_equipment}",
        f"type_of_activity^{b_type_of_activity}",
        f"body_part^{b_body_part}",
        f"type^{b_type}",
        f"instruction^{b_instruction}",
    ]

    # --- Optional filters merged into bool.filter ---
    es_bool_filter: List[Dict] = []
    if filters:
        if "bool" in filters and "filter" in filters["bool"]:
            es_bool_filter.extend(filters["bool"]["filter"])
        else:
            es_bool_filter.append(filters)

    # --- Vector for kNN branch ---
    vector = embeddings.embed_query(query)

    body: Dict = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "type": "cross_fields",
                            "operator": operator,  # tuned: 'or'
                            "fields": fields
                        }
                    }
                ],
                **({"filter": es_bool_filter} if es_bool_filter else {})
            }
        },
        "knn": {
            "field": "ei_vector",
            "query_vector": vector,
            "k": int(knn_k),                         # tuned: 41
            "num_candidates": int(num_candidates),   # tuned: 1763
            "boost": float(knn_boost),               # tuned: ~0.839
            **({"filter": es_bool_filter} if es_bool_filter else {})
        },
        "size": int(size)                            # tuned: 9
    }

    res = es_client.search(index=index_name, body=body)
    hits = res.get("hits", {}).get("hits", [])

    out: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source", {})
        out.append({
            "id": src.get(id_field, h.get("_id")),
            "exercise_name":           src.get("exercise_name", ""),
            "type_of_activity":        src.get("type_of_activity", ""),
            "type_of_equipment":       src.get("type_of_equipment", ""),
            "body_part":               src.get("body_part", ""),
            "type":                    src.get("type", ""),
            "muscle_groups_activated": src.get("muscle_groups_activated", ""),
            "instruction":             src.get("instruction", "")
        })
    return out


# ## RAG evaluation (LLM as a Judge)

# In[125]:


prompt2_template = """
You are an expert evaluator for a (RAG) system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# In[126]:


ground_truth[0]


# In[127]:


def rag(query):
    search_results = hybrid_search_with_ids_improved(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


# In[128]:


record = ground_truth[0]
question = record['question']
answer_llm = rag(question)


# In[129]:


hybrid_search_with_ids_improved('What equipment do I need to perform the Push-Up Hold exercise?')


# In[132]:


hybrid_search_with_ids_improved('How long should I perform the Feet-Elevated Glute Bridge, and what is the intensity level?')


# In[133]:


rag('How long should I perform the Feet-Elevated Glute Bridge, and what is the intensity level?')


# In[130]:


prompt = prompt2_template.format(question=question, answer_llm=answer_llm)
print(prompt)


# In[119]:


df_sample = df_question.sample(n=200, random_state=1)


# In[120]:


sample = df_sample.to_dict(orient='records')


# In[121]:


from openai import OpenAI

client = OpenAI()


# In[101]:


def llm(prompt):
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# In[102]:


evaluations = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question) 

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations.append((record, answer_llm, evaluation))


# In[105]:


df_eval = pd.DataFrame(evaluations, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[106]:


df_eval.relevance.value_counts(normalize=True)


# In[107]:


df_eval.to_csv('../data/rag-eval-gpt-4o-mini.csv', index=False)


# In[113]:


mask = df_eval['relevance'].eq('NON_RELEVANT')
qs = df_eval.loc[mask, 'question']

for q in qs.head():
    print(q)
    print('-' * 80)


# In[114]:


hybrid_search_with_ids_improved('How long should I perform the Feet-Elevated Glute Bridge, and what is the intensity level?')


# In[115]:


rag('How long should I perform the Feet-Elevated Glute Bridge, and what is the intensity level?')


# There is information about the exercise in the search result, but the gpt4o-mini considered this as NONE

# In[116]:


def llm(prompt,model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# In[134]:


def rag(query,model='gpt-4o-mini'):
    search_results = hybrid_search_with_ids_improved(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt,model=model)
    return answer


# In[140]:


rag('How long should I perform the Feet-Elevated Glute Bridge, and what is the intensity level?',model='gpt-4.1-mini')


# In[137]:


evaluations_gpt4_1 = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question, model='gpt-4.1') 

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations_gpt4_1.append((record, answer_llm, evaluation))


# In[141]:


df_eval = pd.DataFrame(evaluations_gpt4_1, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[142]:


df_eval.relevance.value_counts()


# In[143]:


df_eval.relevance.value_counts(normalize=True)


# In[144]:


df_eval.to_csv('../data/rag-eval-gpt-4_1.csv', index=False)


# In[ ]:




