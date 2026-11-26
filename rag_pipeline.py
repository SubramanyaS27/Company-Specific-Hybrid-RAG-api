from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import time
from typing import Any
from urllib import error, request as urlrequest

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MAX_CONTEXT_TOKENS = 650
MAX_FRAMING_TOKENS = 250
MAX_GENERATION_TOKENS = 500
MAX_TOTAL_TOKENS = MAX_CONTEXT_TOKENS + MAX_FRAMING_TOKENS + MAX_GENERATION_TOKENS  # 1400 < 1500

VALID_ROLES = {"employee", "manager", "admin"}
ROLE_RANK = {"employee": 1, "manager": 2, "admin": 3}

ROLE_SCOPED_CACHE: dict[str, dict[str, Any]] = {}


@dataclass
class PipelineConfig:
    base_dir: Path = BASE_DIR
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
    cross_encoder_model: str = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
    tokenizer_name: str = os.getenv("TOKENIZER_NAME", "sentence-transformers/all-MiniLM-L6-v2").strip()
    hf_inference_base: str = os.getenv("HF_INFERENCE_BASE", "https://router.huggingface.co/hf-inference/models").rstrip("/")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "").strip()
    candidate_k: int = 8
    top_pool: int = 4
    min_ce_score: float = -10.0
    max_sentences: int = 6
    max_context_tokens: int = MAX_CONTEXT_TOKENS
    max_framing_tokens: int = MAX_FRAMING_TOKENS
    max_generation_tokens: int = MAX_GENERATION_TOKENS


def _get_hf_token() -> str:
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing in environment / Render Environment Variables.")
    return token


def _hf_post_json(model_id: str, payload: dict[str, Any], timeout: int = 30) -> Any:
    token = _get_hf_token()
    url = f"{CONFIG.hf_inference_base}/{model_id}"
    last_err: Exception | None = None

    for attempt in range(3):
        req = urlrequest.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "Southwind-Hybrid-RAG/1.0",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 503 and attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                last_err = RuntimeError(f"HF Inference 503 loading model '{model_id}': {err_body}")
                continue
            last_err = RuntimeError(f"HF Inference HTTP {exc.code} for '{model_id}': {err_body}")
            raise last_err from exc
        except Exception as exc:
            last_err = exc
            if attempt < 2:
                time.sleep(1.0)
                continue
            raise

    raise RuntimeError(f"HF Inference failed for '{model_id}': {last_err}")


def _mean_pool_and_normalize(raw_vec: Any) -> list[float]:
    if isinstance(raw_vec, list) and raw_vec and isinstance(raw_vec[0], list):
        dim = len(raw_vec[0])
        pooled = [0.0] * dim
        count = 0
        for row in raw_vec:
            if isinstance(row, list) and len(row) == dim:
                for i, val in enumerate(row):
                    pooled[i] += float(val)
                count += 1
        vec = [v / max(count, 1) for v in pooled]
    elif isinstance(raw_vec, list):
        vec = [float(v) for v in raw_vec]
    else:
        raise RuntimeError(f"Unexpected embedding response shape: {type(raw_vec)}")

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def embed_query_remote(text: str) -> list[float]:
    data = _hf_post_json(
        CONFIG.embedding_model,
        {"inputs": text, "options": {"wait_for_model": True}},
    )
    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], list)
        and (not data[0] or not isinstance(data[0][0], list))
        and len(data) == 1
        and len(data[0]) > 16
    ):
        return _mean_pool_and_normalize(data[0])
    return _mean_pool_and_normalize(data)


def rerank_pairs_remote(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []

    pairs = [[query, p[:1500]] for p in passages]
    try:
        data = _hf_post_json(
            CONFIG.cross_encoder_model,
            {"inputs": pairs, "options": {"wait_for_model": True}},
        )
        scores: list[float] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, (int, float)):
                    scores.append(float(item))
                elif isinstance(item, list) and item:
                    best = max(item, key=lambda x: float(x.get("score", 0.0)) if isinstance(x, dict) else 0.0)
                    scores.append(float(best.get("score", 0.0)) if isinstance(best, dict) else float(best))
                elif isinstance(item, dict) and "score" in item:
                    scores.append(float(item["score"]))
        if len(scores) == len(passages):
            return scores
    except Exception:
        pass

    # Fallback lexical overlap + position prior if remote cross-encoder returns non-matching shape/status
    q_terms = {w.lower() for w in re.findall(r"\w{2,}", query)}
    fallback_scores: list[float] = []
    for idx, p in enumerate(passages):
        p_terms = {w.lower() for w in re.findall(r"\w{2,}", p)}
        overlap = len(q_terms & p_terms)
        pos_prior = max(0.0, 1.0 - (idx * 0.1))
        fallback_scores.append(round(overlap * 1.5 + pos_prior, 4))
    return fallback_scores


class _RuntimeAssets:
    """Lightweight singleton container for indexes, tokenizer, and Qdrant/BM25 stores."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._loaded = False
        self.tokenizer: Any = None
        self.qdrant_client: Any = None
        self.collection_name: str | None = None
        self.bm25_payload: Any = None
        self.corpus_records: list[dict[str, Any]] = []

    def ensure_loaded(self) -> None:
        if self._loaded:
            return

        from qdrant_client import QdrantClient

        try:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_pretrained(self.cfg.tokenizer_name)
        except Exception:
            self.tokenizer = None

        qdrant_candidates = [
            p for p in self.cfg.base_dir.iterdir()
            if p.is_dir() and ("qdrant" in p.name.lower() or (p / "meta.json").exists())
        ]
        if qdrant_candidates:
            self.qdrant_client = QdrantClient(path=str(qdrant_candidates[0]))
            cols = [c.name for c in self.qdrant_client.get_collections().collections]
            if self.cfg.qdrant_collection and self.cfg.qdrant_collection in cols:
                self.collection_name = self.cfg.qdrant_collection
            elif cols:
                self.collection_name = cols[0]

        bm25_candidates = list(self.cfg.base_dir.glob("*bm25*.pkl")) + list(self.cfg.base_dir.glob("*.pkl"))
        seen_pkl: set[Path] = set()
        for pkl_path in bm25_candidates:
            if pkl_path in seen_pkl:
                continue
            seen_pkl.add(pkl_path)
            with open(pkl_path, "rb") as f:
                loaded_obj = pickle.load(f)
            self.bm25_payload = loaded_obj
            if isinstance(loaded_obj, dict):
                for key in ("chunks", "records", "documents", "corpus", "items"):
                    if isinstance(loaded_obj.get(key), list):
                        self.corpus_records = loaded_obj[key]
                        break
            elif isinstance(loaded_obj, list):
                self.corpus_records = loaded_obj
            if self.corpus_records:
                break

        self._loaded = True


CONFIG = PipelineConfig()
ASSETS = _RuntimeAssets(CONFIG)


def count_tokens(text: str) -> int:
    ASSETS.ensure_loaded()
    if not text:
        return 0
    if ASSETS.tokenizer is not None:
        return len(ASSETS.tokenizer.encode(text).ids)
    return max(1, (len(text) + 3) // 4)


def _role_allowed(doc_role: str | None, user_role: str) -> bool:
    if not doc_role:
        return True
    u_rank = ROLE_RANK.get(user_role.lower(), 1)
    d_str = str(doc_role).lower().strip()
    if d_str in ROLE_RANK:
        return u_rank >= ROLE_RANK[d_str]
    return user_role.lower() in {r.strip() for r in d_str.split(",")}


def _normalize_record(raw: dict[str, Any], idx: int, score: float = 0.0, source: str = "hybrid") -> dict[str, Any]:
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else raw
    text = raw.get("text") or raw.get("page_content") or raw.get("content") or ""
    chunk_id = str(raw.get("chunk_id") or meta.get("chunk_id") or raw.get("id") or f"chk_{idx}")
    doc_id = str(raw.get("doc_id") or meta.get("doc_id") or meta.get("source_doc") or "DOC")
    section = str(raw.get("section") or meta.get("section") or meta.get("page_label") or "General")
    allowed_role = meta.get("role") or meta.get("access_role") or meta.get("rbac_role") or raw.get("role")
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "section": section,
        "allowed_role": allowed_role,
        "text": text.strip(),
        "score": float(score),
        "retrieval_source": source,
        "initial_citation_tag": f"[{chunk_id} | {doc_id} | {section}]",
    }


def hybrid_search(query: str, role: str, top_k: int = 8, candidate_k: int = 8) -> list[dict[str, Any]]:
    ASSETS.ensure_loaded()
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Expected one of {sorted(VALID_ROLES)}.")

    rrf_scores: dict[str, float] = {}
    records_by_id: dict[str, dict[str, Any]] = {}
    rrf_k = 60.0

    # 1. Dense Qdrant search via remote HF embedding
    if ASSETS.qdrant_client is not None and ASSETS.collection_name:
        q_vec = embed_query_remote(query)
        search_res = ASSETS.qdrant_client.query_points(
            collection_name=ASSETS.collection_name,
            query=q_vec,
            limit=max(candidate_k * 2, 16),
            with_payload=True,
        )
        points = getattr(search_res, "points", search_res)
        rank = 0
        for pt in points:
            payload = dict(pt.payload or {})
            rec = _normalize_record(payload, idx=rank, score=float(getattr(pt, "score", 0.0)), source="dense")
            if not rec["text"] or not _role_allowed(rec["allowed_role"], role):
                continue
            rank += 1
            cid = rec["chunk_id"]
            records_by_id[cid] = rec
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    # 2. Lexical BM25 search
    bm25_obj = None
    corpus_docs = ASSETS.corpus_records
    if isinstance(ASSETS.bm25_payload, dict):
        bm25_obj = ASSETS.bm25_payload.get("bm25") or ASSETS.bm25_payload.get("index")
        if not corpus_docs and isinstance(ASSETS.bm25_payload.get("texts"), list):
            corpus_docs = [{"text": t, "chunk_id": f"bm25_{i}"} for i, t in enumerate(ASSETS.bm25_payload["texts"])]

    if bm25_obj is not None and corpus_docs:
        tokens = re.findall(r"\w+", query.lower())
        scores = bm25_obj.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(candidate_k * 2, 16)]
        rank = 0
        for idx in ranked_indices:
            raw_item = corpus_docs[idx] if isinstance(corpus_docs[idx], dict) else {"text": str(corpus_docs[idx]), "chunk_id": f"bm25_{idx}"}
            rec = _normalize_record(raw_item, idx=idx, score=float(scores[idx]), source="bm25")
            if not rec["text"] or not _role_allowed(rec["allowed_role"], role):
                continue
            rank += 1
            cid = rec["chunk_id"]
            if cid not in records_by_id:
                records_by_id[cid] = rec
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    merged = []
    for cid, r_score in sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        item = dict(records_by_id[cid])
        item["rrf_score"] = r_score
        merged.append(item)
    return merged


def deduplicate_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for h in hits:
        cid = h.get("chunk_id")
        norm_txt = re.sub(r"\s+", " ", (h.get("text") or "").strip().lower())
        if not norm_txt or cid in seen_ids or norm_txt in seen_texts:
            continue
        if cid:
            seen_ids.add(cid)
        seen_texts.add(norm_txt)
        deduped.append(h)
    return deduped


def select_high_quality_pool(
    rescored_hits: list[dict[str, Any]],
    top_pool: int = CONFIG.top_pool,
    min_ce_score: float = CONFIG.min_ce_score,
) -> list[dict[str, Any]]:
    ordered = sorted(rescored_hits, key=lambda x: x.get("ce_score", x.get("rrf_score", 0.0)), reverse=True)
    filtered = [h for h in ordered if h.get("ce_score", 0.0) >= min_ce_score]
    return filtered[:top_pool] if filtered else ordered[:top_pool]


def compress_chunk_text(text: str, query: str, max_sentences: int = CONFIG.max_sentences) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    q_terms = {w.lower() for w in re.findall(r"\w{3,}", query)}
    scored = []
    for idx, sent in enumerate(sentences):
        s_terms = {w.lower() for w in re.findall(r"\w{3,}", sent)}
        overlap = len(q_terms & s_terms)
        position_bonus = 0.25 if idx == 0 else 0.0
        scored.append((overlap + position_bonus, idx, sent))

    top_sents = sorted(scored, key=lambda t: (-t[0], t[1]))[:max_sentences]
    top_sents_in_order = sorted(top_sents, key=lambda t: t[1])
    return " ".join(sent for _, _, sent in top_sents_in_order)


def _fit_text_to_token_budget(text: str, query: str, token_budget: int) -> str:
    ASSETS.ensure_loaded()
    if token_budget <= 0 or not text:
        return ""
    if count_tokens(text) <= token_budget:
        return text
    for n_sent in (5, 4, 3, 2, 1):
        cand = compress_chunk_text(text, query=query, max_sentences=n_sent)
        if cand and count_tokens(cand) <= token_budget:
            return cand
    if ASSETS.tokenizer is not None:
        token_ids = ASSETS.tokenizer.encode(text).ids[:token_budget]
        return ASSETS.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return text[: token_budget * 4].strip()


def build_adaptive_context(
    query: str,
    role: str,
    candidate_k: int = CONFIG.candidate_k,
    top_pool: int = CONFIG.top_pool,
    min_ce_score: float = CONFIG.min_ce_score,
    max_context_tokens: int = CONFIG.max_context_tokens,
    max_framing_tokens: int = CONFIG.max_framing_tokens,
    max_generation_tokens: int = CONFIG.max_generation_tokens,
    max_sentences: int = CONFIG.max_sentences,
) -> dict[str, Any]:
    ASSETS.ensure_loaded()
    raw_hits = hybrid_search(query, role=role, top_k=candidate_k, candidate_k=candidate_k)
    deduped = deduplicate_hits(raw_hits)

    rescored: list[dict[str, Any]] = []
    if deduped:
        ce_scores = rerank_pairs_remote(query, [h["text"] for h in deduped])
        for h, score in zip(deduped, ce_scores):
            item = dict(h)
            item["ce_score"] = float(score)
            rescored.append(item)

    hq_pool = select_high_quality_pool(rescored, top_pool=top_pool, min_ce_score=min_ce_score)

    def _pack(pool_items: list[dict[str, Any]], use_compression: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        packed: list[dict[str, Any]] = []
        pruned: list[dict[str, Any]] = []
        tokens_used = 0

        for item in pool_items:
            cite_num = len(packed) + 1
            cite_id = f"S{cite_num}"
            citation_tag = f"[{cite_id}: {item['chunk_id']} | {item['doc_id']} | {item['section']}]"
            header_tokens = count_tokens(f"{citation_tag}\n")
            remaining_budget = max_context_tokens - tokens_used - header_tokens

            if remaining_budget <= 15:
                pruned.append({
                    "dropped_initial_tag": item["initial_citation_tag"],
                    "chunk_id": item["chunk_id"],
                    "doc_id": item["doc_id"],
                    "ce_score": round(item.get("ce_score", 0.0), 3),
                    "reason": "context_token_budget_exhausted",
                })
                continue

            if use_compression:
                raw_body = compress_chunk_text(item["text"], query=query, max_sentences=max_sentences)
                chunk_body = _fit_text_to_token_budget(raw_body, query=query, token_budget=remaining_budget)
            else:
                chunk_body = item["text"]

            block = f"{citation_tag}\n{chunk_body}"
            block_tokens = count_tokens(block)

            if chunk_body and tokens_used + block_tokens <= max_context_tokens:
                entry = dict(item)
                entry["cite_id"] = cite_id
                entry["citation_tag"] = citation_tag
                entry["final_text"] = chunk_body
                entry["block_tokens"] = block_tokens
                packed.append(entry)
                tokens_used += block_tokens
            else:
                pruned.append({
                    "dropped_initial_tag": item["initial_citation_tag"],
                    "chunk_id": item["chunk_id"],
                    "doc_id": item["doc_id"],
                    "ce_score": round(item.get("ce_score", 0.0), 3),
                    "reason": "exceeded_context_token_budget",
                })

        return packed, pruned, tokens_used

    packed_chunks, pruned_chunks, used_tokens = _pack(hq_pool, use_compression=False)
    compression_mode = "full_uncompressed"

    if pruned_chunks:
        comp_packed, comp_pruned, comp_tokens = _pack(hq_pool, use_compression=True)
        if comp_tokens <= max_context_tokens and (
            len(comp_packed) > len(packed_chunks)
            or (len(comp_packed) == len(packed_chunks) and comp_tokens >= used_tokens)
        ):
            packed_chunks, pruned_chunks, used_tokens = comp_packed, comp_pruned, comp_tokens
            compression_mode = "query_aware_compressed"

    assert used_tokens <= max_context_tokens, f"Context budget breached: {used_tokens} > {max_context_tokens}"

    final_context = "\n\n---\n\n".join(
        f"{c['citation_tag']}\n{c['final_text']}" for c in packed_chunks
    )

    return {
        "role": role,
        "query": query,
        "compression_mode": compression_mode,
        "packed_count": len(packed_chunks),
        "used_context_tokens": used_tokens,
        "max_context_tokens": max_context_tokens,
        "max_framing_tokens": max_framing_tokens,
        "max_generation_tokens": max_generation_tokens,
        "total_planned_tokens": used_tokens + max_framing_tokens + max_generation_tokens,
        "active_citations": [c["citation_tag"] for c in packed_chunks],
        "pruned_chunks": pruned_chunks,
        "packed_chunks": packed_chunks,
        "context": final_context,
    }


def make_cache_key(role_scope: str, query: str, packed_chunks: list[dict[str, Any]]) -> str:
    norm_q = re.sub(r"\s+", " ", query.strip().lower())
    chunk_sig = ",".join(
        f"{c.get('cite_id', '')}:{c.get('chunk_id', '')}:{len(c.get('final_text', ''))}"
        for c in packed_chunks
    )
    raw = f"{role_scope}||{norm_q}||{chunk_sig}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sanitize_answer_citations(answer_text: str, valid_cite_ids: set[str]) -> str:
    def _replace_tag(match: re.Match[str]) -> str:
        cite_id = match.group(1)
        return f"[{cite_id}]" if cite_id in valid_cite_ids else ""

    cleaned = re.sub(r"\[(S\d+)\]", _replace_tag, answer_text)
    cleaned = re.sub(r"\s+([.,;!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def call_groq_chat(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = CONFIG.max_generation_tokens,
    model: str | None = None,
) -> dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing in environment / .env.")

    chosen_model = (model or os.getenv("GROQ_MODEL") or CONFIG.groq_model or "openai/gpt-oss-20b").strip()
    payload: dict[str, Any] = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "reasoning_effort": "low",
    }

    def _post(body_dict: dict[str, Any]) -> dict[str, Any]:
        req = urlrequest.Request(
            CONFIG.groq_api_url,
            data=json.dumps(body_dict).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Southwind-Hybrid-RAG/1.0",
            },
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        data = _post(payload)
    except error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 400 and "reasoning_effort" in err_body:
            fallback_payload = dict(payload)
            fallback_payload.pop("reasoning_effort", None)
            data = _post(fallback_payload)
        else:
            raise RuntimeError(f"Groq API HTTP {exc.code}: {err_body}") from exc

    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    usage = data.get("usage") or {}
    return {
        "content": content,
        "model": chosen_model,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    }


def run_rag_query(
    query: str,
    role: str,
    use_cache: bool = True,
    short_circuit_cache: bool = False,
) -> dict[str, Any]:
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Expected one of {sorted(VALID_ROLES)}.")

    t_start = time.perf_counter()

    fast_key = hashlib.sha256(
        f"{role}||{re.sub(r'\\s+', ' ', query.strip().lower())}".encode("utf-8")
    ).hexdigest()
    if short_circuit_cache and use_cache and fast_key in ROLE_SCOPED_CACHE:
        cached = dict(ROLE_SCOPED_CACHE[fast_key])
        cached["cache_hit"] = True
        cached["telemetry"] = {
            "retrieval_rerank_ms": 0.0,
            "context_pack_ms": 0.0,
            "llm_ms": 0.0,
            "total_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
        }
        return cached

    t0 = time.perf_counter()
    raw_hits = hybrid_search(query, role=role, top_k=CONFIG.candidate_k, candidate_k=CONFIG.candidate_k)
    deduped = deduplicate_hits(raw_hits)
    rescored: list[dict[str, Any]] = []
    if deduped:
        ce_scores = rerank_pairs_remote(query, [h["text"] for h in deduped])
        for h, score in zip(deduped, ce_scores):
            item = dict(h)
            item["ce_score"] = float(score)
            rescored.append(item)
    t_retrieval_rerank = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    hq_pool = select_high_quality_pool(rescored, top_pool=CONFIG.top_pool, min_ce_score=CONFIG.min_ce_score)
    ctx_bundle = build_adaptive_context(
        query=query,
        role=role,
        candidate_k=CONFIG.candidate_k,
        top_pool=len(hq_pool) if hq_pool else CONFIG.top_pool,
        min_ce_score=CONFIG.min_ce_score,
        max_context_tokens=CONFIG.max_context_tokens,
        max_framing_tokens=CONFIG.max_framing_tokens,
        max_generation_tokens=CONFIG.max_generation_tokens,
        max_sentences=CONFIG.max_sentences,
    )
    t_pack = (time.perf_counter() - t1) * 1000.0

    cache_key = make_cache_key(
        f"{role}|mode={ctx_bundle['compression_mode']}",
        query,
        ctx_bundle["packed_chunks"],
    )

    if use_cache and cache_key in ROLE_SCOPED_CACHE:
        cached = dict(ROLE_SCOPED_CACHE[cache_key])
        cached["cache_hit"] = True
        cached["telemetry"] = {
            "retrieval_rerank_ms": round(t_retrieval_rerank, 2),
            "context_pack_ms": round(t_pack, 2),
            "llm_ms": 0.0,
            "total_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
        }
        return cached

    t2 = time.perf_counter()
    valid_ids = {c["cite_id"] for c in ctx_bundle["packed_chunks"]}
    system_prompt = (
        f"You are a role-aware assistant for Southwind Labs answering for role='{role}'. "
        "Use ONLY the provided context blocks. Provide a clear, complete answer grounded in the context, "
        "and cite sources inline strictly in brackets like [S1], [S2]. If restricted or absent from context, state that clearly."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Authorized Context:\n{ctx_bundle['context'] if ctx_bundle['context'] else '(No authorized context)'}"
    )

    llm_resp = call_groq_chat(system_prompt, user_prompt, max_tokens=CONFIG.max_generation_tokens)
    clean_answer = sanitize_answer_citations(llm_resp["content"], valid_ids)
    t_llm = (time.perf_counter() - t2) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    result: dict[str, Any] = {
        "role": role,
        "query": query,
        "cache_hit": False,
        "compression_mode": ctx_bundle["compression_mode"],
        "used_context_tokens": ctx_bundle["used_context_tokens"],
        "max_context_tokens": ctx_bundle["max_context_tokens"],
        "max_generation_tokens": ctx_bundle["max_generation_tokens"],
        "total_planned_tokens": ctx_bundle["total_planned_tokens"],
        "active_citations": ctx_bundle["active_citations"],
        "pruned_chunks": ctx_bundle["pruned_chunks"],
        "answer": clean_answer,
        "usage": llm_resp["usage"],
        "telemetry": {
            "retrieval_rerank_ms": round(t_retrieval_rerank, 2),
            "context_pack_ms": round(t_pack, 2),
            "llm_ms": round(t_llm, 2),
            "total_ms": round(total_ms, 2),
        },
    }

    if use_cache:
        ROLE_SCOPED_CACHE[cache_key] = dict(result)
        if short_circuit_cache:
            ROLE_SCOPED_CACHE[fast_key] = dict(result)

    return result


if __name__ == "__main__":
    sample = run_rag_query(
        "What are the rules for leave approval and laptop security?",
        role="employee",
        use_cache=False,
    )
    print(f"Role={sample['role']} | Mode={sample['compression_mode']} | Tokens={sample['used_context_tokens']}/{sample['total_planned_tokens']}")
    print("Telemetry:", sample["telemetry"])
    print("Answer:\n", sample["answer"])