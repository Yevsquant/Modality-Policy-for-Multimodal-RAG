# Modality-Policy-for-Multimodal-RAG

Multimodal RAG baseline and evaluation; generation goes through a **vLLM** OpenAI-compatible API (see [`start_server.sh`](start_server.sh) with LMCache).

## KV caching and compression notes

Background on **ToMe**, **KV compression** (e.g. ZSMerge, KVMerger), **CacheBlend**, and how they relate to **vLLM + LMCache** in this project:

- [docs/kv_compression_tome_cacheblend.md](docs/kv_compression_tome_cacheblend.md)
