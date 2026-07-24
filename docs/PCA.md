# PCA Experiment

This experiment projects the original embedding vectors into a lower-dimensional
PCA space, then performs brute-force dot-product search in that space. Running
the benchmark at several projection sizes shows how much retrieval recall is
retained as fewer dimensions are used.

The implementation is in [`exps/pca.py`](../exps/pca.py).

## Prepare Embeddings

The commands below prepare 1,000 MS MARCO queries and retain the top 50
ground-truth documents for each query. The first command uses E5-base-v2:

```bash
uv run prepare \
  --dataset msmarco \
  --model intfloat/e5-base-v2 \
  --top-k 50 \
  --num-queries 1000 \
  --index-out corpus_e5.hd5 \
  --queries-out queries_e5.hd5
```

To prepare the same dataset with the MiniLM model instead:

```bash
uv run prepare \
  --dataset msmarco \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --top-k 50 \
  --num-queries 1000 \
  --index-out corpus_minilm.hd5 \
  --queries-out queries_minilm.hd5
```

The models are downloaded from Hugging Face the first time they are used.

## Run PCA

Run the MiniLM artifacts with a 50-dimensional PCA projection:

```bash
uv run vector-bench \
  --index corpus_minilm.hd5 \
  --queries queries_minilm.hd5 \
  -- uv run python -m exps.pca --dimensions 50
```

Repeat with 100 dimensions:

```bash
uv run vector-bench \
  --index corpus_minilm.hd5 \
  --queries queries_minilm.hd5 \
  -- uv run python -m exps.pca --dimensions 100
```

Repeat with 200 dimensions:

```bash
uv run vector-bench \
  --index corpus_minilm.hd5 \
  --queries queries_minilm.hd5 \
  -- uv run python -m exps.pca --dimensions 200
```

The benchmark reports recall@50 by default. Increasing the PCA dimensions
keeps more information from the original vectors, so recall will generally
increase, at the cost of more memory and slower scoring.
