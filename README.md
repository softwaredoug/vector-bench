
## Vector Bench

This is a set of tooling for benchmarking student vector search solutions.

It will take an application with index and search capabilities. It will index,
then it will search with the dataset queries, scoring recall and latency.

## The student's CLI

The student CLI works as follows:

```
cmd --documents --index <dataset.h5> --port 1234
```

The following is also acceptable:

```
cmd <dataset.h5> 1234 <results.csv>
```

The command outputs "READY" when the index is ready to accept queries.

At that point, its expected HTTP server at port 1234 can accept queries. The interface works via:

```
POST http://localhost:1234/query
query_id=<query_id>&vector=<comma_seperated_vector>
```

The result:

CSV of 

```
rank,query_id,doc_id
```

Optionally the vector can be appended, and will be ignored in evaluation.


## Demo CLI

There's a silly CLI in this repo that does brute-force vector search to implement these requirements.

For local experiments, the demo scripts also support a non-HTTP test loop. It
loads the index, repeatedly searches a randomly selected corpus vector, and
stops with Ctrl+C:

```
naive-vector-search --index corpus.h5 --dimensions 60 --test
```

Use `--test-max-index-size N` to index only the first `N` corpus vectors while
debugging:

```
naive-vector-search --index corpus.h5 --dimensions 60 --test --test-max-index-size 1000
```

The same `--test` option is available to other scripts in `exps/` that use the
shared launcher.


## Data prep CLI

A data prep task exists to construct embeddings + ground truth.

Why? To let us distribute index / ground truth with training without the student needing to embed everything. To avoid
keeping embeddings in memory during evaluation (hopefully only the student's own script needs to do this)

```
prepare --dataset msmarco --index-out corpus.h5 --queries-out queries.h5 --num-queries <N>
```

This prepares a portable HDF5 corpus of vectors to be indexed by the student's script and HDF5 query ground truth to replay. The project depends on `h5py` for this format.

The index HDF5 file contains:

```
doc_ids: one UTF-8 document ID per row
vectors: a two-dimensional numeric dataset aligned with doc_ids
```

The queries HDF5 file contains:

```
query_ids: one UTF-8 query ID per row
vectors: a two-dimensional numeric dataset aligned with query_ids
ground_truth: ranked document IDs, one row per query
```

The query rows are sorted by query ID and the ground truth columns are sorted by rank. Numeric datasets retain their original NumPy precision.

If --num-queries is specified, then only that many queries will be sampled from the dataset. Otherwise all queries will be used.


## Benchmark CLI

Benchmark CLI works as follows:

```
benchmark --index corpus.h5 --queries queries.h5 -- <student-cli>
```
