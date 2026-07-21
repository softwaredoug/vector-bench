
## Vector Bench

This is a set of tooling for benchmarking student vector search solutions.

It will take an application with index and search capabilities. It will index,
then it will search with the dataset queries, scoring recall and latency.

## The student's CLI

The student CLI works as follows:

```
cmd --documents --index <dataset.csv> --port 1234 --results <results.csv>
```

The following is also acceptable:

```
cmd <dataset.csv> 1234 <results.csv>
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


## Benchmark CLI

Benchmark CLI works as follows:

```
benchmark --dataset <msmarco,...> --model <minilm-l6-v2...> -- <student-cli>
```

Benchmark CLI then loads the dataset (via cheat-at-search), loads the model via sentence transformers, and creates CSVs. 
It calls the student CLI once "READY" is in stdout. It then issues queries, gathers results, and computes latency and recall.
