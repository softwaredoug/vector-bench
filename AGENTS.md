This is a python repo managed by uv

# Mandatory reading:

Examine the README.md before proceeding.

# Development Practices

## Depedencies

This project depends on the cheat-at-search library located here for datasets and utilities:
https://github.com/softwaredoug/cheat-at-search

Install this directly from git.

## How to develop code

When you're asked to make a functional change, follow a TDD flow:

1. ALWAYS create an e2e test
2. Run the test, ensure it fails
3. Make the functional change
4. Run the test, ensure it passes

## What is a good e2e test?

e2e tests 'look like' calling the application from the command line. Specifically, for this application call
`main` function with the correct command line arguments.

e2e tests should test as much functionality as possible. As cheat-at-search is a very close dependency, its
also ok to include this in the testing boundary. 

The best dataset to use for testing is dougs_blog_data, which has a small number of blog posts and queries.

We generally want to avoid mocking, except when a network call is made or filesystem interaction occurs.

## What's in a unit test?

e2e tests are strict and behavioral of the full system. Unit tests are open ended, heavily mocked, and down 
to individual classes, functions, modules, tec

You're free to mock dependencies as needed more freely.

## Python testing practices

- use unittest.patch decorator for mocking
- use pytest fixtures as needed

## Co-authoring with the human

A human being may have working changes in this repo. Don't overwrite those working changes without asking.

## Brevity

Keep it brief and don't be too chatty as we're paying you by the token :)

## Pre-commit, etc

A pre-commit hook runs on commits. Fix the problems it finds before committing.

If the problem is in a test or check, not the actual code, flag to the human and propose a fix. Don't just 'fix tests' without approval.

# Architecture

As stated in the README, this command launches the command after --. It calls that command with --index to index the docs. And waits for the "READY" message. Then it issues queries to the command and collects results.

## Before launching - embedding

The --dataset param specifies a param (one of legal values in the dataset module). We load the corpus / judgments for the dataset.

### Embeddings generation

Then another module, embeddings, uses the [cheat-at-search embeddings utilities](https://github.com/softwaredoug/cheat-at-search/blob/main/cheat_at_search/embeddings.py) to create embeddings. This module requires a `passage_fn` param to construct an embedding. We should use the following:

```
def passage_fn(row):
    title = row.get("title")
    description = row.get("description", "")

    if title:
        return f"{title}\n\n{description}"
    return description
```

Basically, prepend the title / description.

### Embeddings ground-truth generation

As part of the embeddings generation, we also need to generate true embedding rankings for each query in judgments. The embeddings module also have a `ground_truth` method.

The `ground_truth` method:

1. Take as input the in-memory embeddings from the previous step, take judgmetns

For each query in judgments, append to a single numpy array so that (dim 0 - query_id, dim1 - doc_id, dom2...dimN - embedding vector). Then for each query, compute the cosine similarity between the query and all documents. Then rank the documents by cosine similarity, and return a dict of query_id -> list of doc_ids.

    1. Generates a query embedding
    2. Compute true cosine-similarity (np.dot) via these two
    3. Concat a ranked set of N doc_ids
    4. Cache the N doc_ids in a file

If cached docs already exist, then return cached ground truth.

### Embeddings file output

Part of embeddings module, it will return CSV encoded lines as follows:

```
<doc_id>,<comma_separated_vector>
```

Then the main process writes this out to a csv file to pass to the users command line tool.

### Launching the student command line tool

The student command line tool will be launched, in a different process. It will receive the 
embeddings CSV file, and the port to listen on. It will be expected to index the embeddings and then output "READY" to stdout.

Once READY occurs on stdout, the parent process then continues.

## Searching and computing recall and latency

Search is its own module, search, that will issue searches to a ready app and compute recall

We search by POST'ing in the body to a query endpoint expected.

The POST body will be form-encoded, with the following params:

http://localhost:<port>/query
query_id=<query_id>&vector=<comma_seperated_vector>

Once we search per query, we can compute recall@N by comparing the returned doc_ids to the ground truth doc_ids.

The proportion out of the retrieved N doc_ids out of true top N is the recall@N score.

We track latency as the time between sending the POST and receiving the response using perf_counter

Output to stdout in the format:

query_id,latency,recall@N

With final line average:

,average_latency,average_recall@N

## Terminating and cleanup

Once the search is complete, we terminate the student command line tool and cleanup any temporary files.


# Naive vector search application

For testing, we should create a simple, naive search application that uses brute-force numpy.

This application should

- Use only the first 20 dimensions provided
- Store a numpy array in memory
- Use dot product between the query-document when querying

It should follow the different command line interface requirements as specified in the README.md for student applications.

It also should act as a scaffold for the students during their training.

When testing this e2e, you can interact with this application directly. No need to call main.
