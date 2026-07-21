from vector_bench import main


def test_cli_embeds_selected_dataset_with_default_model(monkeypatch):
    calls = {}

    def fake_get_dataset(name):
        calls["dataset"] = name
        return "corpus", "judgments"

    def fake_embed_corpus(corpus, model_name):
        calls["corpus"] = corpus
        calls["model_name"] = model_name
        return "embeddings", "model"

    monkeypatch.setattr(main, "get_dataset", fake_get_dataset)
    monkeypatch.setattr(main, "embed_corpus", fake_embed_corpus)

    result = main.main(["--dataset", "msmarco"])

    assert result == ("corpus", "judgments", "embeddings", "model")
    assert calls == {
        "dataset": "msmarco",
        "corpus": "corpus",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    }


def test_embed_corpus_builds_title_description_passages(monkeypatch):
    captured = {}

    def fake_load_or_create_embeddings(corpus, passage_fn, **kwargs):
        captured["passages"] = [
            passage_fn({"title": "Title", "description": "Description"}),
            passage_fn({"title": "", "description": "Description only"}),
        ]
        captured["kwargs"] = kwargs
        return "embeddings", "model"

    import vector_bench.embeddings as embeddings

    monkeypatch.setattr(
        embeddings,
        "load_or_create_embeddings",
        fake_load_or_create_embeddings,
    )

    result = embeddings.embed_corpus("corpus", model_name="test-model")

    assert result == ("embeddings", "model")
    assert captured["passages"] == ["Title\n\nDescription", "Description only"]
