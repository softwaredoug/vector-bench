"""Dataset registry for vector-bench."""

from types import ModuleType

from cheat_at_search import doug_blog_data, msmarco_data


DATASETS: dict[str, ModuleType] = {
    "dougs_blog_data": doug_blog_data,
    "msmarco": msmarco_data,
}


def get_dataset(name: str) -> tuple[object, object]:
    """Return the corpus and judgments for a registered dataset."""
    key = name.lower()
    try:
        dataset = DATASETS[key]
    except KeyError as error:
        available = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {name!r}; choose from: {available}") from error

    return dataset.corpus, dataset.judgments
