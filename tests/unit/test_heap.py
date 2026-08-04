from exps.utils.heap import MaxHeap


def test_heap():
    heap = MaxHeap(max_size=3)

    heap.pushpop((1.0, "doc-a"))
    heap.pushpop((2.0, "doc-b"))
    heap.pushpop((0.5, "doc-c"))

    sorted_items = heap.sorted
    assert sorted_items == [(2.0, "doc-b"), (1.0, "doc-a"), (0.5, "doc-c")]


def test_heap_overwrites_min():
    heap = MaxHeap(max_size=3)

    heap.pushpop((1.0, "doc-a"))
    heap.pushpop((2.0, "doc-b"))
    heap.pushpop((0.5, "doc-c"))
    heap.pushpop((3.0, "doc-d"))

    sorted_items = heap.sorted
    assert sorted_items == [(3.0, "doc-d"),
                            (2.0, "doc-b"),
                            (1.0, "doc-a")]


def test_heap_returns_popped_item():
    heap = MaxHeap(max_size=3)

    heap.pushpop((1.0, "doc-a"))
    heap.pushpop((2.0, "doc-b"))
    heap.pushpop((0.5, "doc-c"))
    popped = heap.pushpop((3.0, "doc-d"))

    assert popped == (0.5, "doc-c")
