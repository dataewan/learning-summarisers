from summarisers import summarise


def test_order_sentences():
    document = [
        {"label": "a"},
        {"label": "b"},
        {"label": "c"},
        {"label": "d"},
    ]

    pagerank_scores = {
        "a": 0.1,
        "b": 0.6,
        "c": 0.3,
        "d": 0.0,
    }

    expected = [
        {"label": "a", "importance": 2},
        {"label": "b", "importance": 0},
        {"label": "c", "importance": 1},
        {"label": "d", "importance": 3},
    ]

    
    ordered = summarise.order_sentences(document, pagerank_scores)

    assert ordered == expected
