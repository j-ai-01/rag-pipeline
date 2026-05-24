import pytest
from unittest.mock import MagicMock
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from utils.hybrid_retriever import HybridRetriever


def _node(node_id: str, score: float) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=f"text {node_id}", id_=node_id), score=score)


def _retriever(nodes):
    mock = MagicMock()
    mock.retrieve.return_value = nodes
    return mock


def _query():
    return QueryBundle(query_str="test query")


def test_invalid_alpha_raises_value_error():
    with pytest.raises(ValueError, match="alpha"):
        HybridRetriever(_retriever([]), None, alpha=1.5)


def test_alpha_below_zero_raises_value_error():
    with pytest.raises(ValueError, match="alpha"):
        HybridRetriever(_retriever([]), None, alpha=-0.1)


def test_none_bm25_returns_vector_results_only():
    nodes = [_node("a", 0.9), _node("b", 0.5)]
    hybrid = HybridRetriever(_retriever(nodes), None, alpha=0.5, top_k=5)
    result = hybrid._retrieve(_query())
    ids = [n.node.node_id for n in result]
    assert set(ids) == {"a", "b"}


def test_scores_normalized_to_0_1():
    nodes = [_node("a", 0.0), _node("b", 100.0)]
    hybrid = HybridRetriever(_retriever(nodes), None, alpha=1.0, top_k=5)
    result = hybrid._retrieve(_query())
    for n in result:
        assert 0.0 <= (n.score or 0.0) <= 1.0


def test_deduplicates_node_appearing_in_both_retrievers():
    shared = _node("shared", 0.8)
    vector_nodes = [shared, _node("v_only", 0.5)]
    bm25_nodes = [_node("shared", 0.7), _node("b_only", 0.6)]
    hybrid = HybridRetriever(
        _retriever(vector_nodes), _retriever(bm25_nodes), alpha=0.5, top_k=10
    )
    result = hybrid._retrieve(_query())
    ids = [n.node.node_id for n in result]
    assert ids.count("shared") == 1


def test_top_k_limits_results():
    nodes = [_node(str(i), float(i)) for i in range(10)]
    hybrid = HybridRetriever(_retriever(nodes), None, alpha=1.0, top_k=3)
    result = hybrid._retrieve(_query())
    assert len(result) == 3


def test_all_retrievers_contribute_with_equal_alpha():
    vector_nodes = [_node("v", 1.0)]
    bm25_nodes = [_node("b", 1.0)]
    hybrid = HybridRetriever(
        _retriever(vector_nodes), _retriever(bm25_nodes), alpha=0.5, top_k=10
    )
    result = hybrid._retrieve(_query())
    ids = [n.node.node_id for n in result]
    assert "v" in ids
    assert "b" in ids
