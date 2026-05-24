import pytest
from unittest.mock import MagicMock
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from utils.multi_index_retriever import MultiIndexRetriever


def _node(node_id: str, score: float) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=f"text {node_id}", id_=node_id), score=score)


def _retriever(nodes):
    mock = MagicMock()
    mock.retrieve.return_value = nodes
    return mock


def _query():
    return QueryBundle(query_str="test query")


def test_single_index_returns_results():
    nodes = [_node("a", 0.9), _node("b", 0.5)]
    retriever = MultiIndexRetriever([("finance", _retriever(nodes))], top_k=5)
    result = retriever._retrieve(_query())
    ids = [n.node.node_id for n in result]
    assert "a" in ids
    assert "b" in ids


def test_index_name_stamped_on_node_metadata():
    nodes = [_node("a", 0.9)]
    retriever = MultiIndexRetriever([("finance", _retriever(nodes))], top_k=5)
    result = retriever._retrieve(_query())
    assert result[0].node.metadata["index_name"] == "finance"


def test_top_k_limits_across_indexes():
    nodes_a = [_node(str(i), float(i)) for i in range(5)]
    nodes_b = [_node(str(i + 10), float(i)) for i in range(5)]
    retriever = MultiIndexRetriever(
        [("a", _retriever(nodes_a)), ("b", _retriever(nodes_b))],
        top_k=3,
    )
    result = retriever._retrieve(_query())
    assert len(result) == 3


def test_deduplicates_same_node_id_across_indexes():
    nodes_a = [_node("shared", 0.9), _node("a_only", 0.5)]
    nodes_b = [_node("shared", 0.6), _node("b_only", 0.7)]
    retriever = MultiIndexRetriever(
        [("a", _retriever(nodes_a)), ("b", _retriever(nodes_b))],
        top_k=10,
    )
    result = retriever._retrieve(_query())
    ids = [n.node.node_id for n in result]
    assert ids.count("shared") == 1


def test_failing_index_is_skipped_with_warning(capsys):
    bad = MagicMock()
    bad.retrieve.side_effect = RuntimeError("connection failed")
    good_nodes = [_node("a", 0.9)]
    retriever = MultiIndexRetriever(
        [("bad", bad), ("good", _retriever(good_nodes))],
        top_k=5,
    )
    result = retriever._retrieve(_query())
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "bad" in captured.out
    assert any(n.node.node_id == "a" for n in result)


def test_empty_retriever_list_returns_empty():
    retriever = MultiIndexRetriever([], top_k=5)
    result = retriever._retrieve(_query())
    assert result == []
