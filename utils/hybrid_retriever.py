from typing import List, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: Optional[BaseRetriever],
        alpha: float = 0.5,
        top_k: int = 5,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._alpha = alpha
        self._top_k = top_k
        super().__init__()

    @staticmethod
    def _normalize(nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        if not nodes:
            return nodes
        scores = [n.score or 0.0 for n in nodes]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [NodeWithScore(node=n.node, score=1.0) for n in nodes]
        return [
            NodeWithScore(node=n.node, score=((n.score or 0.0) - min_s) / (max_s - min_s))
            for n in nodes
        ]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_nodes = self._normalize(self._vector_retriever.retrieve(query_bundle))

        if self._bm25_retriever is None:
            return vector_nodes[: self._top_k]

        bm25_nodes = self._normalize(self._bm25_retriever.retrieve(query_bundle))

        scores: dict = {}
        node_map: dict = {}

        for n in vector_nodes:
            nid = n.node.node_id
            scores[nid] = self._alpha * (n.score or 0.0)
            node_map[nid] = n

        for n in bm25_nodes:
            nid = n.node.node_id
            contribution = (1 - self._alpha) * (n.score or 0.0)
            if nid in scores:
                scores[nid] += contribution
            else:
                scores[nid] = contribution
                node_map[nid] = n

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for nid, score in ranked[: self._top_k]:
            node = node_map[nid]
            node.score = score
            result.append(node)
        return result
