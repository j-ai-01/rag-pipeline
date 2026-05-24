from typing import List, Tuple
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from utils.hybrid_retriever import HybridRetriever


class MultiIndexRetriever(BaseRetriever):
    def __init__(
        self,
        retrievers: List[Tuple[str, BaseRetriever]],
        top_k: int = 5,
    ):
        self._retrievers = retrievers
        self._top_k = top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        all_nodes: List[NodeWithScore] = []

        for index_name, retriever in self._retrievers:
            try:
                nodes = retriever.retrieve(query_bundle)
                nodes = HybridRetriever._normalize(nodes)
                for n in nodes:
                    n.node.metadata = {**n.node.metadata, "index_name": index_name}
                all_nodes.extend(nodes)
            except Exception as e:
                print(f"Warning: Could not retrieve from index '{index_name}': {e}. Skipping.")

        scores: dict = {}
        node_map: dict = {}
        for n in all_nodes:
            nid = n.node.node_id
            score = n.score or 0.0
            if nid not in scores or score > scores[nid]:
                scores[nid] = score
                node_map[nid] = n

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [node_map[nid] for nid, _ in ranked[: self._top_k]]
