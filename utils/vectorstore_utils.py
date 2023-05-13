from typing import Optional, Iterable, Any, List
from langchain.vectorstores import VectorStore


class EmptyVectorStore(VectorStore):
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        pass

    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        pass

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Any]:
        pass