"""FAISS vector similarity search index management for face embeddings.

This module provides a thread-safe wrapper around a FAISS IndexFlatIP (inner
product) index used to perform nearest-neighbour searches over face embedding
vectors.  Because all vectors are L2-normalised before insertion, inner product
scores are equivalent to cosine similarity, producing values in [-1, 1].

The index and a JSON metadata sidecar file are persisted to disk so that the
system can be restarted without re-computing embeddings from the database.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import faiss
import numpy as np

from app.core.config import Settings


@dataclass(slots=True)
# Lightweight container for a single face embedding with its associated person metadata, used for managing the FAISS index and its contents in memory.
class IndexedEmbedding:
    

    embedding_id: str
    person_id: str
    person_name: str
    person_type: str | None
    vector: np.ndarray

# The main service class for managing the FAISS index, including loading/saving from disk, rebuilding from database records, and performing similarity searches.
class FaissService:
    

    def __init__(self, settings: Settings) -> None:
        
        self.settings = settings
        self.logger = logging.getLogger("sentinel.faiss")
        # Face embeddings from AdaFace are 512-dimensional
        self.dimension = 512
        self._index: faiss.Index | None = None
        self._items: list[IndexedEmbedding] = []
        self._last_updated: datetime | None = None
        # Lock serialises concurrent search/rebuild operations
        self._lock = Lock()
        self.index_path = Path(settings.faiss_index_path)
        self.metadata_path = Path(settings.faiss_metadata_path)

# Initialise an empty FAISS index and ensure the data directory exists.
    def start(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is None:
            # IndexFlatIP uses inner product; with normalised vectors this is cosine similarity
            self._index = faiss.IndexFlatIP(self.dimension)

# Load the persisted index from disk if it exists, otherwise start with an empty index.
    def load_from_disk(self) -> bool:
        
        self.start()
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        try:
            # Read the JSON metadata sidecar that maps index positions to person info
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            # Replace the in-memory index with the one persisted on disk
            self._index = faiss.read_index(str(self.index_path))
            items_payload = payload.get("items", [])
            self._items = []
            for idx, item in enumerate(items_payload):
                # Reconstruct the vector from FAISS so we have it available in memory;
                # fall back to a zero vector if the index has fewer entries than metadata
                vector = self._index.reconstruct(idx) if idx < self._index.ntotal else np.zeros((self.dimension,), dtype=np.float32)
                self._items.append(
                    IndexedEmbedding(
                        embedding_id=str(item["embedding_id"]),
                        person_id=str(item["person_id"]),
                        person_name=str(item["person_name"]),
                        person_type=str(item["person_type"]) if item.get("person_type") else None,
                        vector=np.asarray(vector, dtype=np.float32),
                    )
                )
            # Restore the last-updated timestamp from the sidecar
            last_updated = payload.get("last_updated")
            if isinstance(last_updated, str) and last_updated:
                try:
                    self._last_updated = datetime.fromisoformat(last_updated)
                except ValueError:
                    self._last_updated = datetime.now(timezone.utc)
            else:
                self._last_updated = datetime.now(timezone.utc)
            # Sanity check: the number of vectors in the FAISS index must match metadata
            if self._index.ntotal != len(self._items):
                raise ValueError("Persisted FAISS index metadata count does not match index contents")
            self.logger.info("Loaded FAISS index from %s", self.index_path)
            return True
        except Exception:
            self.logger.exception("Failed to load persisted FAISS index")
            return False

    @staticmethod
    # Helper method to L2-normalise an embedding vector so that inner product equals cosine similarity, with safety check for zero-norm vectors.
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm <= 0:
            return vec
        return vec / norm

# Rebuild the FAISS index from a complete list of embeddings, replacing all existing contents and persisting to disk.
    def rebuild(self, entries: list[IndexedEmbedding]) -> None:
        
        self.start()
        with self._lock:
            # Validate that no zero-norm vectors are present before rebuilding
            for entry in entries:
                if np.linalg.norm(entry.vector.astype(np.float32)) < 1e-6:
                    raise ValueError("Cannot rebuild index with zero-norm vectors; reload from database")
            self._items = entries
            # Create a brand-new flat inner-product index
            self._index = faiss.IndexFlatIP(self.dimension)
            if entries:
                # Stack all normalised vectors into a single matrix and add in bulk
                matrix = np.vstack([self._normalize(entry.vector) for entry in entries]).astype(np.float32)
                self._index.add(matrix)
            self._last_updated = datetime.now(timezone.utc)
            self._persist()
            self.logger.info("FAISS index rebuilt with %s vectors", len(entries))

# Write the FAISS binary index and JSON metadata sidecar to disk.
    def _persist(self) -> None:
        if self._index is None:
            return
        # Write the binary FAISS index file
        faiss.write_index(self._index, str(self.index_path))
        # Build the JSON sidecar payload with per-item metadata
        payload = {
            "last_updated": self._last_updated.isoformat() if self._last_updated else None,
            "items": [
                {
                    "embedding_id": item.embedding_id,
                    "person_id": item.person_id,
                    "person_name": item.person_name,
                    "person_type": item.person_type,
                }
                for item in self._items
            ],
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

# Find the top-k nearest neighbours for a given query embedding.
    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        results = self.search_batch([query_vector], top_k)
        return results[0] if results else []

# Find the top-k nearest neighbours for several query embeddings at once.
    def search_batch(self, query_vectors: list[np.ndarray], top_k: int, *, distinct_people: bool = False) -> list[list[dict]]:
        """Search many embeddings in one FAISS call.

        The live pipeline recognises every visible face in the same cycle, so
        issuing one query per face means one lock acquisition and one BLAS call
        each. Batching collapses those into a single pass.
        """
        if not query_vectors:
            return []

        self.start()
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return [[] for _ in query_vectors]

            # (n, dim) query matrix for a single batched FAISS search
            queries = np.vstack([self._normalize(vector) for vector in query_vectors]).astype(np.float32)
            count = min(max(top_k, 1), self._index.ntotal)
            while True:
                distances, labels = self._index.search(queries, count)
                if not distinct_people or count >= self._index.ntotal:
                    break
                if all(len({self._items[int(label)].person_id for label in row if label >= 0}) >= top_k for row in labels):
                    break
                count = min(count * 2, self._index.ntotal)

            batch_results: list[list[dict]] = []
            for row_distances, row_labels in zip(distances, labels, strict=False):
                results = []
                for distance, label in zip(row_distances, row_labels, strict=False):
                    # FAISS returns -1 for unfilled slots when top_k > ntotal
                    if label < 0 or label >= len(self._items):
                        continue
                    item = self._items[int(label)]
                    results.append(
                        {
                            "embedding_id": item.embedding_id,
                            "person_id": item.person_id,
                            "person_name": item.person_name,
                            "person_type": item.person_type,
                            # Clamp to valid cosine similarity range
                            "similarity": min(max(float(distance), -1.0), 1.0),
                        }
                    )
                if distinct_people:
                    people = {}
                    for result in results:
                        people.setdefault(result['person_id'], result)
                    results = list(people.values())[:top_k]
                batch_results.append(results)
            return batch_results

# Check whether every indexed item has a non-None ``person_type``, used to detect if the metadata sidecar is outdated and requires rebuilding from the database.
    def has_person_type_metadata(self) -> bool:
        
        with self._lock:
            return all(item.person_type is not None for item in self._items)

# Return summary statistics about the current FAISS index.
    def stats(self, total_persons: int) -> dict:
        
        index_size_bytes = self.index_path.stat().st_size if self.index_path.exists() else 0
        total_vectors = len(self._items)
        return {
            "total_vectors": total_vectors,
            "total_persons": total_persons,
            "embedding_dim": self.dimension,
            "index_size_mb": round(index_size_bytes / (1024 * 1024), 3),
            "last_updated": self._last_updated,
        }
