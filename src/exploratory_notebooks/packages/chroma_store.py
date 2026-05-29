"""ChromaDB persistence layer for BirdNET embeddings."""

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings


def get_client(chroma_db_path: str) -> chromadb.PersistentClient:
    os.makedirs(chroma_db_path, exist_ok=True)
    return chromadb.PersistentClient(
        path=chroma_db_path,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_or_create_collection(
    client: chromadb.PersistentClient,
    name: str = "anuraset_singlecall",
) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=name,
        metadata={
            "description": "BirdNET embeddings",
            "embedding_model": "BirdNET-Analyzer-V2.4",
            "embedding_dim": 1024,
            "created_at": datetime.now().isoformat(),
        },
    )


def add_embeddings(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    collection: chromadb.Collection,
    batch_size: int = 1000,
    id_offset: int = 0,
):
    """Upsert embeddings with metadata into the collection in batches."""
    n = len(embeddings)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch_emb = embeddings[i:end]
        batch_meta = metadata_df.iloc[i:end]

        ids = [f"emb_{id_offset + i + j:06d}" for j in range(len(batch_emb))]
        metadatas = []
        for _, row in batch_meta.iterrows():
            meta: dict = {
                "fname": str(row["fname"]),
                "species": str(row["species"]),
                "site": str(row["site"]),
            }
            if "date" in row:
                meta["date"] = str(row["date"])
            if "min_t" in row:
                meta["min_t"] = float(row["min_t"])
            if "max_t" in row:
                meta["max_t"] = float(row["max_t"])
            metadatas.append(meta)

        collection.add(ids=ids, embeddings=batch_emb.tolist(), metadatas=metadatas)


def query_similar(
    query_embedding: np.ndarray,
    collection: chromadb.Collection,
    n_results: int = 10,
    filter_species: Optional[str] = None,
    filter_site: Optional[str] = None,
) -> dict:
    """Return the n_results nearest neighbours with distances and metadata."""
    where: dict = {}
    if filter_species:
        where["species"] = filter_species
    if filter_site:
        where["site"] = filter_site
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where=where if where else None,
    )


def get_by_species(
    collection: chromadb.Collection,
    species: str,
    limit: Optional[int] = None,
) -> dict:
    return collection.get(where={"species": species}, limit=limit)


def export_to_numpy(
    collection: chromadb.Collection,
    output_dir: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Dump the full collection to .npy + .csv files. Returns (embeddings, metadata_df)."""
    results = collection.get(include=["embeddings", "metadatas"])
    embeddings = np.array(results["embeddings"])
    metadata_df = pd.DataFrame(results["metadatas"])

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    np.save(os.path.join(output_dir, f"{ts}_embeddings.npy"), embeddings)
    metadata_df.to_csv(os.path.join(output_dir, f"{ts}_metadata.csv"), index=False)

    return embeddings, metadata_df
