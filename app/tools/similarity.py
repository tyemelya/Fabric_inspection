from typing import List
from annoy import AnnoyIndex
from dataclasses import dataclass
import numpy as np
import json
from app.tools.metadata import Metadata, MetadataTool

@dataclass
class RetrievedCase:
    id: int
    distance: float
    metadata: Metadata
    
class SimilarityTool:
    def __init__(
        self,
        annoy_path: str,
        annoy_index_info: str,
        metadata_tool: MetadataTool
    ):
        with open(annoy_index_info, "r") as f:
            info = json.load(f)

        self.feature_dim = info["feature_dim"]
        self.metric = info["metric"]
        self.metadata_tool = metadata_tool
        self.index = AnnoyIndex(self.feature_dim, metric=self.metric)
        self.index.load(annoy_path)
        
    def search(self, image_embedding: np.ndarray, k=5)-> List[RetrievedCase]:
        """
        Search the ANNOY index for the k most similar fabric images.

        Args:
            image_embedding: Feature embedding produced by the vision model.
            k: Number of nearest neighbors.

        Returns:
            List of similar cases sorted by distance.
        """
    
        if image_embedding.ndim != 1:
            raise ValueError(
                "Expected a single embedding vector"
            )

        if image_embedding.shape[0] != self.feature_dim:    
            raise ValueError(
                f"Expected embedding dimension {self.feature_dim}, got {len(image_embedding)}"
                )
        
        k = min(k, self.index.get_n_items())
            
        indices, dists = self.index.get_nns_by_vector(image_embedding, k, include_distances=True)
        similar_list = []
    
        for idx, dist in zip(indices, dists):
            metadata=self.metadata_tool.get(idx)
            if metadata is None:
                raise ValueError(
                f"Missing metadata for image id {idx}"
            )
                
            similar_list.append(RetrievedCase(
                id=idx,
                distance=dist,
                metadata=metadata
                )
            )
        return similar_list    