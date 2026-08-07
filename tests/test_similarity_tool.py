from annoy import AnnoyIndex
import json
import numpy as np
from app.tools.similarity import SimilarityTool
from app.tools.metadata import Metadata
import pytest

def create_annoy_index(tmp_path):
    feature_dim = 3

    index = AnnoyIndex(feature_dim, "angular")

    index.add_item(0, np.array([1.0, 0.0, 0.0]))
    index.add_item(1, np.array([0.9, 0.1, 0.0]))
    index.add_item(2, np.array([0.0, 1.0, 0.0]))

    index.build(10)

    annoy_path = tmp_path / "test.ann"
    index.save(str(annoy_path))

    info_path = tmp_path / "index_info.json"

    with open(info_path, "w") as f:
        json.dump(
            {
                "feature_dim": feature_dim,
                "metric": "angular",
            },
            f,
        )

    return annoy_path, info_path

class FakeMetadataTool:

    def get(self, idx):
        return Metadata(
            id={idx},
            image_path=f"images/{idx}.jpg",
            defect_class="broken stitch",
            severity="medium",
            description="Test defect",
        )
        
def test_similarity_search(tmp_path):
    annoy_path, info_path = create_annoy_index(tmp_path)

    metadata_tool = FakeMetadataTool()

    similarity_tool = SimilarityTool(
        annoy_path=str(annoy_path),
        annoy_index_info=str(info_path),
        metadata_tool=metadata_tool,
    )

    embedding = np.array([1.0, 0.0, 0.0])

    result = similarity_tool.search(embedding, k=2)

    assert len(result) == 2

    assert result[0].id == 0
    assert result[1].id == 1
    assert result[0].distance <= result[1].distance
    
def test_similarity_wrong_dimension(tmp_path):
    annoy_path, info_path = create_annoy_index(tmp_path)
    
    metadata_tool = FakeMetadataTool()
    
    similarity_tool = SimilarityTool(
        annoy_path=str(annoy_path),
        annoy_index_info=str(info_path),
        metadata_tool=metadata_tool,
    )
    
    embedding = np.array([1.0, 0.0, 0.0, 0.0])
    
    with pytest.raises(ValueError, match="Expected embedding dimension"):
        similarity_tool.search(embedding)

def test_similarity_requires_1d_embedding(tmp_path):
    annoy_path, info_path = create_annoy_index(tmp_path)
        
    metadata_tool = FakeMetadataTool()
        
    similarity_tool = SimilarityTool(
        annoy_path=str(annoy_path),
        annoy_index_info=str(info_path),
        metadata_tool=metadata_tool,
    )
        
    embedding = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Expected a single embedding vector"):
        similarity_tool.search(embedding)        
        
def test_similarity_limits_k(tmp_path):
    annoy_path, info_path = create_annoy_index(tmp_path)
        
    metadata_tool = FakeMetadataTool()
    similarity_tool = SimilarityTool(
        annoy_path=str(annoy_path),
        annoy_index_info=str(info_path),
        metadata_tool=metadata_tool,
    )
        
    embedding = np.array([1.0, 0.0, 0.0])
    result = similarity_tool.search(embedding, 15)
    assert len(result) == 3
    
class MissingMetadataTool:
    def get(self, idx):
        if idx == 1:
            return None

        return Metadata(
            id=idx,
            image_path=f"images/{idx}.jpg",
            defect_class="broken stitch",
            severity="medium",
            description="Test defect",
        )

def test_similarity_missing_metadata(tmp_path):
    annoy_path, info_path = create_annoy_index(tmp_path)

    similarity_tool = SimilarityTool(
        annoy_path=str(annoy_path),
        annoy_index_info=str(info_path),
        metadata_tool=MissingMetadataTool(),
    )

    embedding = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Missing metadata"):
        similarity_tool.search(embedding, k=3)