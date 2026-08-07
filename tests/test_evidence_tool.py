from app.tools.similarity import RetrievedCase
from app.tools.evidence import EvidenceTool
from app.tools.metadata import Metadata

def test_evidence_no_cases():
    evidence_tool = EvidenceTool()
    result = evidence_tool.get_evidence(
        prediction="broken stitch",
        retrieved_cases=[]
    )
    
    assert result.evidence_score == 0.0
    assert result.has_disagreement is True
    assert result.confidence_level == "no evidence"
    assert result.supporting_cases == 0
    assert result.total_cases == 0

def create_case(image_id, defect_class, distance=0.1):
    return RetrievedCase(
        id=image_id,
        metadata=Metadata(
            image_path="data/sample.jpg",
            defect_class=defect_class,
            severity="medium",
            description="Test case",
        ),
        distance=distance,
    )
        
def test_evidence_all_cases_support_prediction():
    evidence_tool = EvidenceTool()
    
    cases = [
        create_case(1, "broken stitch", 0.1),
        create_case(2, "broken stitch", 0.2),
        create_case(3, "broken stitch", 0.3),
    ]

    result = evidence_tool.get_evidence(
        prediction="broken stitch",
        retrieved_cases=cases
    )

    assert result.evidence_score == 1.0
    assert result.has_disagreement is False
    assert result.confidence_level == "high"
    assert result.supporting_cases == 3
    assert result.total_cases == 3
    
def test_evidence_disagreement():
    tool = EvidenceTool()

    cases = [
        create_case(1, "broken stitch", 0.1),
        create_case(2, "hole", 0.2),
        create_case(3, "hole", 0.3),
    ]

    result = tool.get_evidence(
        prediction="broken stitch",
        retrieved_cases=cases
    )

    assert result.has_disagreement is True
    assert result.confidence_level == "low"
    assert result.supporting_cases == 1
    assert result.total_cases == 3