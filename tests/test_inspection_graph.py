from app.graph import FabricInspectionGraph
from app.tools.vision import VisionResult
from app.tools.LLMTool import RoutingDecision
from unittest.mock import Mock
import numpy as np
        
def create_graph():
    graph = object.__new__(FabricInspectionGraph)

    graph.llm_tool = Mock()
    graph.vtool = Mock()
    graph.similarity_tool = Mock()
    graph.evidence_tool = Mock()

    graph.graph = graph._build_graph()

    return graph


def test_graph_vision_only():
    inspection = create_graph()

    inspection.vtool.analyze.return_value = VisionResult(
        prediction="broken stitch",
        confidence=0.95,
        embedding=np.array([1,2,3])
    )

    inspection.llm_tool.plan_workflow.return_value = RoutingDecision(
        use_retrieval=False,
        reason="Only classification requested"
    )
    inspection.llm_tool.generate.return_value = (
        "Broken stitch detected."
    )

    result = inspection.run(
        image_path="test.jpg",
        user_question="What defect is visible?"
    )

    assert result["vision_result"].prediction == "broken stitch"
    assert result["report"] == "Broken stitch detected."

    inspection.similarity_tool.search.assert_not_called()
    inspection.evidence_tool.get_evidence.assert_not_called()
        
def test_graph_with_retrieval():
    inspection = create_graph()
    embedding = np.array([1,2,3])
    inspection.vtool.analyze.return_value = VisionResult(
        prediction="broken stitch",
        confidence=0.95,
        embedding=embedding
    )

    inspection.llm_tool.plan_workflow.return_value = RoutingDecision(
        use_retrieval=True,
        reason="User requested similar cases"
    )

    fake_case = Mock()
    fake_case.id = 1
    fake_case.distance = 0.2
    fake_case.metadata.defect_class = "broken stitch"
    fake_case.metadata.severity = "medium"
    fake_case.metadata.description = "Broken threads"

    inspection.similarity_tool.search.return_value = [
        fake_case
    ]
    evidence = Mock()
    evidence.evidence_score = 0.9
    evidence.confidence_level = "high"
    evidence.supporting_cases = 1
    evidence.total_cases = 1

    inspection.evidence_tool.get_evidence.return_value = evidence
    inspection.llm_tool.generate.return_value = (
        "Similar cases support the prediction."
    )

    result = inspection.run(
        image_path="test.jpg",
        user_question="Find similar examples"
    )

    assert len(result["retrieved_cases"]) == 1
    assert result["evidence"].evidence_score == 0.9
    assert result["report"] == (
        "Similar cases support the prediction."
    )
    inspection.vtool.analyze.assert_called_once_with(
        image_path="test.jpg"
    )
    inspection.similarity_tool.search.assert_called_once_with(
        embedding
    )
    inspection.evidence_tool.get_evidence.assert_called_once_with(
        prediction="broken stitch",
        retrieved_cases=[fake_case]
    )