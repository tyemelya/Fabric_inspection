from app.tools.LLMTool import LLMTool, RoutingDecision, parse_json_response
import json
import pytest

class FakeModel:
    device = "cpu"
    def generate(self, **kwargs):
        return [[1, 2, 3, 4, 5]]

def test_parse_json_response():
    response = """
    ```json
    {
        "use_retrieval": true,
        "reason": "User requested similar cases"
    }
    ```
    """

    result = parse_json_response(response)

    assert result["use_retrieval"] is True
    assert result["reason"] == "User requested similar cases"

def test_parse_json_response_missing_json():
    response = "No JSON here"
    with pytest.raises(ValueError, match="No JSON found"):
        parse_json_response(response)
        
def test_plan_workflow_retrieval():
    llm = object.__new__(LLMTool)

    llm.generate = lambda prompt: """
    {
        "use_retrieval": true,
        "reason": "User wants similar examples"
    }
    """
    result = llm.plan_workflow(
        "Find similar fabric cases"
    )

    assert isinstance(result, RoutingDecision)
    assert result.use_retrieval is True
    assert result.reason == "User wants similar examples"


def test_plan_workflow_without_retrieval():
    llm = object.__new__(LLMTool)
    llm.generate = lambda prompt: """
    {
        "use_retrieval": false,
        "reason": "Only classification requested"
    }
    """

    result = llm.plan_workflow(
        "What defect is visible?"
    )

    assert result.use_retrieval is False
    
def test_plan_workflow_invalid_response():
    llm = object.__new__(LLMTool)
    llm.generate = lambda prompt: "I cannot decide"

    with pytest.raises(
        ValueError,
        match="Invalid routing decision"
    ):
        llm.plan_workflow(
            "Find similar cases"
        )