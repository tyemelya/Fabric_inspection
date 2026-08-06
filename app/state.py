from typing import TypedDict, NotRequired
from app.tools.vision import VisionResult
from app.tools.evidence import EvidenceResult
from app.tools.similarity import RetrievedCase
from app.tools.LLMTool import RoutingDecision

class InspectionState(TypedDict):
    image_path: str
    user_question: str
    vision_result: NotRequired[VisionResult]
    retrieved_cases: NotRequired[list[RetrievedCase]]
    evidence: NotRequired[EvidenceResult]   
    routing: NotRequired[RoutingDecision]
    report: NotRequired[str]