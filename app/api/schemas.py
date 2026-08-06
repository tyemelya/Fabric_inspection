from pydantic import BaseModel
from app.tools.similarity import RetrievedCase
from dataclasses import dataclass

@dataclass
class RetrievedCaseResponse:
    image_id: int
    defect_class: str
    severity: str
    distance: float

class InspectionRequest(BaseModel):
    image_path: str
    user_question: str

class InspectionResponse(BaseModel):
    report: str
    prediction: str
    vision_confidence: float
    evidence_score: float | None
    confidence_level: str | None
    retrieved_cases: list[RetrievedCaseResponse]
    
class HealthResponse(BaseModel):
    status: str
