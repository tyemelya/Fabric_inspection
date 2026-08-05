from pydantic import BaseModel

class InspectionRequest(BaseModel):
    image_path: str
    user_question: str


class InspectionResponse(BaseModel):
    report: str