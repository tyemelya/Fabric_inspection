from app.api.schemas import InspectionResponse
from app.inspection import InspectionService
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, UploadFile, File, Form
import os

router = APIRouter(
    prefix="/inspection",
    tags=["Fabric Inspection"]
)
inspection_service = InspectionService()
    
@router.post(
    "/inspect",
    response_model=InspectionResponse,
    summary="Inspect a fabric image",
    description="""
    Analyze a fabric image for defects using the AI inspection pipeline.

    The workflow:
    
    1. Vision model classifies the defect.
    2. User question is analyzed to determine whether retrieval is needed.
    3. Similar historical cases are retrieved if requested.
    4. Evidence is evaluated from retrieved cases.
    5. LLM generates the final inspection report.

    Supported questions:
    - "What defect is visible?"
    - "Show similar examples"
    - "Why is this classified as a broken stitch?"
    - "Compare with previous cases"
    """,
    responses={
        200: {
            "description": "Successful fabric inspection",
            "content": {
                "application/json": {
                    "example": {
                        "report": (
                            "The image contains a broken stitch defect. "
                            "The prediction confidence is 94%. "
                            "Similar historical cases support this classification."
                        )
                    }
                }
            }
        },
        400: {
            "description": "Invalid image or request"
        }
    }
)
async def inspect(
    uploaded_image: UploadFile = File(
        ...,
        description="Fabric image to inspect (JPEG/PNG)"
    ),
    question: str = Form(
        ...,
        description=(
            "Question about the inspection. "
            "Examples: 'find similar cases', "
            "'explain the decision'"
        ),
        examples=[
            "What defect is present?",
            "Find similar previous cases"
        ]
    )
):
    suffix = Path(uploaded_image.filename).suffix

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await uploaded_image.read()
        tmp.write(contents)
        image_path = tmp.name

    try:
        result = inspection_service.inspect(
            image_path=image_path,
            user_question=question,
        )

        return InspectionResponse(
            report=result["report"]
        )

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)