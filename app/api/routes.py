from app.api.schemas import InspectionResponse, RetrievedCaseResponse, HealthResponse
from app.inspection import InspectionService
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os
from app.tools.metadata import MetadataTool
import mimetypes
from fastapi.responses import FileResponse

router = APIRouter()
inspection_service = InspectionService()
    
@router.post(
    "/inspect",
    response_model=InspectionResponse,
    summary="Inspect a fabric image for defects",
    description="""
    Analyze a fabric image using an AI-powered textile inspection pipeline.

    The system performs:
    
    1. Visual defect classification using a vision model.
    2. Analysis of the user's question to determine whether additional
       retrieval is required.
    3. Retrieval of visually similar historical fabric cases when requested.
    4. Evidence evaluation comparing retrieved cases with the prediction.
    5. Generation of a natural language inspection report.

    The retrieval stage is activated automatically for questions requiring:
    
    - Similar examples
    - Previous defect cases
    - Comparison with historical samples
    - Explanation or justification of the prediction

    Examples of supported questions:

    - "What defect is visible?"
    - "Find similar examples"
    - "Why was this classified as a broken stitch?"
    - "Compare this with previous cases"

    Returns:
    
    - Final inspection report
    - Predicted defect class
    - Vision model confidence
    - Evidence agreement score (when retrieval is used)
    - Retrieved similar cases with metadata
    """,
    responses={
        200: {
            "description": "Successful fabric inspection",
            "content": {
                "application/json": {
                    "example": {
                        "report": (
                            "The image contains a broken stitch defect. "
                            "The vision model detected this with high confidence. "
                            "Retrieved historical cases show strong agreement."
                        ),
                        "prediction": "broken stitch",
                        "vision_confidence": 0.94,
                        "evidence_score": 0.87,
                        "confidence_level": "high",
                        "retrieved_cases": [
                            {
                                "image_id": 15,
                                "defect_class": "broken stitch",
                                "severity": "medium",
                                "distance": 0.21
                            }
                        ]
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

        vision = result["vision_result"]
        evidence = result.get("evidence")

        return InspectionResponse(
            report=result["report"],
            prediction=vision.prediction,
            vision_confidence=vision.confidence,
            evidence_score=evidence.evidence_score if evidence else None,
            confidence_level=evidence.confidence_level if evidence else None,
            retrieved_cases=[
                RetrievedCaseResponse(
                    image_id=case.id,
                    defect_class=case.metadata.defect_class,
                    severity=case.metadata.severity,
                    distance=case.distance,
                )
                for case in result.get("retrieved_cases", [])
            ],
        )

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

    
@router.get(
    "/images/{image_id}",
    summary="Return image by id",
    description="""
    Retrieve the original fabric image associated with a database image id.

    This endpoint is used to display retrieved similar cases
    in client applications.

    """,
    responses={
        200: {
            "description": "Image found"
        },
        404: {
            "description": "Image id not found"
        }
    }
)
async def get_image(image_id: int):
    metadata_tool = MetadataTool()

    try:
        metadata = metadata_tool.get(image_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Image not found"
        )

    media_type, _ = mimetypes.guess_type(metadata.image_path)

    return FileResponse(
        path=metadata.image_path,
        media_type=media_type or "application/octet-stream"
    )

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
def health():
    return HealthResponse(
        status="ok"
    )