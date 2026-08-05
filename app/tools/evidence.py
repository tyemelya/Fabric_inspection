from dataclasses import dataclass
HIGH_CONFIDENCE_THRESHOLD = 0.8
AGREEMENT_THRESHOLD = 0.5

@dataclass
class EvidenceResult:
    evidence_score: float
    has_disagreement: bool
    confidence_level: str
    supporting_cases: int
    total_cases: int
    retrieved_classes: list[str]

class EvidenceTool:
    @staticmethod
    def get_evidence(prediction, retrieved_cases):
        if not retrieved_cases:
            return EvidenceResult(
                evidence_score=0.0,
                has_disagreement=True,
                confidence_level="no evidence",
                supporting_cases=0,
                total_cases=0,
                retrieved_classes = []
            )
        retrieved_classes=[case.metadata.defect_class for case in retrieved_cases]
        
        total_weight = 0.0
        matching_weight = 0.0
        supporting_cases = 0 

        for case in retrieved_cases:
            weight = 1 / (case.distance + 1e-5)
            total_weight += weight
            if case.metadata.defect_class == prediction:
                matching_weight += weight
                supporting_cases += 1

        if total_weight == 0:
            return EvidenceResult(
                evidence_score=0.0,
                has_disagreement=True,
                confidence_level="no evidence",
                supporting_cases=0,
                total_cases=len(retrieved_cases),
                retrieved_classes = retrieved_classes
            )
        
        evidence_score = matching_weight / total_weight
        
        if supporting_cases==len(retrieved_cases):
            has_disagreement = False
        else:
            has_disagreement = True

        confidence_level = ""
        
        if evidence_score < AGREEMENT_THRESHOLD:
            confidence_level = "low"
        elif evidence_score < HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = "medium"
        else:
            confidence_level = "high"

        return EvidenceResult(
            evidence_score = evidence_score,
            has_disagreement = has_disagreement,
            confidence_level = confidence_level,
            supporting_cases=supporting_cases,
            total_cases=len(retrieved_cases),
            retrieved_classes = retrieved_classes
        )