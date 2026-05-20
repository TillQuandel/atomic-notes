from pydantic import BaseModel, field_validator


class AtomicNoteFoss(BaseModel):
    title: str
    concept_type: str
    extracted_body: list[str]
    source_anchors: list[dict]
    related: list[str] = []
    tags: list[str] = []
    source_file: str
    pipeline: str = "foss-atomic"
    hallucination_rate: float = 0.0
    extraction_coverage: float = 0.0
    created: str = ""

    @field_validator("extracted_body")
    @classmethod
    def body_not_empty(cls, v):
        if not v:
            raise ValueError("extracted_body darf nicht leer sein")
        return v
