# app/models/schemas.py
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class Verdict(str, Enum):
    DEEPFAKE = "DEEPFAKE"
    AUTHENTIC = "AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"

class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class AnalysisDimension(BaseModel):
    score: int
    findings: List[str]
    details: str

class ManipulationRegion(BaseModel):
    region: str
    severity: str
    description: str

class AnalysisResult(BaseModel):
    verdict: Verdict
    confidence: int
    risk_level: RiskLevel
    overall_score: int
    analysis: dict
    manipulation_regions: List[ManipulationRegion]
    technical_indicators: List[str]
    generation_method: str
    forensic_summary: str
    recommendations: List[str]
    xai_highlights: List[str]

class DetectionResponse(BaseModel):
    success: bool
    filename: str
    media_type: str
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None
