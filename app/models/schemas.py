from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class QuestionItem(BaseModel):
    question: str
    answer_hint: str
    source_segment: str
    timestamp_start: float


class GenerateQuestionsRequest(BaseModel):
    watch_time_seconds: float = Field(..., ge=0, description="How many seconds watched today")
    num_questions: int = Field(default=5, ge=1, le=10)


class GenerateQuestionsResponse(BaseModel):
    video_id: str
    watch_time_seconds: float
    watched_transcript: str
    segments_used: list[TranscriptSegment]
    questions: list[QuestionItem]
