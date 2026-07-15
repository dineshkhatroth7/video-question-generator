from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """
    Represents a portion of the video transcript.

    Attributes:
        start (float): Start time of the transcript segment in seconds.
        end (float): End time of the transcript segment in seconds.
        text (str): Text content spoken during the segment.
    """

    start: float
    end: float
    text: str


class QuestionItem(BaseModel):
    """
    Represents a generated question derived from the watched transcript.

    Attributes:
        question (str): The generated question based on the video content.
        answer_hint (str): A short hint or expected answer to assist evaluation.
        source_segment (str): Transcript segment used to generate the question.
        timestamp_start (float): Starting timestamp of the source segment in seconds.
    """

    question: str
    answer_hint: str
    source_segment: str
    timestamp_start: float


class GenerateQuestionsRequest(BaseModel):
    """
    Request model for generating questions from a user's watched video content.

    Attributes:
        watch_time_seconds (float): Total number of seconds watched by the user.
        num_questions (int): Number of questions to generate from the watched content.
                             Must be between 1 and 10.
    """

    watch_time_seconds: float = Field(
        ...,
        ge=0,
        description="How many seconds of the video the user has watched."
    )

    num_questions: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of questions to generate (1-10)."
    )


class GenerateQuestionsResponse(BaseModel):
    """
    Response model returned after generating questions.

    Attributes:
        video_id (str): Unique identifier of the processed video.
        watch_time_seconds (float): Number of seconds watched by the user.
        watched_transcript (str): Combined transcript text corresponding to the watched duration.
        segments_used (list[TranscriptSegment]): Transcript segments used for question generation.
        questions (list[QuestionItem]): List of generated questions and associated metadata.
    """

    video_id: str
    watch_time_seconds: float
    watched_transcript: str
    segments_used: list[TranscriptSegment]
    questions: list[QuestionItem]