from pydantic import BaseModel

class Question(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    retrieved_document_id: int
    retrieved_document: str
    question: str
    answers: str