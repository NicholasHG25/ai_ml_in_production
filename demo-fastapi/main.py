from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Demo Model Serving - Kuliah Tamu AI/ML in Production",
    description="Contoh sederhana mengubah model menjadi layanan API (synchronous) dengan FastAPI dan Pydantic.",
)

POSITIVE = {"bagus", "baik", "mantap", "puas", "suka", "keren", "recommended"}
NEGATIVE = {"buruk", "jelek", "kecewa", "lambat", "mahal", "rusak", "parah"}


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Hotelnya bagus dan pelayanannya mantap"])


class PredictResponse(BaseModel):
    input: str
    sentiment: str
    score: int


def predict_sentiment(text: str) -> tuple[str, int]:
    """Model dummy berbasis kata kunci (pengganti model ML asli)."""
    words = text.lower().split()
    score = sum(w.strip(".,!?") in POSITIVE for w in words) - sum(
        w.strip(".,!?") in NEGATIVE for w in words
    )
    if score > 0:
        return "positif", score
    if score < 0:
        return "negatif", score
    return "netral", score


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    label, score = predict_sentiment(req.text)
    return PredictResponse(input=req.text, sentiment=label, score=score)
