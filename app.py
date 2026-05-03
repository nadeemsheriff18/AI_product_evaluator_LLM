import os
import logging
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# -------------------- ENV --------------------
load_dotenv()

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

# -------------------- APP --------------------
app = FastAPI()

DEFAULT_MODEL = "llama-3.1-8b-instant"

# -------------------- REQUEST DTO --------------------
class IdeaRequest(BaseModel):
    businessIdea: str
    industry: str | None = None
    targetAudience: str | None = None
    region: str | None = None
    budgetRange: str | None = None
    problemStatement: str | None = None

# -------------------- RESPONSE DTO --------------------
class MarketResearchResponse(BaseModel):
    summary: dict | str
    competitors: list[str]
    marketSize: dict | str
    opportunities: list[str]
    risks: list[str]
    swotAnalysis: dict | str

# -------------------- PROMPTS --------------------
SYSTEM_PROMPT = """You are a market research analyst for early-stage product ideas.
Return only valid JSON that matches the requested schema. Be practical, specific,
and honest about uncertainty. Do not include markdown, citations, or extra keys."""

def build_user_prompt(data: IdeaRequest):
    return f"""Analyze this product/business idea and return JSON with exactly these keys:
summary, competitors, marketSize, opportunities, risks, swotAnalysis.

Business idea: {data.businessIdea}
Industry: {data.industry}
Target audience: {data.targetAudience}
Region: {data.region}
Budget range: {data.budgetRange}
Problem statement: {data.problemStatement}

Keep arrays to 3-5 items and keep each field concise."""

# -------------------- LLM --------------------
def get_llm():
    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    temperature = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

    logger.info(f"Initializing LLM | model={model} | temperature={temperature}")

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )

# -------------------- NORMALIZER --------------------
def normalize(result):
    if isinstance(result.summary, dict):
        result.summary = str(result.summary)

    if isinstance(result.marketSize, dict):
        result.marketSize = str(result.marketSize)

    if isinstance(result.swotAnalysis, dict):
        result.swotAnalysis = str(result.swotAnalysis)

    return result

# -------------------- ROUTES --------------------
@app.get("/health")
async def health():
    logger.info("Health check called")
    return {"status": "LLM service running"}

@app.post("/analyze", response_model=MarketResearchResponse)
async def analyze(request: IdeaRequest):
    start_time = time.time()

    logger.info(f"/analyze called | input={request.dict()}")

    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY not configured")
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        llm = get_llm()

        structured_llm = llm.with_structured_output(
            MarketResearchResponse,
            method="json_mode",
        )

        user_prompt = build_user_prompt(request)

        logger.info("Sending request to LLM...")

        # ⚠️ LangChain is sync → run in threadpool
        result = structured_llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )

        result = normalize(result)

        duration = round(time.time() - start_time, 2)
        logger.info(f"LLM response received in {duration}s")

        return result

    except Exception as exc:
        logger.error("LLM request failed", exc_info=True)
        raise HTTPException(status_code=502, detail=str(exc))