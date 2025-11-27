# main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router as chat_router

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# --- !! NAME CHANGE IS HERE !! ---
app = FastAPI(
    title="Luma Chatbot API",
    description="Provides RAG + NLU + LLM conversation for emotional support.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("--- Application logging configured and started ---")
# ---------------------------------


# --- CORS (Cross-Origin Resource Sharing) ---
origins = [
    "http://localhost:3000", # Your Next.js dev server
    "https://luma-479300.uc.r.appspot.com", # Your production domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------------

# Include the chat router
app.include_router(chat_router, prefix="/api/v1")

@app.get("/", tags=["Health Check"])
async def root():
    """
    Root endpoint for health checks.
    """
    return {"status": "ok", "message": "Luma API is running"}