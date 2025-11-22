"""
BELUX IA - Backend FastAPI
Versão atualizada usando Lean (Emergent AI) direto
Para deploy em Render, Railway, Vercel, etc.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta
from openai import OpenAI
import os
import logging
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# ==================================================
# CONFIGURAÇÃO
# ==================================================

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(title="BELUX IA API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique seu domínio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "belux_ia_db")

# Lean (Emergent AI)
EMERGENT_LLM_KEY = os.getenv("EMERGENT_LLM_KEY")

# Cliente OpenAI apontando para Lean
lean_client = OpenAI(
    api_key=EMERGENT_LLM_KEY,
    base_url="https://api.emergent.ai/v1"
)

# MongoDB Client
db_client = None
db = None

# ==================================================
# EVENTOS DE INICIALIZAÇÃO
# ==================================================

@app.on_event("startup")
async def startup_db_client():
    global db_client, db
    try:
        logger.info("Connecting to MongoDB...")
        db_client = AsyncIOMotorClient(MONGO_URL)
        db = db_client[DB_NAME]
        
        # Teste de conexão
        await db.command('ping')
        logger.info("MongoDB connection initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_db_client():
    if db_client:
        db_client.close()
        logger.info("MongoDB connection closed")


# ==================================================
# MODELOS PYDANTIC
# ==================================================

class RegisterPremiumRequest(BaseModel):
    name: str
    email: EmailStr


class FacialAnalysisRequest(BaseModel):
    user_email: EmailStr
    image_base64: str


class SaveRoutineRequest(BaseModel):
    user_email: EmailStr
    date: str
    notes: str
    mood: str = "neutral"


# ==================================================
# ENDPOINTS
# ==================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "message": "BELUX IA API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.post("/api/register-premium")
async def register_premium(request: RegisterPremiumRequest):
    """
    Registra usuário com acesso premium (30 dias)
    Chamado após pagamento via Kiwify
    """
    try:
        users = db.users
        
        # Verificar se já existe
        existing_user = await users.find_one({"email": request.email})
        
        if existing_user:
            # Atualizar premium
            premium_expiry = datetime.utcnow() + timedelta(days=30)
            await users.update_one(
                {"email": request.email},
                {
                    "$set": {
                        "premium_status": True,
                        "premium_expiry": premium_expiry,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            logger.info(f"Updated premium for existing user: {request.email}")
        else:
            # Criar novo usuário
            premium_expiry = datetime.utcnow() + timedelta(days=30)
            user_data = {
                "name": request.nam
