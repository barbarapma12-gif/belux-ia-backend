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
                "name": request.name,
                "email": request.email,
                "premium_status": True,
                "premium_expiry": premium_expiry,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            await users.insert_one(user_data)
            logger.info(f"Created new premium user: {request.email}")
        
        # Retornar dados do usuário
        user = await users.find_one({"email": request.email}, {"_id": 0})
        
        return {
            "message": "User registered with premium access",
            "user": {
                "name": user["name"],
                "email": user["email"],
                "premium_status": user["premium_status"],
                "premium_expiry": user["premium_expiry"].isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error registering premium user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/{email}")
async def get_user(email: str):
    """Obtém dados do usuário"""
    try:
        users = db.users
        user = await users.find_one({"email": email}, {"_id": 0})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-face")
async def analyze_face(request: FacialAnalysisRequest):
    """
    Análise facial com IA usando Lean GPT Vision
    """
    try:
        # Verificar se usuário tem premium
        users = db.users
        user = await users.find_one({"email": request.user_email})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.get("premium_status"):
            raise HTTPException(status_code=403, detail="Premium access required")
        
        # Verificar se premium expirou
        if user.get("premium_expiry") and user["premium_expiry"] < datetime.utcnow():
            raise HTTPException(status_code=403, detail="Premium access expired")
        
        # Garantir formato correto da imagem
        image_base64 = request.image_base64
        if not image_base64.startswith('data:image'):
            image_base64 = f"data:image/jpeg;base64,{image_base64}"
        
        # Prompt para análise de pele
        prompt = """
        Você é um especialista em dermatologia e análise de pele.
        
        Analise esta foto facial e forneça:
        
        1. **Tipo de Pele**: oleosa, seca, mista ou normal
        2. **Condições Observadas**: acne, manchas, linhas de expressão, oleosidade, ressecamento
        3. **Nível de Hidratação**: baixo, médio ou alto
        4. **Textura da Pele**: lisa, irregular, áspera
        5. **Recomendações Personalizadas**:
           - Rotina matinal (3 passos principais)
           - Rotina noturna (3 passos principais)
           - Produtos específicos recomendados
           - Ingredientes ativos ideais (ex: ácido hialurônico, niacinamida, etc.)
        
        Seja específico e prático nas recomendações.
        Use linguagem acessível e amigável.
        """
        
        # Chamada para Lean GPT Vision
        logger.info(f"Analyzing face for user: {request.user_email}")
        
        response = lean_client.chat.completions.create(
            model="lean-gpt-vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_base64
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        analysis_result = response.choices[0].message.content
        
        # Salvar análise no MongoDB
        facial_analyses = db.facial_analyses
        analysis_doc = {
            "user_email": request.user_email,
            "image_base64": request.image_base64,  # Salva imagem para histórico
            "analysis_result": analysis_result,
            "model_used": "lean-gpt-vision",
            "created_at": datetime.utcnow()
        }
        await facial_analyses.insert_one(analysis_doc)
        
        logger.info(f"Analysis saved for user: {request.user_email}")
        
        return {
            "message": "Analysis completed successfully",
            "analysis": analysis_result,
            "created_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save-routine")
async def save_routine(request: SaveRoutineRequest):
    """Salva entrada do diário de rotina"""
    try:
        routine_diary = db.routine_diary
        
        # Verificar se já existe entrada para esta data
        existing = await routine_diary.find_one({
            "user_email": request.user_email,
            "date": request.date
        })
        
        if existing:
            # Atualizar existente
            await routine_diary.update_one(
                {"user_email": request.user_email, "date": request.date},
                {
                    "$set": {
                        "notes": request.notes,
                        "mood": request.mood,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            message = "Routine updated successfully"
        else:
            # Criar nova entrada
            routine_doc = {
                "user_email": request.user_email,
                "date": request.date,
                "notes": request.notes,
                "mood": request.mood,
                "created_at": datetime.utcnow()
            }
            await routine_diary.insert_one(routine_doc)
            message = "Routine saved successfully"
        
        logger.info(f"Routine saved for user: {request.user_email}, date: {request.date}")
        
        return {"message": message}
        
    except Exception as e:
        logger.error(f"Error saving routine: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/routine/{email}")
async def get_routine(email: str):
    """Obtém entradas do diário de rotina"""
    try:
        routine_diary = db.routine_diary
        
        # Buscar todas as entradas do usuário, ordenadas por data (mais recente primeiro)
        cursor = routine_diary.find(
            {"user_email": email},
            {"_id": 0}
        ).sort("date", -1)
        
        entries = await cursor.to_list(length=100)
        
        return {
            "user_email": email,
            "entries": entries,
            "count": len(entries)
        }
        
    except Exception as e:
        logger.error(f"Error fetching routine: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyses/{email}")
async def get_analyses(email: str):
    """Obtém histórico de análises faciais"""
    try:
        facial_analyses = db.facial_analyses
        
        # Buscar análises (sem as imagens para reduzir payload)
        cursor = facial_analyses.find(
            {"user_email": email},
            {"_id": 0, "image_base64": 0}  # Exclui imagem
        ).sort("created_at", -1)
        
        analyses = await cursor.to_list(length=50)
        
        return {
            "user_email": email,
            "analyses": analyses,
            "count": len(analyses)
        }
        
    except Exception as e:
        logger.error(f"Error fetching analyses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================================================
# EXECUTAR SERVIDOR
# ==================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(
        "server_com_lean:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
