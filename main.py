import os
import gc
import hmac
import hashlib
import json
import uuid
import requests
import traceback
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Security, Request, Header
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from supabase import create_client, Client

from engine import MLEngine

# --- 1. CONFIGURATION & ENV SETUP ---
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

RAW_SUPABASE_URL = os.getenv("SUPABASE_URL", "https://vrirjtjhmpgydrhqpcib.supabase.co")
SUPABASE_URL = RAW_SUPABASE_URL.split("/rest/v1")[0].rstrip("/")

SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZyaXJqdGpobXBneWRyaHFwY2liIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4OTA2NDY1NiwiZXhwIjoyMTA0NjQwNjU2fQ.QmX84ySRqIdcjwbaKHgaKogkanRWHjfalY-gXDP0z4c"
)

PAYSTACK_SECRET_KEY = os.getenv(
    "PAYSTACK_SECRET_KEY",
    "sk_test_f85c7c33012e50b93ea8ee74f96731d593b2007e"
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI(
    title="No-Code ML Platform API",
    description="Backend API for automated machine learning training, Supabase logging, and billing.",
    version="1.0.0"
)

MODELS_DIR = "saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# --- CORS CONFIGURATION ---
origins = [
    "https://nocode-ai.netlify.app",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# --- 2. SCHEMAS ---
class PaymentInitRequest(BaseModel):
    user_id: str
    email: EmailStr
    payment_type: str

class PredictionRequest(BaseModel):
    input_data: dict


# --- 3. AUTHENTICATION ---
async def verify_api_key(api_key: str = Security(api_key_header)):
    print(f"\n-> [AUTH] Verifying API Key: {repr(api_key)}")
    if not api_key:
        api_key = "my_secret_test_key_123"
    try:
        res = supabase.rpc("get_user_id_by_key", {"p_key": api_key}).execute()
        user_id = res.data
        if user_id:
            print(f"-> [AUTH SUCCESS] Authenticated Real User ID: {user_id}")
            return {"user_id": user_id}
    except Exception as e:
        print(f"-> [WARNING] RPC validation bypassed: {str(e)}")
    
    fallback_user_id = "85cf2871-9c82-4044-8fb1-4ef74ef74ef7" 
    print(f"-> [AUTH FALLBACK] Using development test user_id: {fallback_user_id}")
    return {"user_id": fallback_user_id}


# --- 4. ROUTES ---
@app.get("/")
def read_root():
    return {"status": "online", "message": "No-Code ML Backend is running successfully."}


# --- 5. PAYSTACK BILLING ROUTES ---
@app.post("/pay/initialize")
def initialize_payment(payload: PaymentInitRequest):
    if not PAYSTACK_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Paystack secret key not configured")

    headers = {
        "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json"
    }

    if payload.payment_type == "credit_pack":
        amount_kobo = 5000 * 100
        metadata = {"user_id": payload.user_id, "payment_type": "credit_pack", "credits_to_add": 10}
        data = {
            "email": payload.email,
            "amount": amount_kobo,
            "metadata": metadata
        }
    elif payload.payment_type == "pro_subscription":
        data = {
            "email": payload.email,
            "amount": 15000 * 100,
            "plan": "PLN_tlessu0cswidxs5",  
            "metadata": {"user_id": payload.user_id, "payment_type": "pro_subscription"}
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid payment type")

    response = requests.post("https://api.paystack.co/transaction/initialize", json=data, headers=headers)
    res_data = response.json()

    if not res_data.get("status"):
        raise HTTPException(status_code=400, detail=res_data.get("message", "Paystack initialization failed"))

    return res_data["data"]

@app.post("/webhook/paystack")
async def paystack_webhook(request: Request, x_paystack_signature: str = Header(None)):
    body = await request.body()
    computed_signature = hmac.new(
        PAYSTACK_SECRET_KEY.encode('utf-8'),
        body,
        hashlib.sha512
    ).hexdigest()

    if computed_signature != x_paystack_signature:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_data = json.loads(body)
    event_type = event_data.get("event")
    data = event_data.get("data", {})
    reference = data.get("reference")
    
    if reference:
        tx_check = supabase.table("transactions").select("reference").eq("reference", reference).execute()
        if len(tx_check.data) > 0:
            print(f"-> [WEBHOOK] Transaction {reference} already processed. Ignoring replay attack.")
            return {"status": "success", "message": "already_processed"}

    if event_type == "charge.success":
        metadata = data.get("metadata", {})
        user_id = metadata.get("user_id")

        if user_id:
            payment_type = metadata.get("payment_type")
            profile_res = supabase.table("profiles").select("*").eq("id", user_id).execute()
            profile_exists = len(profile_res.data) > 0
            current_credits = profile_res.data[0].get("credits", 0) if profile_exists else 0

            if payment_type == "credit_pack":
                credits_to_add = int(metadata.get("credits_to_add", 10))
                new_credits = current_credits + credits_to_add
                
                if profile_exists:
                    supabase.table("profiles").update({"credits": new_credits, "plan_type": "free"}).eq("id", user_id).execute()
                else:
                    supabase.table("profiles").insert({"id": user_id, "credits": new_credits, "plan_type": "free"}).execute()

            elif payment_type == "pro_subscription":
                expiry_date = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
                if profile_exists:
                    supabase.table("profiles").update({"plan_type": "pro_subscriber", "subscription_expires_at": expiry_date}).eq("id", user_id).execute()
                else:
                    supabase.table("profiles").insert({"id": user_id, "credits": current_credits, "plan_type": "pro_subscriber", "subscription_expires_at": expiry_date}).execute()

            if reference:
                supabase.table("transactions").insert({"reference": reference, "user_id": user_id, "event_type": event_type}).execute()

    elif event_type in ["invoice.payment_failed", "subscription.disable", "charge.failed"]:
        metadata = data.get("metadata", {})
        user_id = metadata.get("user_id")
        if user_id:
            supabase.table("profiles").update({"plan_type": "free"}).eq("id", user_id).execute()
            if reference:
                supabase.table("transactions").insert({"reference": reference, "user_id": user_id, "event_type": event_type}).execute()

    return {"status": "success"}


# --- 6. CORE ML ROUTE (DELEGATED TO MLEngine) ---
@app.post("/upload-and-train/")
async def upload_and_train(
    file: UploadFile = File(...),
    target_column: str = Query(...),
    authorization: str = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = authorization.split(" ")[1]
    user_response = supabase.auth.get_user(token)
    
    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Invalid session token")

    user_id = user_response.user.id

    profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
    if not profile_response.data:
        raise HTTPException(status_code=400, detail="Profile not found")

    user_profile = profile_response.data[0]
    plan_type = user_profile.get("plan_type", "free")
    expires_at_str = user_profile.get("subscription_expires_at")

    if plan_type == "pro_subscriber":
        is_expired = True
        if expires_at_str:
            try:
                expires_at = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) < expires_at:
                    is_expired = False
            except ValueError:
                pass
                
        if is_expired:
            plan_type = "free"
            supabase.table("profiles").update({"plan_type": "free"}).eq("id", user_id).execute()

    if plan_type != "pro_subscriber":
        deduction_res = supabase.rpc("decrement_credits", {"user_id_param": user_id}).execute()
        if not deduction_res.data:
            raise HTTPException(status_code=402, detail="Insufficient credits. Please purchase a credit pack or upgrade.")

    try:
        MAX_FILE_SIZE = 100 * 1024 * 1024  
        if file.size and file.size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 100MB.")

        filename = file.filename.lower()
        
        # 1. ENFORCE CSV ONLY
        if not filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="For system stability, only .csv files are supported. Please convert your dataset to CSV.")

        # 2. MEMORY-SAFE INGESTION (Chunking & On-the-fly Sampling)
        chunk_list = []
        max_rows_target = 20000 
        current_rows = 0
        
        # Read the file in small 10,000-row chunks so RAM never spikes
        for chunk in pd.read_csv(file.file, chunksize=10000, low_memory=True):
            # Sample down large chunks immediately before storing them
            if len(chunk) > 2000:
                chunk = chunk.sample(frac=0.5, random_state=42)
                
            chunk_list.append(chunk)
            current_rows += len(chunk)
            
            # Cap the maximum rows loaded into the engine to prevent ML training OOM
            if current_rows >= max_rows_target:
                break
                
        df = pd.concat(chunk_list, ignore_index=True)
        
        file.file.close()
        gc.collect() # Force garbage collection immediately

        # 3. MEMORY DOWNCASTING (Halves the final RAM footprint)
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset.")

        model_id = f"mod_{uuid.uuid4().hex[:10]}"
        
        # Instantiate and run MLEngine from engine.py
        engine = MLEngine(df=df, target_column=target_column)
        training_result = engine.train_and_save(model_id=model_id)

        model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")

        try:
            with open(model_path, "rb") as f_model:
                supabase.storage.from_("models").upload(
                    file=f_model,
                    path=f"{model_id}.joblib",
                    file_options={"upsert": "true", "content-type": "application/octet-stream"}
                )
            print(f"-> [STORAGE] Successfully backed up model {model_id} to Supabase Storage.")
        except Exception as storage_err:
            print(f"-> [WARNING] Storage backup failed: {str(storage_err)}")

        return {
            "status": "success",
            "details": training_result
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/predict/{model_id}")
async def predict_model(model_id: str, payload: PredictionRequest):
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    
    if not os.path.exists(model_path):
        try:
            print(f"-> [STORAGE] Model {model_id} missing locally. Downloading from Supabase Storage...")
            file_bytes = supabase.storage.from_("models").download(f"{model_id}.joblib")
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(model_path, "wb") as f_out:
                f_out.write(file_bytes)
        except Exception as e:
            raise HTTPException(status_code=404, detail="Model not found or expired in cloud storage.")

    try:
        artifacts = joblib.load(model_path)
        model = artifacts["model"]
        model_type = artifacts["model_type"]
        features = artifacts["features"]
        label_encoders = artifacts["label_encoders"]
        feature_defaults = artifacts["feature_defaults"]
        target_column = artifacts["target_column"]

        input_df = pd.DataFrame([payload.input_data])

        # Preprocess input matching MLEngine's encoding standards
        for col in features:
            if col not in input_df.columns:
                input_df[col] = feature_defaults.get(col, 0)
            
            if col in label_encoders:
                le = label_encoders[col]
                val = str(input_df[col].iloc[0])
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    # Fallback to default class index if unseen category appears
                    input_df[col] = le.transform([le.classes_[0]])[0]
            else:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(feature_defaults.get(col, 0.0))

        input_df = input_df[features]
        prediction = model.predict(input_df)[0]
        
        # Inverse transform encoded classification target back to string labels if applicable
        if model_type == "classification" and target_column in label_encoders:
            target_le = label_encoders[target_column]
            prediction = target_le.inverse_transform([int(prediction)])[0]
        elif isinstance(prediction, (np.integer, np.floating)):
            prediction = prediction.item()
            
        return {"prediction": prediction}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-model/{model_id}")
async def download_model(model_id: str):
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    
    if not os.path.exists(model_path):
        try:
            print(f"-> [STORAGE] Model {model_id} missing locally for download. Fetching from Supabase...")
            file_bytes = supabase.storage.from_("models").download(f"{model_id}.joblib")
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(model_path, "wb") as f_out:
                f_out.write(file_bytes)
        except Exception as e:
            raise HTTPException(status_code=404, detail="Model file not found in cloud storage.")
            
    return FileResponse(model_path, media_type="application/octet-stream", filename=f"{model_id}.joblib")