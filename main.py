import os
import shutil
import hmac
import hashlib
import json
import requests
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query, Depends, HTTPException, Security, Request, Header
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from supabase import create_client, Client

# --- 1. CONFIGURATION & ENV SETUP ---
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Cleaned base URL (stripped /rest/v1/ so Supabase SDK routes correctly)
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

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI(
    title="No-Code ML Platform API",
    description="Backend API for automated machine learning training, Supabase logging, and billing.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# --- 2. SCHEMAS ---
class PaymentInitRequest(BaseModel):
    user_id: str
    email: EmailStr
    payment_type: str  # "credit_pack" or "pro_subscription"


# --- 3. AUTHENTICATION ---
async def verify_api_key(api_key: str = Security(api_key_header)):
    """Validates API key with a fallback test user ID."""
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
    """Initializes a Paystack transaction for credit purchase or subscription."""
    if not PAYSTACK_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Paystack secret key not configured")

    headers = {
        "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json"
    }

    if payload.payment_type == "credit_pack":
        amount_kobo = 5000 * 100  # ₦5,000 for 50 credits
        metadata = {"user_id": payload.user_id, "payment_type": "credit_pack", "credits_to_add": 50}
        data = {
            "email": payload.email,
            "amount": amount_kobo,
            "metadata": metadata
        }
    elif payload.payment_type == "pro_subscription":
        data = {
            "email": payload.email,
            "amount": 15000 * 100,  # ₦15,000/month
            "plan": "PLN_YOUR_PLAN_CODE",  # Replace with your Plan Code from Paystack Dashboard
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
    """Listens for background payment confirmations from Paystack."""
    body = await request.body()

    computed_signature = hmac.new(
        PAYSTACK_SECRET_KEY.encode('utf-8'),
        body,
        hashlib.sha512
    ).hexdigest()

    if computed_signature != x_paystack_signature:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_data = json.loads(body)

    if event_data.get("event") == "charge.success":
        data = event_data["data"]
        metadata = data.get("metadata", {})
        user_id = metadata.get("user_id")

        if user_id:
            payment_type = metadata.get("payment_type")
            
            # Use execute() instead of single() to prevent exceptions when user has no profile
            profile_res = supabase.table("profiles").select("*").eq("id", user_id).execute()
            profile_exists = len(profile_res.data) > 0
            current_credits = profile_res.data[0].get("credits", 0) if profile_exists else 0

            if payment_type == "credit_pack":
                credits_to_add = metadata.get("credits_to_add", 50)
                new_credits = current_credits + credits_to_add
                
                if profile_exists:
                    supabase.table("profiles").update({
                        "credits": new_credits,
                        "plan_type": "credit_pack"
                    }).eq("id", user_id).execute()
                else:
                    supabase.table("profiles").insert({
                        "id": user_id, 
                        "credits": new_credits,
                        "plan_type": "credit_pack"
                    }).execute()

            elif payment_type == "pro_subscription":
                if profile_exists:
                    supabase.table("profiles").update({
                        "plan_type": "pro_subscriber"
                    }).eq("id", user_id).execute()
                else:
                    supabase.table("profiles").insert({
                        "id": user_id,
                        "credits": current_credits,
                        "plan_type": "pro_subscriber"
                    }).execute()

    return {"status": "success"}


@app.post("/train")
async def train_model(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization token")

    token = authorization.split(" ")[1]
    
    # 1. Verify user identity via Supabase Auth
    user_response = supabase_admin.auth.get_user(token)
    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Invalid auth token")
        
    user_id = user_response.user.id

    # 2. Fetch credits directly using admin client
    profile = supabase_admin.table("profiles").select("credits").eq("id", user_id).execute()
    
    if not profile.data or profile.data[0].get("credits", 0) < 1:
        raise HTTPException(status_code=400, detail="Insufficient credits. Please purchase a credit pack.")

    current_credits = profile.data[0]["credits"]

    # --- YOUR MACHINE LEARNING TRAINING LOGIC HERE ---

    # 3. Deduct 1 credit after successful training execution
    supabase_admin.table("profiles").update({"credits": current_credits - 1}).eq("id", user_id).execute()

    return {"status": "success", "remaining_credits": current_credits - 1}


# --- 6. CORE ML ROUTE ---
@app.post("/upload-and-train/")
async def upload_and_train(
    file: UploadFile = File(...),
    target_column: str = Query(...),
    authorization: str = Header(None)
):
    # Extract real User ID from Supabase Auth Token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = authorization.split(" ")[1]
    user_response = supabase.auth.get_user(token)
    
    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Invalid session token")

    user_id = user_response.user.id

    # Query profiles using the authenticated user_id
    profile_response = supabase.table("profiles").select("*").eq("id", user_id).execute()
    
    if not profile_response.data:
        raise HTTPException(status_code=400, detail="Profile not found")

    user_profile = profile_response.data[0]
    credits = user_profile.get("credits", 0)
    plan_type = user_profile.get("plan_type", "free")

    if plan_type != "pro_subscriber" and credits <= 0:
        raise HTTPException(status_code=402, detail="Insufficient credits. Please purchase a credit pack.")

    # Deduct credit and proceed with training
    if plan_type != "pro_subscriber":
        supabase.table("profiles").update({"credits": credits - 1}).eq("id", user_id).execute()

    # ... REST OF YOUR ML PIPELINE CODE ...