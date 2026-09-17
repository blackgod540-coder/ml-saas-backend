import os
import shutil
import hmac
import hashlib
import json
import io
import uuid
import requests
import traceback
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, UploadFile, File, Query, Depends, HTTPException, Security, Request, Header
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from supabase import create_client, Client

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

class PredictionRequest(BaseModel):
    input_data: dict


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
        amount_kobo = 5000 * 100  # ₦5,000 for 10 credits
        metadata = {"user_id": payload.user_id, "payment_type": "credit_pack", "credits_to_add": 10}
        data = {
            "email": payload.email,
            "amount": amount_kobo,
            "metadata": metadata
        }
    elif payload.payment_type == "pro_subscription":
        data = {
            "email": payload.email,
            "amount": 15000 * 100,  # ₦15,000/month
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
    """Listens for background payment confirmations and recurring charges."""
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
    
    # --- WEBHOOK REPLAY GUARD ---
    if reference:
        tx_check = supabase.table("transactions").select("reference").eq("reference", reference).execute()
        if len(tx_check.data) > 0:
            print(f"-> [WEBHOOK] Transaction {reference} already processed. Ignoring replay attack.")
            return {"status": "success", "message": "already_processed"}

    # Handle Successful Charges (Initial and Recurring)
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
                    supabase.table("profiles").update({
                        "credits": new_credits,
                        "plan_type": "free"
                    }).eq("id", user_id).execute()
                else:
                    supabase.table("profiles").insert({
                        "id": user_id, 
                        "credits": new_credits,
                        "plan_type": "free"
                    }).execute()

            elif payment_type == "pro_subscription":
                expiry_date = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
                
                if profile_exists:
                    supabase.table("profiles").update({
                        "plan_type": "pro_subscriber",
                        "subscription_expires_at": expiry_date
                    }).eq("id", user_id).execute()
                else:
                    supabase.table("profiles").insert({
                        "id": user_id,
                        "credits": current_credits,
                        "plan_type": "pro_subscriber",
                        "subscription_expires_at": expiry_date
                    }).execute()

            # Log the transaction to prevent future replay attacks
            if reference:
                supabase.table("transactions").insert({
                    "reference": reference,
                    "user_id": user_id,
                    "event_type": event_type
                }).execute()

    # Handle Failed Recurring Subscriptions
    elif event_type in ["invoice.payment_failed", "subscription.disable", "charge.failed"]:
        metadata = data.get("metadata", {})
        user_id = metadata.get("user_id")
        
        if user_id:
            supabase.table("profiles").update({
                "plan_type": "free"
            }).eq("id", user_id).execute()
            
            if reference:
                supabase.table("transactions").insert({
                    "reference": reference,
                    "user_id": user_id,
                    "event_type": event_type
                }).execute()

    return {"status": "success"}


# --- 6. CORE ML ROUTE (WITH ATOMIC CREDIT DEDUCTION) ---
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

    # --- TIME-LOCK VALIDATION ---
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

    # --- ATOMIC CREDIT DEDUCTION (RPC) ---
    if plan_type != "pro_subscriber":
        # Let Postgres handle the credit check and deduction atomically to prevent race conditions
        deduction_res = supabase.rpc("decrement_credits", {"user_id_param": user_id}).execute()
        if not deduction_res.data:
            raise HTTPException(status_code=402, detail="Insufficient credits. Please purchase a credit pack or upgrade.")

    try:
        content = await file.read()
        
        # --- FILE SIZE GUARD (Max 15MB to prevent Render RAM crashes) ---
        MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 25MB.")

        filename = file.filename.lower()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel.")

        # --- ROW COUNT GUARD (Max 100,000 rows) ---
        if len(df) > 100000:
            raise HTTPException(status_code=400, detail="Dataset exceeds the maximum limit of 100,000 rows.")

        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset.")

        df = df.dropna(subset=[target_column])

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X = X.loc[:, X.nunique() > 1]
        if X.empty:
            raise HTTPException(status_code=400, detail="Dataset has no valid feature columns remaining after filtering.")

        is_classification = True
        if pd.api.types.is_numeric_dtype(y):
            if y.nunique() > 20:
                is_classification = False

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numeric_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        if is_classification:
            model = RandomForestClassifier(random_state=42)
            model_type = "classification"
        else:
            model = RandomForestRegressor(random_state=42)
            model_type = "regression"

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        if is_classification:
            avg_type = 'binary' if y.nunique() == 2 else 'weighted'
            performance_metrics = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, average=avg_type, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, average=avg_type, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, average=avg_type, zero_division=0)), 4)
            }
        else:
            performance_metrics = {
                "r2_score": round(float(r2_score(y_test, y_pred)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                "mae": round(float(mean_absolute_error(y_test, y_pred)), 4)
            }

        feature_defaults = {}
        categorical_options = {}
        for col in X.columns:
            if col in numeric_cols:
                feature_defaults[col] = float(X[col].median()) if not X[col].empty else 0.0
            else:
                top_val = str(X[col].mode()[0]) if not X[col].mode().empty else "missing"
                feature_defaults[col] = top_val
                categorical_options[col] = X[col].dropna().astype(str).unique().tolist()[:50]

        model_id = f"mod_{uuid.uuid4().hex[:10]}"
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"{model_id}.joblib")
        joblib.dump(pipeline, model_path)

        # --- BACKUP MODEL TO SUPABASE STORAGE ---
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
            "details": {
                "model_id": model_id,
                "model_type": model_type,
                "features_used": list(X.columns),
                "performance_metrics": performance_metrics,
                "feature_defaults": feature_defaults,
                "categorical_options": categorical_options
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.post("/predict/{model_id}")
async def predict_model(model_id: str, payload: PredictionRequest):
    model_path = os.path.join("models", f"{model_id}.joblib")
    
    if not os.path.exists(model_path):
        try:
            print(f"-> [STORAGE] Model {model_id} missing locally. Downloading from Supabase Storage...")
            file_bytes = supabase.storage.from_("models").download(f"{model_id}.joblib")
            os.makedirs("models", exist_ok=True)
            with open(model_path, "wb") as f_out:
                f_out.write(file_bytes)
        except Exception as e:
            raise HTTPException(status_code=404, detail="Model not found or expired in cloud storage.")

    try:
        loaded_pipeline = joblib.load(model_path)
        input_df = pd.DataFrame([payload.input_data])
        prediction = loaded_pipeline.predict(input_df)[0]
        
        if isinstance(prediction, (np.integer, np.floating)):
            prediction = prediction.item()
            
        return {"prediction": prediction}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-model/{model_id}")
async def download_model(model_id: str):
    model_path = os.path.join("models", f"{model_id}.joblib")
    
    if not os.path.exists(model_path):
        try:
            print(f"-> [STORAGE] Model {model_id} missing locally for download. Fetching from Supabase...")
            file_bytes = supabase.storage.from_("models").download(f"{model_id}.joblib")
            os.makedirs("models", exist_ok=True)
            with open(model_path, "wb") as f_out:
                f_out.write(file_bytes)
        except Exception as e:
            raise HTTPException(status_code=404, detail="Model file not found in cloud storage.")
            
    return FileResponse(model_path, media_type="application/octet-stream", filename=f"{model_id}.joblib")