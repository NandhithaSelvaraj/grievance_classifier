"""
Member 3 — Dashboard / Experience Module
FastAPI backend: stores complaints, calls Member 2's classifier, tracks status.

Run with:
    uvicorn main:app --reload --port 8000

Member 2's classifier must be running separately, e.g.:
    uvicorn main:app --reload --port 8002   (run from inside her repo folder)
"""

import uuid
import sqlite3
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# URL where Member 2's classifier is running. Update the port if hers differs.
CLASSIFIER_URL = "http://127.0.0.1:8002/classify"

DB_PATH = "grievance.db"

app = FastAPI(title="Grievance Redressal Dashboard")

# Allow the dashboard.html (opened directly as a file, or from any origin)
# to call this API. For a college project this open policy is fine;
# tighten allow_origins for production use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            complaint_id TEXT PRIMARY KEY,
            citizen_name TEXT,
            contact TEXT,
            original_text TEXT,
            audio_url TEXT,
            translated_text TEXT,
            category TEXT,
            department TEXT,
            confidence REAL,
            status TEXT DEFAULT 'Pending',
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ComplaintCreate(BaseModel):
    citizen_name: str
    contact: str
    # For now, take translated_text directly (typed complaint).
    # Once Member 1's module is ready, this can instead come from her output.
    translated_text: str
    original_text: Optional[str] = None
    audio_url: Optional[str] = None


class StatusUpdate(BaseModel):
    status: str  # "Pending" | "In Progress" | "Resolved"


# ---------------------------------------------------------------------------
# Helper: call Member 2's classifier
# ---------------------------------------------------------------------------

def classify_complaint(text: str) -> dict:
    """
    Calls Member 2's classifier API and returns category/department/confidence.
    Falls back to safe defaults if her service is down, so the dashboard
    never crashes during development.
    """
    try:
        response = httpx.post(CLASSIFIER_URL, json={"text": text}, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except (httpx.RequestError, httpx.HTTPStatusError):
        return {
            "category": "Uncategorized",
            "department": "Needs manual review",
            "confidence": 0.0,
        }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/complaints")
def create_complaint(payload: ComplaintCreate):
    classification = classify_complaint(payload.translated_text)

    complaint_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    conn = get_db()
    conn.execute(
        """
        INSERT INTO complaints (
            complaint_id, citizen_name, contact,
            original_text, audio_url, translated_text,
            category, department, confidence,
            status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            complaint_id,
            payload.citizen_name,
            payload.contact,
            payload.original_text,
            payload.audio_url,
            payload.translated_text,
            classification["category"],
            classification["department"],
            classification["confidence"],
            "Pending",
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()

    return {"complaint_id": complaint_id, **classification, "status": "Pending"}


@app.get("/complaints")
def list_complaints(status: Optional[str] = None, department: Optional[str] = None):
    conn = get_db()
    query = "SELECT * FROM complaints WHERE 1=1"
    params = []

    if status:
        query += " AND status = ?"
        params.append(status)
    if department:
        query += " AND department = ?"
        params.append(department)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.get("/complaints/by-contact/{contact}")
def get_complaints_by_contact(contact: str):
    """
    Lets a citizen look up all their complaints using just their
    contact number, since most people won't remember a complaint ID.
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM complaints WHERE contact = ? ORDER BY created_at DESC",
        (contact,),
    ).fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No complaints found for this contact number")
    return [dict(row) for row in rows]


@app.get("/complaints/{complaint_id}")
def get_complaint(complaint_id: str):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,)
    ).fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return dict(row)


@app.patch("/complaints/{complaint_id}")
def update_status(complaint_id: str, payload: StatusUpdate):
    valid_statuses = {"Pending", "In Progress", "Resolved"}
    if payload.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid_statuses}")

    conn = get_db()
    row = conn.execute(
        "SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,)
    ).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Complaint not found")

    now = datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE complaints SET status = ?, updated_at = ? WHERE complaint_id = ?",
        (payload.status, now, complaint_id),
    )
    conn.commit()
    conn.close()

    # Notification stub — swap for real Twilio/SendGrid later
    print(f"[NOTIFY] {row['contact']}: your complaint {complaint_id} is now '{payload.status}'")

    return {"complaint_id": complaint_id, "status": payload.status, "updated_at": now}