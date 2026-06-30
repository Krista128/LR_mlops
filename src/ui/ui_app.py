from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = Path(os.getenv("UI_TEMPLATE_DIR", BASE_DIR / "ui_templates"))
STATIC_DIR = Path(os.getenv("UI_STATIC_DIR", BASE_DIR / "ui_static"))

APP_TITLE = os.getenv("APP_TITLE", "Employee Attrition UI")
PREDICTOR_API_URL = os.getenv(
    "PREDICTOR_API_URL", "http://predictor-service:8000"
).rstrip("/")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow-service:5000").rstrip("/")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://grafana-service:3000").rstrip(
    "/"
)


app = FastAPI(title=APP_TITLE)
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Эти поля нужны predictor API, но модель их дальше дропает.
HIDDEN_DEFAULTS: dict[str, Any] = {
    "EmployeeCount": 1,
    "EmployeeNumber": 0,
    "MonthlyIncome": 0,
    "Over18": "Y",
    "PercentSalaryHike": 15,
    "StandardHours": 80,
    "TotalWorkingYears": 0,
    "YearsInCurrentRole": 0,
    "YearsWithCurrManager": 0,
}

# Видимые поля UI. Поля из HIDDEN_DEFAULTS намеренно не показываем.
FIELDS: list[dict[str, Any]] = [
    {
        "name": "Age",
        "label": "Age",
        "type": "number",
        "default": 35,
        "min": 18,
        "max": 65,
    },
    {
        "name": "BusinessTravel",
        "label": "Business travel",
        "type": "select",
        "default": "Travel_Rarely",
        "options": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
    },
    {
        "name": "DailyRate",
        "label": "Daily rate",
        "type": "number",
        "default": 800,
        "min": 100,
        "max": 1500,
    },
    {
        "name": "Department",
        "label": "Department",
        "type": "select",
        "default": "Research & Development",
        "options": ["Sales", "Research & Development", "Human Resources"],
    },
    {
        "name": "DistanceFromHome",
        "label": "Distance from home",
        "type": "number",
        "default": 5,
        "min": 1,
        "max": 29,
    },
    {
        "name": "Education",
        "label": "Education",
        "type": "select",
        "default": 3,
        "options": [1, 2, 3, 4, 5],
    },
    {
        "name": "EducationField",
        "label": "Education field",
        "type": "select",
        "default": "Life Sciences",
        "options": [
            "Life Sciences",
            "Other",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
        ],
    },
    {
        "name": "EnvironmentSatisfaction",
        "label": "Environment satisfaction",
        "type": "select",
        "default": 3,
        "options": [1, 2, 3, 4],
    },
    {
        "name": "Gender",
        "label": "Gender",
        "type": "select",
        "default": "Male",
        "options": ["Female", "Male"],
    },
    {
        "name": "HourlyRate",
        "label": "Hourly rate",
        "type": "number",
        "default": 65,
        "min": 30,
        "max": 100,
    },
    {
        "name": "JobInvolvement",
        "label": "Job involvement",
        "type": "select",
        "default": 3,
        "options": [1, 2, 3, 4],
    },
    {
        "name": "JobLevel",
        "label": "Job level",
        "type": "select",
        "default": 2,
        "options": [1, 2, 3, 4, 5],
    },
    {
        "name": "JobRole",
        "label": "Job role",
        "type": "select",
        "default": "Research Scientist",
        "options": [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources",
        ],
    },
    {
        "name": "JobSatisfaction",
        "label": "Job satisfaction",
        "type": "select",
        "default": 3,
        "options": [1, 2, 3, 4],
    },
    {
        "name": "MaritalStatus",
        "label": "Marital status",
        "type": "select",
        "default": "Married",
        "options": ["Single", "Married", "Divorced"],
    },
    {
        "name": "MonthlyRate",
        "label": "Monthly rate",
        "type": "number",
        "default": 15000,
        "min": 2000,
        "max": 27000,
    },
    {
        "name": "NumCompaniesWorked",
        "label": "Number of companies worked",
        "type": "select",
        "default": 2,
        "options": list(range(10)),
    },
    {
        "name": "OverTime",
        "label": "Overtime",
        "type": "select",
        "default": "No",
        "options": ["Yes", "No"],
    },
    {
        "name": "PerformanceRating",
        "label": "Performance rating",
        "type": "select",
        "default": 3,
        "options": [3, 4],
    },
    {
        "name": "RelationshipSatisfaction",
        "label": "Relationship satisfaction",
        "type": "select",
        "default": 3,
        "options": [1, 2, 3, 4],
    },
    {
        "name": "StockOptionLevel",
        "label": "Stock option level",
        "type": "select",
        "default": 1,
        "options": [0, 1, 2, 3],
    },
    {
        "name": "TrainingTimesLastYear",
        "label": "Training times last year",
        "type": "select",
        "default": 3,
        "options": [0, 1, 2, 3, 4, 5, 6],
    },
    {
        "name": "WorkLifeBalance",
        "label": "Work-life balance",
        "type": "select",
        "default": 3,
        "options": [1, 2, 3, 4],
    },
    {
        "name": "YearsAtCompany",
        "label": "Years at company",
        "type": "number",
        "default": 5,
        "min": 0,
        "max": 40,
    },
    {
        "name": "YearsSinceLastPromotion",
        "label": "Years since last promotion",
        "type": "number",
        "default": 1,
        "min": 0,
        "max": 15,
    },
]

INT_FIELDS = {
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsSinceLastPromotion",
    *HIDDEN_DEFAULTS.keys(),
}

_engine: Engine | None = None


def get_db_url() -> str:
    host = os.getenv("DB_HOST", "postgres-service")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "attr_db")
    user = os.getenv("POSTGRES_USER", "user1")
    password = os.getenv("POSTGRES_PASSWORD", "")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(get_db_url(), pool_pre_ping=True)
    return _engine


def build_payload(form_data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(HIDDEN_DEFAULTS)
    for field in FIELDS:
        name = field["name"]
        value = form_data.get(name, field.get("default"))
        if name in INT_FIELDS:
            value = int(value)
        payload[name] = value
    return payload


def attrition_text(value: Any) -> str:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return "Unknown"
    if numeric == 1:
        return "True — attrition risk detected"
    if numeric == 0:
        return "False — attrition risk not detected"
    return "Unknown"


def bool_text(value: Any, none_text: str = "Waiting") -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return none_text


def bool_badge_class(value: Any) -> str:
    if value is True:
        return "badge badge-danger"
    if value is False:
        return "badge badge-ok"
    return "badge badge-muted"


def short_run(run_id: Any) -> str:
    text_value = "" if run_id is None else str(run_id)
    return text_value if len(text_value) <= 12 else f"{text_value[:12]}..."


def fetch_history(limit: int = 50) -> list[dict[str, Any]]:
    query = text("""
        SELECT
            row_id,
            request_time,
            attrition,
            age,
            businesstravel,
            department,
            jobrole,
            overtime,
            jobsatisfaction,
            worklifebalance,
            model_train_run_id
        FROM history
        ORDER BY row_id DESC
        LIMIT :limit;
        """)
    with get_engine().begin() as conn:
        result = conn.execute(query, {"limit": limit})
        rows = result.mappings().all()

    readable_rows = []
    for row in rows:
        readable_rows.append(
            {
                "Case ID": row["row_id"],
                "Time": row["request_time"],
                "Prediction": attrition_text(row["attrition"]),
                "Age": row["age"],
                "Business travel": row["businesstravel"],
                "Department": row["department"],
                "Job role": row["jobrole"],
                "Overtime": row["overtime"],
                "Job satisfaction": row["jobsatisfaction"],
                "Work-life balance": row["worklifebalance"],
                "Model run": short_run(row["model_train_run_id"]),
            }
        )
    return readable_rows


def fetch_drift(limit: int = 50) -> list[dict[str, Any]]:
    query = text("""
        SELECT
            window_id,
            w_start,
            w_stop,
            run_id,
            data_drift,
            target_drift,
            concept_drift,
            trained
        FROM drift
        ORDER BY window_id DESC
        LIMIT :limit;
        """)
    with get_engine().begin() as conn:
        result = conn.execute(query, {"limit": limit})
        rows = result.mappings().all()

    readable_rows = []
    for row in rows:
        readable_rows.append(
            {
                "Window ID": row["window_id"],
                "Rows": f"{row['w_start']}–{row['w_stop']}",
                "Model run": short_run(row["run_id"]),
                "Data drift": bool_text(row["data_drift"]),
                "Target drift": bool_text(
                    row["target_drift"], none_text="Waiting for labels"
                ),
                "Concept drift": bool_text(
                    row["concept_drift"], none_text="Waiting for labels"
                ),
                "Retrained": "Yes" if row["trained"] else "No",
                "_classes": {
                    "Data drift": bool_badge_class(row["data_drift"]),
                    "Target drift": bool_badge_class(row["target_drift"]),
                    "Concept drift": bool_badge_class(row["concept_drift"]),
                    "Retrained": (
                        "badge badge-ok"
                        if row["trained"]
                        else "badge badge-muted"
                    ),
                },
            }
        )
    return readable_rows


def predictor_health() -> dict[str, Any]:
    try:
        response = requests.get(f"{PREDICTOR_API_URL}/health", timeout=3.0)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"status": f"unavailable: {type(exc).__name__}", "run_id": None}


def db_status() -> str:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return "available"
    except SQLAlchemyError as exc:
        return f"unavailable: {type(exc).__name__}"


@app.get("/", response_class=HTMLResponse)
def index() -> RedirectResponse:
    return RedirectResponse(url="/inference")


@app.get("/inference", response_class=HTMLResponse)
def inference_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "inference.html",
        {
            "request": request,
            "title": APP_TITLE,
            "active": "inference",
            "fields": FIELDS,
            "payload": None,
            "result": None,
            "error": None,
            "predictor_health": predictor_health(),
            "db_status": db_status(),
        },
    )


@app.post("/inference", response_class=HTMLResponse)
async def submit_inference(request: Request) -> HTMLResponse:
    form = await request.form()
    form_data = dict(form)
    payload = None
    result = None
    error = None

    try:
        payload = build_payload(form_data)
        response = requests.post(
            f"{PREDICTOR_API_URL}/predict", json=payload, timeout=10.0
        )
        response.raise_for_status()
        raw_result = response.json()
        result = {
            "Prediction": attrition_text(raw_result.get("Attrition")),
            "Case ID": raw_result.get("Your_case_id"),
        }
    except (ValueError, requests.RequestException) as exc:
        error = f"Prediction request failed: {type(exc).__name__}: {exc}"

    return templates.TemplateResponse(
        "inference.html",
        {
            "request": request,
            "title": APP_TITLE,
            "active": "inference",
            "fields": FIELDS,
            "payload": payload,
            "result": result,
            "error": error,
            "predictor_health": predictor_health(),
            "db_status": db_status(),
        },
    )


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request, limit: int = 50) -> HTMLResponse:
    limit = max(10, min(limit, 500))
    rows: list[dict[str, Any]] = []
    error = None
    try:
        rows = fetch_history(limit)
    except SQLAlchemyError as exc:
        error = f"Could not read history table: {type(exc).__name__}: {exc}"

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "title": APP_TITLE,
            "active": "history",
            "rows": rows,
            "limit": limit,
            "error": error,
        },
    )


@app.get("/drift", response_class=HTMLResponse)
def drift_page(request: Request, limit: int = 50) -> HTMLResponse:
    limit = max(10, min(limit, 500))
    rows: list[dict[str, Any]] = []
    error = None
    try:
        rows = fetch_drift(limit)
    except SQLAlchemyError as exc:
        error = f"Could not read drift table: {type(exc).__name__}: {exc}"

    return templates.TemplateResponse(
        "drift.html",
        {
            "request": request,
            "title": APP_TITLE,
            "active": "drift",
            "rows": rows,
            "limit": limit,
            "error": error,
        },
    )


@app.get("/tools", response_class=HTMLResponse)
def tools_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "tools.html",
        {
            "request": request,
            "title": APP_TITLE,
            "active": "tools",
            "predictor_health": predictor_health(),
            "db_status": db_status(),
            "mlflow_proxy_url": "/external/mlflow/",
            "grafana_proxy_url": "/external/grafana/",
        },
    )


@app.get("/external/{service}")
def external_root(service: str) -> RedirectResponse:
    return RedirectResponse(url=f"/external/{service}/")


@app.api_route(
    "/external/{service}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def external_proxy(
    service: str, path: str, request: Request
) -> Response:
    bases = {
        "mlflow": MLFLOW_URL,
        "grafana": GRAFANA_URL,
    }
    if service not in bases:
        raise HTTPException(status_code=404, detail="Unknown external service")

    base_url = bases[service]
    target_url = f"{base_url}/{path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    request_headers = dict(request.headers)
    for key in ["host", "content-length", "accept-encoding"]:
        request_headers.pop(key, None)

    try:
        async with httpx.AsyncClient(
            follow_redirects=False, timeout=30.0
        ) as client:
            upstream = await client.request(
                request.method,
                target_url,
                headers=request_headers,
                content=await request.body(),
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"Proxy error: {exc}"
        ) from exc

    excluded_headers = {
        "content-encoding",
        "transfer-encoding",
        "connection",
        "content-length",
    }
    headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in excluded_headers
    }

    location = headers.get("location")
    if location:
        if location.startswith(base_url):
            headers["location"] = location.replace(
                base_url, f"/external/{service}", 1
            )
        elif location.startswith("/"):
            headers["location"] = f"/external/{service}{location}"

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers,
    )
