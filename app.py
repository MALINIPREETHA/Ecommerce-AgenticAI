import sqlite3
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
import matplotlib.pyplot as plt
import io
import base64
import time
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# ---------------- LOAD ENV ---------------- #
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ---------------- FASTAPI APP ---------------- #
app = FastAPI(
    title="E-commerce AI Agent",
    description="Ask any natural language question about sales data and get SQL-powered answers",
    version="1.0"
)

DB_PATH = "ecommerce.db"

# Allow all CORS for UI testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- GEMINI PROMPT ---------------- #
prompt = ["""
You are an SQL query generator for an e-commerce database.

Database schema:
- PRD_ADSALES(date, item_id, ad_sales, impressions, ad_spend, clicks, units_sold)
- PRD_ELIG(eligibility_datetime_utc, item_id, eligibility, message)
- PRD_TOTALSALES(date, item_id, total_sales, total_units_ordered)

Rules:
1. Only return the SQL query. Do NOT include explanations or markdown formatting.
2. Use SUM, AVG, or ORDER BY when necessary.
3. Use table names and columns exactly as shown above.
4. For CPC (Cost Per Click), calculate: ad_spend * 1.0 / clicks (where clicks > 0).
5. For RoAS (Return on Ad Spend), calculate: SUM(ad_sales) / SUM(ad_spend).
6. Use LIMIT 10 for general "show top" queries to avoid large results.
"""]

# ---------------- HELPER FUNCTIONS ---------------- #
def generate_sql(question: str) -> str:
    """Convert natural language to SQL using Gemini."""
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content([prompt[0], question])
    sql_query = response.text.strip().replace("```sql", "").replace("```", "").strip()
    return sql_query

def execute_sql(sql_query: str):
    """Execute SQL query and return DataFrame."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        return {"error": str(e)}

def dataframe_to_dict(df: pd.DataFrame):
    return df.to_dict(orient="records")

# ---------------- API ENDPOINTS ---------------- #
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    """Answer a question using Gemini-generated SQL."""
    question = request.question
    sql_query = generate_sql(question)
    df = execute_sql(sql_query)

    if isinstance(df, dict) and "error" in df:
        return JSONResponse(content={"error": df["error"], "sql_query": sql_query})

    # Generate chart (if possible)
    chart_base64 = None
    if len(df.columns) >= 2 and len(df) > 0:
        chart_base64 = create_chart_base64(df, df.columns[0], df.columns[1], question)

    return {
        "sql_query": sql_query,
        "results": dataframe_to_dict(df),
        "chart_base64": chart_base64
    }

@app.get("/visualize")
def visualize(question: str = Query(..., description="Natural language question")):
    """Generate visualization (bar chart) for a given question."""
    sql_query = generate_sql(question)
    df = execute_sql(sql_query)

    if isinstance(df, dict) and "error" in df:
        return JSONResponse(content={"error": df["error"], "sql_query": sql_query})

    if len(df.columns) >= 2 and len(df) > 0:
        img = io.BytesIO()
        plt.figure(figsize=(6, 4))
        plt.bar(df[df.columns[0]], df[df.columns[1]], color="skyblue")
        plt.title(question)
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.tight_layout()
        plt.savefig(img, format="png")
        img.seek(0)
        return StreamingResponse(img, media_type="image/png")

    return {"sql_query": sql_query, "results": dataframe_to_dict(df)}

@app.post("/ask_stream")
def ask_stream(request: QuestionRequest):
    """Simulate streaming response (typing effect)."""
    question = request.question
    sql_query = generate_sql(question)
    df = execute_sql(sql_query)

    if isinstance(df, dict) and "error" in df:
        return JSONResponse(content={"error": df["error"], "sql_query": sql_query})

    def event_stream():
        response_text = f"SQL Query: {sql_query}\n\nResults:\n{dataframe_to_dict(df)}"
        for char in response_text:
            yield char
            time.sleep(0.03)  # typing delay

    return StreamingResponse(event_stream(), media_type="text/plain")

# ---------------- CHART HELPER ---------------- #
def create_chart_base64(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    img = io.BytesIO()
    plt.figure(figsize=(6, 4))
    plt.bar(df[x_col], df[y_col], color="skyblue")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(img, format="png")
    plt.close()
    img.seek(0)
    return "data:image/png;base64," + base64.b64encode(img.read()).decode()

# ---------------- RUN ---------------- #
# Run with: uvicorn app:app --reload
