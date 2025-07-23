import sqlite3
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import time

# ---------------- LOAD ENV ---------------- #
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ---------------- CONFIGURATION ---------------- #
DB_PATH = "ecommerce.db"

app = FastAPI(
    title="E-commerce AI Agent",
    description="Ask any natural language question about sales data and get SQL-powered answers",
    version="1.0"
)

# Enable CORS for UI access
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

def create_plot(df, x_col, y_col, title="Chart"):
    """Generate bar chart and return as base64 image."""
    plt.figure(figsize=(6, 4))
    plt.bar(df[x_col], df[y_col], color='skyblue')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64

# ---------------- API MODELS ---------------- #
class QuestionRequest(BaseModel):
    question: str
    visualize: bool = False
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import io

@app.get("/visualize")
def visualize(question: str):
    """Generate visualization for a given question as an image."""
    sql_query = generate_sql(question)
    df = execute_sql(sql_query)
    if isinstance(df, dict) and "error" in df:
        return JSONResponse(content={"error": df["error"], "sql_query": sql_query})

    if len(df.columns) >= 2:
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

# ---------------- MAIN ASK ENDPOINT ---------------- #
@app.post("/ask")
def ask_question(request: QuestionRequest):
    """Answer any question dynamically using Gemini-generated SQL."""
    question = request.question
    sql_query = generate_sql(question)
    df = execute_sql(sql_query)

    if isinstance(df, dict) and "error" in df:
        return JSONResponse(content={"error": df["error"], "sql_query": sql_query})

    response = {"sql_query": sql_query, "results": dataframe_to_dict(df)}

    # Optional visualization
    if request.visualize and not df.empty and len(df.columns) >= 2:
        img = create_plot(df, df.columns[0], df.columns[1], question)
        response["chart_base64"] = f"data:image/png;base64,{img}"

    return response

# ---------------- STREAMING ENDPOINT ---------------- #
@app.post("/ask/stream")
def ask_question_stream(request: QuestionRequest):
    """Stream response (SQL + results) gradually for live typing effect."""

    def stream():
        question = request.question
        yield "Generating SQL query...\n"
        sql_query = generate_sql(question)
        for char in f"SQL Query: {sql_query}\n\n":
            yield char
            time.sleep(0.02)

        yield "Fetching results...\n"
        df = execute_sql(sql_query)
        if isinstance(df, dict) and "error" in df:
            yield f"Error: {df['error']}\n"
            return

        results = dataframe_to_dict(df)
        if not results:
            yield "No results found.\n"
            return

        yield "Results:\n"
        for row in results:
            yield f"{row}\n"
            time.sleep(0.05)

    return StreamingResponse(stream(), media_type="text/plain")

# ---------------- RUN ---------------- #
# Run with: uvicorn app:app --reload
