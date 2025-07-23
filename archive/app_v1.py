import sqlite3
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
import matplotlib.pyplot as plt
import io
import base64
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import asyncio

import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt


# ---------------- CONFIGURATION ---------------- #
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env file!")
genai.configure(api_key=GOOGLE_API_KEY)

DB_PATH = "ecommerce.db"

app = FastAPI(
    title="E-commerce AI Agent",
    description="Ask natural language questions about sales data, with optional chart visualization and event streaming.",
    version="1.3"
)

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
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content([prompt[0], question])
    return response.text.strip().replace("```sql", "").replace("```", "").strip()

def execute_sql(sql_query: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        return {"error": str(e)}

def dataframe_to_dict(df: pd.DataFrame):
    return df.to_dict(orient="records")

def create_chart_image(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> bytes:
    """Generate bar chart and return raw PNG bytes."""
    plt.figure(figsize=(8, 5))
    plt.bar(df[x_col], df[y_col], color="skyblue")
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    plt.close()
    return img_buffer.getvalue()

def should_visualize(df: pd.DataFrame) -> bool:
    """Check if data is suitable for visualization (2 columns, numeric y-axis)."""
    return (
        not df.empty
        and len(df.columns) >= 2
        and pd.api.types.is_numeric_dtype(df[df.columns[1]])
    )

# ---------------- API ENDPOINTS ---------------- #
class QuestionRequest(BaseModel):
    question: str
    visualize: bool = False  # Optional chart flag


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """Answer any question dynamically using Gemini-generated SQL."""
    sql_query = generate_sql(request.question)
    df = execute_sql(sql_query)

    if isinstance(df, dict) and "error" in df:
        return JSONResponse(content={"error": df["error"], "sql_query": sql_query})

    response = {"sql_query": sql_query, "results": dataframe_to_dict(df)}

    # Add chart URL if visualizable
    if request.visualize and should_visualize(df):
        response["chart_url"] = f"/ask/chart?question={request.question}"

    return response


@app.get("/ask/chart")
def ask_chart(question: str):
    """Generate chart and return as image/png."""
    sql_query = generate_sql(question)
    df = execute_sql(sql_query)

    if isinstance(df, dict) and "error" in df:
        return JSONResponse(content={"error": df["error"], "sql_query": sql_query})

    if not should_visualize(df):
        return JSONResponse(content={"sql_query": sql_query, "message": "No data available to visualize."})

    img_bytes = create_chart_image(df, df.columns[0], df.columns[1], question)
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


@app.get("/ask/stream")
async def stream_answer(question: str):
    """Event streaming endpoint for real-time response."""
    async def event_stream():
        yield "data: Generating SQL query...\n\n"
        await asyncio.sleep(0.5)
        sql_query = generate_sql(question)
        yield f"data: SQL Query: {sql_query}\n\n"

        await asyncio.sleep(0.5)
        df = execute_sql(sql_query)
        if isinstance(df, dict) and "error" in df:
            yield f"data: Error: {df['error']}\n\n"
        else:
            result_str = str(dataframe_to_dict(df))
            yield f"data: Results: {result_str}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ---------------- RUN ---------------- #
# Run: uvicorn app:app --reload
