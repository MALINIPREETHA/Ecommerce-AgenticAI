import streamlit as st
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
import time

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="E-commerce AI Agent", layout="wide")
st.title("🛒 E-commerce AI Agent")
st.write("Ask natural language questions about your e-commerce data.")

question = st.text_input("Enter your question:")
visualize = st.checkbox("Visualize data", value=True)
stream_mode = st.checkbox("Enable Live Typing (Streaming Mode)", value=True)


def simulate_typing(response_iter, delay=0.02):
    """Simulates streaming text character by character."""
    placeholder = st.empty()
    buffer = ""
    for chunk in response_iter.iter_content(chunk_size=1, decode_unicode=True):
        if chunk:
            buffer += chunk
            placeholder.text(buffer)
            time.sleep(delay)
    return buffer


def display_chart_if_available(question):
    """Fetches chart (and table) from /ask endpoint after streaming is done."""
    try:
        response = requests.post(
            f"{API_BASE}/ask",
            json={"question": question, "visualize": True}
        )
        if response.status_code == 200:
            data = response.json()

            st.subheader("Generated SQL Query:")
            st.code(data.get("sql_query", ""), language="sql")

            results = data.get("results", [])
            if results:
                st.subheader("Results:")
                df = pd.DataFrame(results)
                st.dataframe(df)
            else:
                st.info("No results found.")

            if "chart_base64" in data:
                img_base64 = data["chart_base64"].split(",")[1]
                img_bytes = base64.b64decode(img_base64)
                img = Image.open(BytesIO(img_bytes))
                st.image(img, caption="Visualization", use_column_width=True)

        else:
            st.error(f"Error fetching visualization: {response.text}")

    except Exception as e:
        st.error(f"Failed to fetch visualization: {e}")


if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        if stream_mode:
            st.info("**Streaming Mode Enabled:** Watch responses appear in real-time.")
            try:
                with st.spinner("Processing your question..."):
                    response = requests.post(
                        f"{API_BASE}/ask_stream",
                        json={"question": question, "visualize": visualize},
                        stream=True
                    )
                    if response.status_code == 200:
                        simulate_typing(response)
                        st.success("Streaming finished!")
                        if visualize:
                            st.subheader("Fetching visualization...")
                            display_chart_if_available(question)
                    else:
                        st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

        else:
            with st.spinner("Fetching results..."):
                display_chart_if_available(question)
