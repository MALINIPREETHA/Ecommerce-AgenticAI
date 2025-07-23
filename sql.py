import sqlite3
import pandas as pd
from datetime import datetime
import numpy as np
import os

base_path = "C:\\Users\\malin\\Documents\\AgenticAI\\datasets\\"


def load_and_clean_data():
    # Loading given datasets
    prd_adsales = pd.read_csv(base_path + "ProductLevelAdSalesandMetrics.csv")
    prd_elig = pd.read_csv(base_path + "ProductLevelEligibilityTable.csv")
    prd_totalsales = pd.read_csv(base_path + "ProductLevelTotalSalesandMetrics.csv")

    # Cleaning column names
    prd_adsales.rename(columns={
        "ad sales": "ad_sales",
        "impressions": "impressions",
        "ad spend": "ad_spend",
        "clicks": "clicks",
        "units sold": "units_sold"
    }, inplace=True, errors="ignore")

    prd_elig.rename(columns={
        "eligibility datetime utc": "eligibility_datetime_utc"
    }, inplace=True, errors="ignore")

    prd_totalsales.rename(columns={
        "total sales": "total_sales",
        "total units ordered": "total_units_ordered"
    }, inplace=True, errors="ignore")

    # Droping duplicates
    prd_adsales.drop_duplicates(subset=["date", "item_id"], keep="first", inplace=True)
    prd_totalsales.drop_duplicates(subset=["date", "item_id"], keep="first", inplace=True)
    prd_elig.drop_duplicates(subset=["item_id", "eligibility_datetime_utc"], keep="first", inplace=True)

    return prd_adsales, prd_elig, prd_totalsales


def create_tables(cursor):
    tables = [
        """CREATE TABLE PRD_ADSALES (
            date TEXT,
            item_id TEXT,
            ad_sales REAL,
            impressions INTEGER,
            ad_spend REAL,
            clicks INTEGER,
            units_sold INTEGER,
            load_date DATE,
            latest_updt_ts TIMESTAMP
        )""",
        """CREATE TABLE PRD_ELIG (
            eligibility_datetime_utc TEXT,
            item_id TEXT,
            eligibility TEXT,
            message TEXT,
            load_date DATE,
            latest_updt_ts TIMESTAMP
        )""",
        """CREATE TABLE PRD_TOTALSALES (
            date TEXT,
            item_id TEXT,
            total_sales REAL,
            total_units_ordered INTEGER,
            load_date DATE,
            latest_updt_ts TIMESTAMP
        )"""
    ]

    # Droping old tables if they exist
    for table in ["PRD_ADSALES", "PRD_ELIG", "PRD_TOTALSALES"]:
        cursor.execute(f"DROP TABLE IF EXISTS {table};")

    # Creating new tables
    for create_stmt in tables:
        cursor.execute(create_stmt)


# Inserting data into tables

def insert_data(conn, prd_adsales, prd_elig, prd_totalsales):
    datasets = [
        ("PRD_ADSALES", prd_adsales),
        ("PRD_ELIG", prd_elig),
        ("PRD_TOTALSALES", prd_totalsales)
    ]

    for table_name, df in datasets:
        # Add audit columns
        df["load_date"] = datetime.now().strftime("%Y-%m-%d")
        df["latest_updt_ts"] = datetime.now()
        try:
            df.to_sql(table_name, conn, if_exists="append", index=False)
            print(f" Inserted {len(df)} records into {table_name}")
        except Exception as e:
            print(f" Error inserting into {table_name}: {e}")

# Main Execution

if __name__ == "__main__":
    try:
     
        prd_adsales, prd_elig, prd_totalsales = load_and_clean_data()

        # Connect to database
        conn = sqlite3.connect("ecommerce.db")
        cursor = conn.cursor()

        # Create tables
        create_tables(cursor)

        # Insert data
        insert_data(conn, prd_adsales, prd_elig, prd_totalsales)

        conn.commit()
        print("\n E-commerce database created successfully!")

    except Exception as e:
        print(f" Error: {str(e)}")

    finally:
        if 'conn' in locals():
            conn.close()
