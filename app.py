import streamlit as st
import pandas as pd

from loader import load_dataset
from profiling import profile_dataframe
from analytics import run_analytics
from gemini import init_gemini, get_gemini_insights
from plots import generate_plots

# ------------------ PAGE CONFIG ------------------ #

st.set_page_config(
    page_title="Automated EDA with Gemini",
    layout="wide"
)

st.title("📊 Automated Exploratory Data Analysis")
st.caption("LLM-assisted EDA using Gemini + Pandas")

# ------------------ SIDEBAR ------------------ #

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xls", "xlsx"]
)


# ------------------ MAIN FLOW ------------------ #

if uploaded_file:
    try:
        # Load data
        df = load_dataset(uploaded_file)
        st.success("Dataset loaded successfully")

        # Preview
        with st.expander("🔍 Dataset Preview"):
            st.dataframe(df.head())

        # Column profiling
        profiling = profile_dataframe(df)
        column_profiles = profiling["columns"]

        # Analytics
        analytics = run_analytics(df, column_profiles)

        # ------------------ OVERVIEW ------------------ #

        st.header("📌 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", analytics["dataset"]["num_rows"])
        col2.metric("Columns", analytics["dataset"]["num_columns"])
        col3.metric("Missing (%)", analytics["dataset"]["total_missing_pct"])
        col4.metric("Duplicates (%)", analytics["dataset"]["duplicate_rows_pct"])

        # ------------------ COLUMN PROFILING ------------------ #

        st.header("🧩 Column Profiling")

        profiling_table = []

        for col, meta in column_profiles.items():
            profiling_table.append({
                "Column": col,
                "Logical Type": meta["logical_type"],
                "Null %": meta["null_pct"],
                "Unique Values": meta["unique_values"]
            })

        st.dataframe(pd.DataFrame(profiling_table))

        # ------------------ GEMINI INSIGHTS ------------------ #

        # ------------------ GEMINI INSIGHTS ------------------ #

        gemini_output = None
        
        try:
            init_gemini()  # Reads from environment only
            with st.spinner("Generating Gemini insights..."):
                gemini_output = get_gemini_insights(column_profiles, analytics)
        
            st.header("🧠 Gemini Insights")
        
            st.subheader("Dataset Summary")
            st.write(gemini_output.get("dataset_summary", ""))
        
            if gemini_output.get("data_quality_issues"):
                st.subheader("Data Quality Issues")
                for issue in gemini_output["data_quality_issues"]:
                    st.write(f"- {issue}")
        
            if gemini_output.get("key_insights"):
                st.subheader("Key Insights")
                for insight in gemini_output["key_insights"]:
                    st.write(f"- {insight}")
        
            if gemini_output.get("ml_suggestions"):
                st.subheader("ML Task Suggestions")
                for task in gemini_output["ml_suggestions"]:
                    st.write(f"- {task}")
        
        except Exception as e:
            st.error(f"Gemini insights unavailable: {e}")


        # ------------------ VISUALIZATIONS ------------------ #

        st.header("📈 Visualizations")
        
        if not gemini_output:
            st.info("Gemini insights are unavailable.")
        elif not gemini_output.get("plots"):
            st.info("Gemini did not suggest any visualizations for this dataset.")
        else:
            figures = generate_plots(
                df,
                gemini_output["plots"],
                column_profiles
            )
        
            if figures:
                for fig in figures:
                    st.pyplot(fig)
            else:
                st.info("No valid plots could be generated from Gemini suggestions.")

    except Exception as e:
        st.error(str(e))

else:
    st.info("👈 Upload a dataset to begin EDA")
