import streamlit as st
import pandas as pd
import os
import shutil
from pathlib import Path
from edapipeline.core import EDAPipeline
from edapipeline.nlp_insight import NLPInsightGenerator

# Define the output directory for EDA pipeline
EDA_OUTPUT_DIR = "./streamlit_eda_outputs"

st.set_page_config(page_title="EDA & NLP Insight Dashboard", layout="wide")

st.title("Automated EDA & NLP Insight Generator 📊🧠")
st.markdown("Upload your dataset to generate an automated Exploratory Data Analysis report enriched with NLP insights using T5.")

@st.cache_resource
def load_nlp_model():
    return NLPInsightGenerator(model_name="t5-small")

nlp_generator = load_nlp_model()

def clear_previous_outputs():
    if os.path.exists(EDA_OUTPUT_DIR):
        shutil.rmtree(EDA_OUTPUT_DIR)

uploaded_file = st.file_uploader("Upload CSV or Excel dataset", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("File uploaded successfully!")
        
        st.subheader("Dataset Preview")
        st.write(df.head())
        
        st.write(f"**Shape:** {df.shape[0]} rows and {df.shape[1]} columns")
        
        # User selection for Target column (optional)
        target_col = st.selectbox("Select Target Column (Optional)", ["None"] + list(df.columns))
        target_col = None if target_col == "None" else target_col

        if st.button("Generate EDA & NLP Insights"):
            with st.spinner("Running EDA Pipeline and gathering insights... Please wait."):
                clear_previous_outputs()
                
                # We need to capture the logs. `EDAPipeline` writes to a text file when `save_outputs=True`.
                # We can also generate a short statistical summary suitable for the T5 prompt.
                eda = EDAPipeline(
                    df=df,
                    target_col=target_col,
                    save_outputs=True,
                    output_dir=EDA_OUTPUT_DIR
                )
                
                # Run the analysis
                eda.run_complete_analysis()
                
                st.success("Analysis complete!")
                
                # Generate NLP Insight
                # We format a summary context to pass to T5
                missing_values = df.isnull().sum().sum()
                num_cols = len(eda.numerical_cols)
                cat_cols = len(eda.categorical_cols)
                context = f"The dataset has {df.shape[0]} rows, {num_cols} numerical features, and {cat_cols} categorical features. "
                if missing_values > 0:
                    context += f"There are {missing_values} missing values in total. "
                else:
                    context += "There are no missing values. "
                    
                st.subheader("🧠 AI-Generated Insights (T5 Model)")
                insight = nlp_generator.generate_insight(context)
                st.info(insight)
                
                # Display textual log report
                st.subheader("📝 EDA Text Report")
                metrics_files = list(Path(EDA_OUTPUT_DIR).glob("*.txt"))
                if metrics_files:
                    with open(metrics_files[0], 'r', encoding='utf-8') as f:
                        text_report = f.read()
                        with st.expander("View Full Textual Metrics Log"):
                            st.text(text_report)
                
                st.subheader("📈 Visualizations")
                plots_dir = os.path.join(EDA_OUTPUT_DIR, "plots")
                if os.path.exists(plots_dir):
                    plot_files = sorted(os.listdir(plots_dir))
                    if not plot_files:
                        st.write("No plots generated.")
                    else:
                        cols = st.columns(2)
                        for idx, plot_file in enumerate(plot_files):
                            with cols[idx % 2]:
                                st.image(os.path.join(plots_dir, plot_file), use_container_width=True)
                                
    except Exception as e:
        st.error(f"Error processing the file: {e}")
