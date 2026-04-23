import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
st.set_page_config(
    page_title="AI Data Analyzer",
    page_icon="🤖",
    layout="wide"
)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png",width=100)
    st.title("AI DATA ANALYZER")
    st.write("Built with Python, Streamlit and LLaMA AI")
    st.write("----------")
    st.write("### HOW TO USE:")
    st.write("1. Upload a 'csv' file")
    st.write("2. Explore your data and charts")
    st.write("3. Enter your Groq API key")
    st.write("4.Click 'Analyze with AI' to get insights")
    st.write("----------")
    st.write("Made by Neev Sardana")
st.title("AI DATA ANALYZER")
st.write("upload your csv file here and i will analyze it for you")
file=st.file_uploader("upload your csv file here",type=["csv"])
if file is not None:
    df=pd.read_csv(file)
    if df.shape[1]>50:
        st.write("warning your dataset has more than 50 columns , showing only first 50 columns")
        df=df.iloc[:,:50]
    st.write("### Your Data:")
    if df.shape[0]>10000:
        st.warning("warning your dataset exceeds the limit , shhowing only first 10000 rows")
        st.dataframe(df.head(10000))
    else:
        st.dataframe(df)
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    st.write(df.describe())
    st.write("## Your Charts")
    columns=df.columns.tolist()
    selected_columns=st.selectbox("select columns to vizualize",columns)
    chart_type=st.selectbox("select type of chart you want ",["Histogram","Bar chart","line chart"])
    fig,ax=plt.subplots()
    if chart_type=="Histogram":
        sns.histplot(data=df,x=selected_columns,ax=ax)
    elif chart_type=="Bar chart":
        df[selected_columns].value_counts().head(10).plot(kind="bar",ax=ax)
    elif chart_type=="line chart":
        st.write(df[selected_columns].dtype)
        if pd.api.types.is_numeric_dtype(df[selected_columns]):
            df[selected_columns].plot(kind="line",ax=ax)
        else:
            st.warning("warning: line chart is only for numeric data please select a numeric column")
            st.stop()
    st.pyplot(fig)
    st.write("### AI Insights")
    api_key=st.text_input("enter your API key here for insights by AI ",type="password")
    if api_key:
        if st.button("Analyze with AI "):
            with st.spinner("AI is analyzing your data..."):
                summary=df.describe().to_string()
                columns_info=str(df.columns.tolist())
                prompt=f"I have a dataset with these columns:{columns_info} and this is the summary {summary} analyze this and tell any 10 insight about the data in plain english"
                headers={
                    "Authorization": f"bearer {api_key}",
                    "Content-Type":"application/json"
                }
                body={
                    "model": "llama-3.3-70b-versatile",
                    "messages":[{"role":"user","content":prompt}]
                }
                response=requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers,json=body)
                result=response.json()
                if "choices" in result:
                    st.write(result["choices"][0]["message"]["content"])
                else:
                    st.write("something went wrong check YOUR API key.")