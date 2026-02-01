import streamlit as st
from src.predict import predict_category

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="News Document Classification System",
    page_icon="📰",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2C3E50;'>
    📰 News Document Classification System
    </h1>
    <h4 style='text-align: center; color: gray;'>
    Business • Entertainment • Health
    </h4>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- DESCRIPTION ----------------
st.markdown(
    """
    This system uses **TF-IDF vectorization** and a **Naive Bayes classifier**
    to automatically categorize news documents into one of the following classes:
    
    - 📊 **Business**
    - 🎬 **Entertainment**
    - 🏥 **Health**
    """
)

# ---------------- TEXT INPUT ----------------
st.subheader("✍️ Enter a news article or sentence")

user_input = st.text_area(
    "",
    height=180,
    placeholder="Example: The stock market reacted positively after major companies reported strong earnings..."
)

# ---------------- BUTTON ----------------
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    classify_btn = st.button("🔍 Classify Document")

# ---------------- RESULT ----------------
if classify_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before classification.")
    else:
        category = predict_category(user_input)

        if category.lower() == "business":
            st.success("📊 **Predicted Category: BUSINESS**")
        elif category.lower() == "entertainment":
            st.success("🎬 **Predicted Category: ENTERTAINMENT**")
        elif category.lower() == "health":
            st.success("🏥 **Predicted Category: HEALTH**")
        else:
            st.info(f"Predicted Category: {category}")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray; font-size: 13px;'>
    Information Retrieval Project • Naive Bayes Text Classification
    </p>
    """,
    unsafe_allow_html=True
)
