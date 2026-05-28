import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Olist Customer Churn Prediction",
    page_icon="🛒",
    layout="wide"
)

# =========================
# CUSTOM TITLE
# =========================
st.title("🛒 Olist Customer Churn Prediction Dashboard")
st.markdown("""
This dashboard predicts whether a customer is likely to churn using a trained **LightGBM machine learning model**.
It helps identify high-risk customers and provides business recommendations.
""")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        features = joblib.load("features.pkl")
        return model, features
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, feature_cols = load_artifacts()

if model is None or feature_cols is None:
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🧾 Customer Details")

recency_days = st.sidebar.number_input(
    "Days Since Last Order",
    min_value=0,
    max_value=1000,
    value=30
)

num_orders = st.sidebar.number_input(
    "Total Number of Orders",
    min_value=1,
    max_value=100,
    value=5
)

total_spend = st.sidebar.number_input(
    "Total Customer Spend",
    min_value=0.0,
    max_value=100000.0,
    value=500.0
)

avg_order_value = st.sidebar.number_input(
    "Average Order Value",
    min_value=0.0,
    max_value=50000.0,
    value=100.0
)

avg_review_score = st.sidebar.slider(
    "Average Review Score",
    min_value=1.0,
    max_value=5.0,
    value=4.0
)

avg_delivery_delay = st.sidebar.number_input(
    "Average Delivery Delay in Days",
    min_value=0.0,
    max_value=50.0,
    value=3.0
)

customer_state = st.sidebar.selectbox(
    "Customer State",
    ["SP", "RJ", "MG", "RS", "PR"]
)

# =========================
# KPI INPUT PREVIEW
# =========================
st.subheader("📌 Customer Profile Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Recency Days", recency_days)
col2.metric("Orders", num_orders)
col3.metric("Total Spend", f"{total_spend:.2f}")
col4.metric("Review Score", avg_review_score)

# =========================
# BUILD INPUT DATA
# =========================
input_data = pd.DataFrame([{
    "recency_days": recency_days,
    "num_orders": num_orders,
    "total_spend": total_spend,
    "avg_order_value": avg_order_value,
    "avg_review_score": avg_review_score,
    "avg_delivery_delay": avg_delivery_delay,
    "customer_state": customer_state
}])

input_encoded = pd.get_dummies(input_data)

for col in feature_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[feature_cols]

# =========================
# PREDICTION
# =========================
st.markdown("---")
st.subheader("🔮 Churn Prediction Result")

if st.button("Predict Customer Churn"):

    try:
        churn_prob = model.predict_proba(input_encoded)[0][1]
        prediction = int(churn_prob >= 0.5)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")

        with col2:
            if prediction == 1:
                st.metric("Prediction", "High Churn Risk")
            else:
                st.metric("Prediction", "Active Customer")

        with col3:
            if churn_prob > 0.7:
                st.metric("Risk Level", "High")
            elif churn_prob > 0.4:
                st.metric("Risk Level", "Medium")
            else:
                st.metric("Risk Level", "Low")

        st.progress(float(churn_prob))

        # =========================
        # RISK MESSAGE
        # =========================
        if churn_prob > 0.7:
            st.error("🚨 This customer has a high chance of churn.")
            st.markdown("""
            **Recommended Actions:**
            - Offer discount or loyalty reward  
            - Send personalized email campaign  
            - Improve delivery experience  
            - Follow up on customer satisfaction  
            """)

        elif churn_prob > 0.4:
            st.warning("⚠️ This customer has a medium churn risk.")
            st.markdown("""
            **Recommended Actions:**
            - Monitor future orders  
            - Provide product recommendations  
            - Encourage repeat purchases  
            """)

        else:
            st.success("✅ This customer has a low churn risk.")
            st.markdown("""
            **Recommended Actions:**
            - Maintain engagement  
            - Recommend premium products  
            - Add customer to loyalty program  
            """)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# =========================
# BUSINESS INSIGHTS SECTION
# =========================
st.markdown("---")
st.subheader("📊 Business Value of This App")

st.markdown("""
This application can help an e-commerce business:

- Identify customers likely to stop purchasing
- Improve customer retention strategy
- Reduce revenue loss
- Support marketing campaigns
- Make data-driven business decisions
""")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built using Python, Streamlit, LightGBM, Pandas and Joblib")