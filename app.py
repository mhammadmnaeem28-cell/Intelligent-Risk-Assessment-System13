import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Dual axis ke liye zaroori hai
from datetime import datetime

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Risk System" ,layout="wide")
st.title("🏦 Intelligent Loan Risk Assessment System")
st.markdown("---")

# ─── Load Model & Data ────────────────────────────────────────────────────────
# Note: Make sure 'loan_model.pkl' and 'loan_data.csv.csv' are in the same folder
data_pack = pickle.load(open('loan_model.pkl', 'rb'))
model     = data_pack['model']
accuracy  = data_pack['acc']
df        = pd.read_csv('loan_data.csv.csv')

# ─── Top Metrics ──────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("System Accuracy", f"{accuracy*100:.1f}%")
c2.metric("Project Status",  "Online")
c3.metric("Dataset Size",    f"{len(df)} Rows")
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
sb = st.sidebar

sb.header("📝 Registration Details")
cust_name   = sb.text_input("Full Name",    "Muhammad Hammad")
cust_id     = sb.text_input("Customer ID",  "ID-2024-045")
city        = sb.text_input("City",         "Karachi")

sb.header("💰 Financial Inputs")
age       = sb.number_input("Age",                         18,    100,    28)
income    = sb.number_input("Annual Income ($)",        1000, 1000000, 65000)
emp_len   = sb.slider      ("Work Experience (Years)",     0,    40,     6)
loan_amt  = sb.number_input("Loan Amount Requested ($)",  500,  50000, 15000)
int_rate  = sb.slider      ("Interest Rate (%)",          5.0,  25.0,  11.5)
home      = sb.selectbox   ("Home Ownership",  ["RENT", "OWN", "MORTGAGE"])
intent    = sb.selectbox   ("Loan Purpose",    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION"])
grade     = sb.selectbox   ("Loan Grade",      ["A", "B", "C", "D"])
history   = sb.slider      ("Credit History (Years)",      1,    30,     5)
default   = sb.radio       ("Previous Default?",           ["N", "Y"])

# ─── Prediction ───────────────────────────────────────────────────────────────
if sb.button("Generate Assessment Report"):

    # Model input banana
    input_df = pd.DataFrame(
        [[age, income, home, emp_len, intent, grade,
          loan_amt, int_rate, loan_amt / income, default, history]],
        columns=model.feature_names_in_
    )

    # Default probability
    prob = model.predict_proba(input_df)[0][1]

    # Risk level decide karna
    if prob < 0.30:
        risk, status, show = "LOW",     "Highly Recommended",    st.success
    elif prob < 0.65:
        risk, status, show = "MEDIUM", "Conditional Approval",  st.warning
    else:
        risk, status, show = "HIGH",   "Rejected / High Risk",  st.error

    # Result dikhana
    st.subheader("📋 Loan Assessment Result")
    icons = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "❌"}
    show(f"{icons[risk]} Loan {risk} RISK  |  {status}  |  Risk Score: {prob*100:.1f}%")

st.divider()

# ─── Charts ───────────────────────────────────────────────────────────────────
st.subheader("📊 System Insights (Real Data Analysis)")

col_a, col_b = st.columns(2)

# Chart 1 – Loan Status by Intent
with col_a:
    fig1 = px.histogram(df, x="loan_intent", color="loan_status",
                        barmode="group", title="Loan Status by Intent")
    st.plotly_chart(fig1, use_container_width=True)

# Chart 2 – Age vs Income Scatter
with col_b:
    fig2 = px.scatter(df.sample(800, random_state=42),
                      x="person_age", y="person_income",
                      color="loan_status", title="Age vs Income Risk Scatter")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("📈 Additional Data Insights")

col_c, col_d = st.columns(2)

# Chart 3 – Home Ownership Pie
with col_c:
    fig3 = px.pie(df, names="person_home_ownership",
                  title="Home Ownership Distribution", hole=0.3)
    st.plotly_chart(fig3, use_container_width=True)

# Chart 4 – Avg Loan by Grade
with col_d:
    avg_grade = (df.groupby("loan_grade")["loan_amnt"]
                   .mean().reset_index()
                   .rename(columns={"loan_amnt": "avg_loan_amount"}))
    fig4 = px.bar(avg_grade, x="loan_grade", y="avg_loan_amount",
                  color="loan_grade", text_auto=".2s",
                  title="Average Loan Amount by Grade")
    st.plotly_chart(fig4, use_container_width=True)

# ─── Chart 5 – FINAL UPDATED MERGED GRAPH (SIMPLE & EASY) ─────────────────────
st.subheader("🔀 Merged Graph: Avg Loan Amount vs Interest Rate")

# Step 1: Data ko group karna (short variables)
g_data = df.groupby("loan_grade").agg(
    amt=("loan_amnt", "mean"), 
    rate=("loan_int_rate", "mean")
).reset_index()

# Step 2: Dual Y-Axis figure setup
fig5 = make_subplots(specs=[[{"secondary_y": True}]])

# Step 3: Bar Chart (Left Side - Paison ke liye)
fig5.add_trace(
    go.Bar(x=g_data["loan_grade"], y=g_data["amt"], name="Amount ($)", marker_color="#636EFA"),
    secondary_y=False
)

# Step 4: Line Chart (Right Side - Percentage ke liye)
fig5.add_trace(
    go.Scatter(x=g_data["loan_grade"], y=g_data["rate"], name="Rate (%)", 
               mode='lines+markers', line=dict(color="#EF553B", width=3)),
    secondary_y=True
)

# Step 5: Simple Layout
fig5.update_layout(
    title="Comparison: Loan Amount vs Interest Rate by Grade",
    xaxis_title="Loan Grade",
    height=500,
    legend=dict(orientation="h", y=1.1)
)

# Axis labels set karna taaki sab ko samajh aaye
fig5.update_yaxes(title_text="Avg Loan Amount ($)", secondary_y=False)
fig5.update_yaxes(title_text="Avg Interest Rate (%)", secondary_y=True)

st.plotly_chart(fig5, use_container_width=True)