import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title="🛒 Smart Shop Purchase Prediction Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

model = joblib.load("model/decision_tree.pkl")

tree = model.named_steps["model"]
feature_names = model.named_steps["preprocess"].get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": tree.feature_importances_
}).sort_values("Importance", ascending=True).tail(10)

# ... continue with Part 2

importance=importance.sort_values(
    "Importance",
    ascending=False
).head(10)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.main{
    background-color:#F7F9FB;
}

.block-container{
    padding-top:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

h1{
    color:#222;
    text-align:center;
    font-weight:700;
}

h3{
    color:#2E4053;
}

.stButton>button{
    width:100%;
    height:60px;
    border-radius:12px;
    border:none;
    background:linear-gradient(90deg,#4F8BF9,#3465D9);
    color:white;
    font-size:20px;
    font-weight:bold;
}

.stButton>button:hover{
    transform:scale(1.02);
}

[data-testid="stMetricValue"]{
    font-size:36px;
}

</style>
""", unsafe_allow_html=True)
# -------------------------
# Title
# -------------------------

st.title("🛒 Smart Shop Purchase Predictor")


st.caption(
    "Predict whether an online customer is likely to purchase based on browsing behaviour."
)
#kpi cards
kpi1,kpi2,kpi3=st.columns(3)

with kpi1:
    st.metric(
        "Model",
        "Decision Tree"
    )

with kpi2:
    st.metric(
        "Dataset Size",
        "12.3K"
    )

with kpi3:
    st.metric(
        "Purchase Rate",
        "15.5%"
    )


st.divider()

left, right = st.columns(2)

# ==========================
# LEFT COLUMN
# ==========================

with left:

    st.subheader("📊 Browsing Behaviour")

    administrative = st.number_input(
        "Administrative Pages",
        min_value=0,
        value=2
    )

    administrative_duration = st.number_input(
        "Administrative Duration",
        min_value=0.0,
        value=25.0
    )

    informational = st.number_input(
        "Informational Pages",
        min_value=0,
        value=0
    )

    informational_duration = st.number_input(
        "Informational Duration",
        min_value=0.0,
        value=0.0
    )

    product_related = st.number_input(
        "Product Related Pages",
        min_value=0,
        value=30
    )

    product_related_duration = st.number_input(
        "Product Related Duration",
        min_value=0.0,
        value=800.0
    )

    bounce = st.number_input(
        "Bounce Rate",
        min_value=0.0,
        value=0.02,
        format="%.4f"
    )

    exit_rate = st.number_input(
        "Exit Rate",
        min_value=0.0,
        value=0.05,
        format="%.4f"
    )

    page_values = st.number_input(
        "Page Values",
        min_value=0.0,
        value=0.0
    )

    special_day = st.slider(
        "Special Day",
        0.0,
        1.0,
        0.0,
        step=0.1
    )

# ==========================
# RIGHT COLUMN
# ==========================

with right:

    st.subheader("👤Visitor Information")

    operating_system = st.selectbox(
        "Operating System",
        [1,2,3,4,5,6,7,8]
    )

    browser = st.selectbox(
        "Browser",
        [1,2,3,4,5,6,7,8,9,10,11,12,13]
    )

    region = st.selectbox(
        "Region",
        [1,2,3,4,5,6,7,8,9]
    )

    traffic_type = st.selectbox(
        "Traffic Type",
        list(range(1,21))
    )

    visitor_type = st.selectbox(
        "Visitor Type",
        [0,1],
        format_func=lambda x: "Returning Visitor" if x==1 else "New Visitor"
    )

    weekend = st.selectbox(
        "Weekend Visit",
        [0,1],
        format_func=lambda x: "Yes" if x==1 else "No"
    )

    month = st.selectbox(
        "Month",
        [
            "Feb",
            "Mar",
            "May",
            "June",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec"
        ]
    )

# -------------------------
# Create Input DataFrame
# -------------------------

input_df = pd.DataFrame({

    "Administrative":[administrative],
    "Administrative_Duration":[administrative_duration],
    "Informational":[informational],
    "Informational_Duration":[informational_duration],
    "ProductRelated":[product_related],
    "ProductRelated_Duration":[product_related_duration],
    "BounceRates":[bounce],
    "ExitRates":[exit_rate],
    "PageValues":[page_values],
    "SpecialDay":[special_day],
    "OperatingSystems":[operating_system],
    "Browser":[browser],
    "Region":[region],
    "TrafficType":[traffic_type],
    "VisitorType":[visitor_type],
    "Weekend":[weekend],
    "Month":[month]

})

st.divider()

# -------------------------
# Prediction
# -------------------------

if st.button("🔮 Predict Purchase", use_container_width=True):

    st.subheader("📋 Input Summary")

    st.dataframe(
        input_df,
        use_container_width=True
    )

    with st.spinner("Running prediction..."):

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

    st.divider()

    pred_col, chart_col = st.columns([1,1.3])

    with pred_col:

        st.subheader("Prediction")

        st.metric(
            "Purchase Probability",
            f"{probability*100:.2f}%"
        )

        st.progress(float(probability))

        if probability >= 0.80:

            st.success("High Confidence")

        elif probability >= 0.60:

            st.info("Moderate Confidence")

        else:

            st.warning("Low Confidence")

        if prediction == 1:

            st.success("🟢 Likely to Purchase")

        else:

            st.error("🔴 Unlikely to Purchase")

    with chart_col:

        st.subheader("Top 10 Feature Importance")

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

st.divider()

with st.expander("About this Project"):

    st.markdown("""

### Smart Shop Purchase Prediction

This project predicts whether an online customer is likely to purchase.

### Machine Learning Pipeline

- Data Cleaning
- One Hot Encoding
- Standard Scaling
- Stratified Train-Test Split
- GridSearchCV
- Cross Validation
- Decision Tree Classifier

### Metrics

- Accuracy
- Precision
- Recall
- F1 Score


""")

st.caption(
    "Built with ❤️ using Streamlit • Scikit-learn • Plotly"
)