import joblib
import streamlit as st
import pandas as pd

model = joblib.load("battery_rul_model.pkl")

# Run: streamlit run app.py

st.set_page_config(
    page_title="Battery RUL Predictor",
    page_icon="📊",
    layout="centered"
)


# SPLASH SCREEN
def show_splash_screen():
    st.markdown(
        """
        <style>
        .splash-box {
            padding: 2.2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            border: 1px solid #e5e7eb;
            text-align: center;
            margin-top: 3rem;
        }
        .splash-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .splash-subtitle {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.2rem;
        }
        .feature-card {
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            background-color: white;
            height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="splash-box">
            <div class="splash-title">Battery RUL Predictor</div>
            <div class="splash-subtitle">
                Estimating Battery Remaining Useful Life Using Machine Learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Predicting", type="primary", use_container_width=True):
            st.session_state["show_app"] = True
            st.rerun()
            
    with col2:
        if st.button("ABOUT", type="secondary", use_container_width=True):
            st.session_state["show_about"] = not st.session_state["show_about"]
            st.rerun()

    if st.session_state["show_about"]:
        st.markdown("---")
        st.subheader("Abstract")
        st.markdown(
            """
            Battery degradation has become a crucial challenge to environmental safety and energy management systems due to the limited lifespan of lithium-ion batteries and their improper handling. This project aims to develop lightweight machine learning models to estimate the remaining useful life (RUL) of lithium-ion batteries, aligning with SDG 7 (Affordable and Clean Energy) and SDG 12 (Responsible Consumption and Production) to support responsible battery usage. We implement and compare five regression models to predict battery RUL, such as Gradient Boosting, XGBoost, Random Forest, Decision Tree, and Polynomial Regression. Among these, the Gradient Boosting model demonstrates the superior performance and is thus deployed in this predictor.
            """
        )
        st.write("")
        st.link_button(
            "🔗 View DOI", 
            "https://doi.org/10.1109/ICIMCIS68501.2025.11327063", 
            use_container_width=True
        )

if "show_app" not in st.session_state:
    st.session_state["show_app"] = False

if "show_about" not in st.session_state:
    st.session_state["show_about"] = False

if not st.session_state["show_app"]:
    show_splash_screen()
    st.stop()


# MAIN APP
st.set_page_config(page_title="Battery RUL Predictor", layout="centered")

st.markdown(
    """
    <style>
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Battery RUL Predictor")
st.markdown("Input the battery feature parameters below to estimate the Remaining Useful Life (RUL).")

with st.form("prediction_form"):
    discharge_time = st.number_input("Discharge Time (s)", min_value=0.0, value=None, placeholder="e.g., 6026.76")
    decrement_v = st.number_input("Decrement 3.6-3.4V (s)", min_value=0.0, value=None, placeholder="e.g., 1220.00")
    max_v_discharge = st.number_input("Max. Voltage Discharge (V)", min_value=0.0, value=None, placeholder="e.g., 4.25")
    min_v_charge = st.number_input("Min. Voltage Charge (V)", min_value=0.0, value=None, placeholder="e.g., 3.21")
    time_415v = st.number_input("Time at 4.15V (s)", min_value=0.0, value=None, placeholder="e.g., 5505.99")
    time_cc = st.number_input("Time Constant Current (s)", min_value=0.0, value=None, placeholder="e.g., 6161.00")
    charging_time = st.number_input("Charging Time (s)", min_value=0.0, value=None, placeholder="e.g., 10500.33")
    
    st.write("")
    
    col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
    with col_b2:
        submit_button = st.form_submit_button(label="Predict RUL", use_container_width=True)

if submit_button:
    input_fields = [discharge_time, decrement_v, max_v_discharge, min_v_charge, time_415v, time_cc, charging_time]
    
    if any(v is None for v in input_fields):
        st.error("All feature parameters must be filled out.")
    else:
        input_data = pd.DataFrame([[
            discharge_time, decrement_v, max_v_discharge, min_v_charge, 
            time_415v, time_cc, charging_time
        ]], columns=[
            "Discharge Time (s)", "Decrement 3.6-3.4V (s)", "Max. Voltage Discharge (V)", 
            "Min. Voltage Charge (V)", "Time at 4.15V (s)", "Time Constant Current (s)", "Charging Time (s)"
        ])
        
        try:
            model = joblib.load("battery_rul_model.pkl") 
            prediction = model.predict(input_data)
            
            st.markdown(f"""
                <div style="background-color: #d1e7dd; color: #0f5132; padding: 18px; border-radius: 12px; text-align: center; border: 1px solid #badbcc; font-weight: bold; font-size: 1.25rem; margin-top: 1.5rem;">
                    Predicted Remaining Useful Life (RUL): {int(round(prediction[0]))} cycles.
                </div>
            """, unsafe_allow_html=True)
            
        except FileNotFoundError:
            st.error("Pickle of best model not found.")

with st.expander("View Model Insights"):
    st.markdown("The contribution of each feature to the Gradient Boosting model's prediction decision:")
    
    importance_data = pd.DataFrame({
        "Feature": ["Discharge Time (s)", "Decrement 3.6-3.4V (s)", "Max. Voltage Discharge (V)", 
                    "Min. Voltage Charge (V)", "Time at 4.15V (s)", "Time Constant Current (s)", "Charging Time (s)"],
        "Importance": [0.3907, 0.0742, 0.0147, 0.0083, 0.3132, 0.1927, 0.0061]
    }).sort_values(by="Importance", ascending=True)
    
    st.bar_chart(data=importance_data, x="Feature", y="Importance", horizontal=True)

    st.divider()

    st.markdown("The scatter plot comparing the predicted versus actual Remaining Useful Life values:")
    
    try:
        st.image("results/scatterplot_gradient_boosting.png", caption="Scatter Plot of Gradient Boosting", use_container_width=True)
    except:
        st.warning("Scatter plot image file not found.")

    st.markdown(
        """
        * **$R^2$ Score ($\t{RUL} \le 200$):** `0.7706`
        * **$R^2$ Score ($\t{RUL} > 200$):** `0.9956`
        """
    )
