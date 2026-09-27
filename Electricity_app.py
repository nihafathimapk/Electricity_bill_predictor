import streamlit as st

# Import numpy for numerical calculations
import numpy as np

# Import pandas for building the appliance-wise breakdown table/chart
import pandas as pd

# Import joblib for loading the saved model and scaler
import joblib


# ======================================================
# PAGE SETTINGS
# ======================================================

st.set_page_config(
    page_title="Electricity Consumption Predictor",
    page_icon="⚡",
    layout="centered"
)

# Small style tweaks so metrics/cards feel a bit more "app-like"
st.markdown(
    """
    <style>
        div[data-testid="stMetric"] {
            background-color: rgba(250, 250, 250, 0.05);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            padding: 12px 16px;
        }
        .tip-box {
            background-color: rgba(46, 204, 113, 0.12);
            border-left: 5px solid #2ecc71;
            border-radius: 8px;
            padding: 14px 16px;
            margin-top: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# TITLE
# ======================================================

st.title("⚡ Electricity Consumption Predictor")

st.write(
    "Tell us a bit about how your household uses electricity, and we'll "
    "estimate your **expected monthly usage**, your **estimated bill**, "
    "and a quick tip on how to bring both down."
)


# ======================================================
# LOAD MODEL AND SCALER
# ======================================================

# The trained Linear Regression model predicts instantaneous household
# power draw (kW) from electrical readings. We reuse it here to estimate
# the "background" load of a home (lighting, chargers, kitchen appliances,
# etc.) that isn't tied to one single named appliance below.
model = joblib.load("electricity_model.pkl")
scaler = joblib.load("electricity_scaler.pkl")


# ======================================================
# SIDEBAR — APPLIANCE POWER ASSUMPTIONS (ADVANCED / OPTIONAL)
# ======================================================

with st.sidebar:

    st.header("⚙️ Assumptions")

    st.caption(
        "Typical appliance power ratings used for the estimate. "
        "Adjust these if you know your own appliances' ratings."
    )

    ac_kw = st.slider("AC power rating (kW)", 0.8, 3.0, 1.5, 0.1)
    fan_kw = st.slider("Fan power, per fan (kW)", 0.03, 0.12, 0.07, 0.01)
    fridge_kwh_day = st.slider("Refrigerator (kWh/day)", 0.8, 2.5, 1.2, 0.1)
    tv_kw = st.slider("TV power (kW)", 0.05, 0.30, 0.12, 0.01)
    wm_kwh_load = st.slider("Washing machine (kWh/load)", 0.3, 1.2, 0.5, 0.05)

    st.divider()

    tariff = st.number_input(
        "Electricity Tariff (₹ per kWh)",
        min_value=0.0,
        value=7.0
    )

    co2_factor = st.slider(
        "Grid emission factor (kg CO₂ per kWh)",
        0.3, 1.0, 0.82, 0.01,
        help="Used only to show the rough environmental impact of your usage."
    )


# ======================================================
# HOUSEHOLD INPUT SECTION
# ======================================================

st.subheader("🏠 Your Household")

col1, col2 = st.columns(2)

with col1:

    ac_hours = st.slider(
        "❄️ AC usage (hours/day)",
        min_value=0, max_value=24, value=4
    )

    fan_hours = st.slider(
        "🌀 Fan usage (hours/day)",
        min_value=0, max_value=24, value=8
    )

    fridge = st.radio(
        "🧊 Refrigerator",
        ["Yes", "No"],
        horizontal=True
    )

    tv_hours = st.slider(
        "📺 TV usage (hours/day)",
        min_value=0, max_value=12, value=3
    )

with col2:

    wm_loads = st.slider(
        "🧺 Washing machine (loads/week)",
        min_value=0, max_value=14, value=4
    )

    people = st.number_input(
        "👨‍👩‍👧‍👦 Number of people in household",
        min_value=1, max_value=20, value=4
    )

    prev_consumption = st.number_input(
        "📄 Previous month's consumption (kWh)",
        min_value=0.0, value=0.0,
        help="Optional. Leave as 0 if you don't know it. "
             "Used only to compare against this month's estimate."
    )


# Number of fans is not asked directly — estimated from household size,
# but shown here so the assumption is transparent.
fans_count = max(1, round(people / 2))
st.caption(f"Assuming **{fans_count} fan(s)** running for a household of {people}.")


predict_button = st.button("⚡ Predict My Electricity Usage", use_container_width=True)


# ======================================================
# PREDICTION
# ======================================================

if predict_button:

    # --------------------------------------------------
    # APPLIANCE-WISE DAILY CONSUMPTION (kWh/day)
    # --------------------------------------------------

    ac_daily = ac_kw * ac_hours
    fan_daily = fan_kw * fan_hours * fans_count
    fridge_daily = fridge_kwh_day if fridge == "Yes" else 0.0
    tv_daily = tv_kw * tv_hours
    wm_daily = (wm_kwh_load * wm_loads) / 7

    # --------------------------------------------------
    # "OTHER HOUSEHOLD LOAD" — predicted using the trained model
    # --------------------------------------------------

    # Build proxy electrical readings from the household inputs so the
    # trained model can estimate the background power draw (lighting,
    # kitchen sockets, chargers, etc.) that isn't covered by the named
    # appliances above.
    proxy_input = np.array([[
        0.1,                                    # Global Reactive Power (typical)
        230.0,                                  # Voltage (typical)
        4.0 + 0.5 * people,                      # Global Intensity (scales with people)
        1.0 + 0.3 * people,                      # Sub Metering 1 (kitchen)
        (1.0 if fridge == "Yes" else 0.0) + 0.5 * wm_loads,  # Sub Metering 2 (laundry)
        0.5 * ac_hours                           # Sub Metering 3 (AC / water heater)
    ]])

    proxy_scaled = scaler.transform(proxy_input)
    other_power_kw = model.predict(proxy_scaled)[0]

    # Prevent negative prediction
    if other_power_kw < 0:
        other_power_kw = 0

    # Treated as a background load running for most of the day
    other_daily = other_power_kw * 24

    # --------------------------------------------------
    # TOTALS
    # --------------------------------------------------

    total_daily = ac_daily + fan_daily + fridge_daily + tv_daily + wm_daily + other_daily
    total_monthly = total_daily * 30
    monthly_bill = total_monthly * tariff
    co2_kg = total_monthly * co2_factor

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.subheader("🔮 Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Expected Monthly Usage", f"{total_monthly:.0f} kWh")

    with result_col2:
        st.metric("Estimated Monthly Bill", f"₹ {monthly_bill:,.0f}")

    with result_col3:
        if prev_consumption > 0:
            delta = total_monthly - prev_consumption
            st.metric(
                "vs Previous Month",
                f"{prev_consumption:.0f} kWh",
                delta=f"{delta:+.0f} kWh"
            )
        else:
            st.metric("Estimated CO₂ Impact", f"{co2_kg:.0f} kg")

    # Consumption level message
    if total_monthly < 150:
        st.info("Your estimated electricity consumption is relatively **low**.")
    elif total_monthly < 300:
        st.info("Your estimated electricity consumption is **moderate**.")
    else:
        st.warning("Your estimated electricity consumption is relatively **high**.")

    # --------------------------------------------------
    # APPLIANCE-WISE BREAKDOWN
    # --------------------------------------------------

    st.subheader("📊 Where Your Usage Comes From")

    breakdown = pd.DataFrame({
        "Appliance": ["AC", "Fans", "Refrigerator", "TV", "Washing Machine", "Other (lights, plugs, etc.)"],
        "Monthly kWh": [
            ac_daily * 30,
            fan_daily * 30,
            fridge_daily * 30,
            tv_daily * 30,
            wm_daily * 30,
            other_daily * 30
        ]
    }).set_index("Appliance")

    st.bar_chart(breakdown)

    # --------------------------------------------------
    # ENERGY-SAVING TIP (PRACTICAL ENVIRONMENTAL ANGLE)
    # --------------------------------------------------

    st.subheader("💡 Smart Tip")

    if ac_hours > 0:
        reduced_ac_daily = ac_kw * (ac_hours - 1)
        reduced_total_daily = reduced_ac_daily + fan_daily + fridge_daily + tv_daily + wm_daily + other_daily
        reduced_monthly = reduced_total_daily * 30
        saved_monthly_kwh = total_monthly - reduced_monthly
        saved_monthly_cost = saved_monthly_kwh * tariff
        saved_co2 = saved_monthly_kwh * co2_factor

        st.markdown(
            f"""
            <div class="tip-box">
            💡 Reducing AC usage by <b>1 hour/day</b> could lower your estimated
            consumption to about <b>{reduced_monthly:.0f} kWh/month</b> —
            saving roughly <b>{saved_monthly_kwh:.0f} kWh</b>,
            <b>₹{saved_monthly_cost:,.0f}</b>, and about
            <b>{saved_co2:.0f} kg of CO₂</b> every month.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="tip-box">
            💡 You're already using 0 AC hours/day — try switching off one more
            fan or reducing washing machine loads by one per week for extra savings.
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------
    # GENERAL ENERGY SAVING TIPS
    # --------------------------------------------------

    with st.expander("🌱 More ways to save energy"):
        st.write("• Set the AC temperature to 24°C or higher.")
        st.write("• Switch off electrical appliances when not in use.")
        st.write("• Use LED bulbs instead of traditional bulbs.")
        st.write("• Avoid keeping appliances on standby.")
        st.write("• Wash full loads of laundry instead of small, frequent ones.")
        st.write("• Clean refrigerator coils regularly to keep it efficient.")

    # --------------------------------------------------
    # HOW THIS IS CALCULATED (TRANSPARENCY)
    # --------------------------------------------------

    with st.expander("ℹ️ How this estimate is calculated"):
        st.write(
            "- **AC, Fans, TV, Washing machine**: calculated from the hours/loads "
            "you entered × typical appliance power ratings (adjustable in the sidebar)."
        )
        st.write(
            "- **Refrigerator**: assumed to run continuously, using a typical "
            "daily energy figure (adjustable in the sidebar)."
        )
        st.write(
            "- **Other (lights, plugs, kitchen, etc.)**: predicted by the trained "
            "Linear Regression model from the notebook, based on household size "
            "and appliance usage patterns."
        )
        st.write(
            "- **Bill** = Expected Monthly Usage × Electricity Tariff."
        )
        st.write(
            "- **CO₂ impact** = Expected Monthly Usage × Grid Emission Factor."
        )
        st.caption(
            "This is an estimate meant for awareness and comparison, not a "
            "substitute for your actual electricity meter reading."
        )
