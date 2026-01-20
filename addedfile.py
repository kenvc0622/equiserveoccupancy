import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.optimize import fsolve
import pandas as pd

# Set page configuration FIRST
st.set_page_config(
    page_title="Call Center Occupancy Analysis",
    page_icon="📊",
    layout="wide"
)

# ========================
# CORE CALCULATION FUNCTIONS
# ========================

def erlang_c_probability_wait(N, A):
    """Calculate probability of wait using Erlang C formula"""
    if A >= N:
        return 1.0

    sum_term = 0
    for i in range(int(N)):
        sum_term += (A**i) / factorial(i)

    C = (A**N / factorial(N)) * (N / (N - A)) / (sum_term + (A**N / factorial(N)) * (N / (N - A)))
    return C

def calculate_service_level(N, A, AHT, target_time):
    """Calculate service level (% answered within target_time)"""
    if A >= N:
        return 0.0

    P_wait = erlang_c_probability_wait(N, A)
    SL = 1 - P_wait * np.exp(-(N - A) * target_time / AHT)
    return max(0, min(1, SL))

def calculate_occupancy(volume, AHT, headcount, interval_seconds):
    """Calculate occupancy percentage"""
    return (volume * AHT) / (headcount * interval_seconds)

# ========================
# MAIN APP
# ========================

def main():
    st.title("📞 Call Center Occupancy Analysis Tool")
    st.markdown("""
    This tool helps analyze and optimize the trade-off between agent occupancy and service level (SLA) 
    in call center operations using Erlang C calculations.
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dynamic Trade-off Curve Generator",
        "⚙️ Dynamic Optimization Tool", 
        "📐 Mathematical Curve Analysis",
        "🔍 Mathematical Analysis Results"
    ])
    
    with tab1:
        st.header("📈 DYNAMIC TRADE-OFF CURVE GENERATOR")
        
        col1, col2 = st.columns(2)
        
        with col1:
            volume = st.slider("Volume (calls/hr):", 1, 200, 45, 1, key="tradeoff_volume")
            AHT = st.slider("AHT (seconds):", 60, 1200, 390, 10, key="tradeoff_aht")
            ASA_target = st.slider("ASA Target (seconds):", 5, 180, 30, 5, key="tradeoff_asa")
        
        with col2:
            headcount = st.slider("Headcount:", 1.0, 50.0, 7.1, 0.5, key="tradeoff_hc")
            target_occ = st.slider("Target Occupancy:", 0.1, 1.0, 0.8, 0.01, format="%.0%%", key="tradeoff_occ")
            interval_minutes = st.selectbox("Interval (minutes):", [15, 30, 60], index=2, key="tradeoff_interval")
        
        # Convert interval to seconds
        interval_seconds = interval_minutes * 60
        
        # Calculate current metrics
        current_occ = calculate_occupancy(volume, AHT, headcount, interval_seconds)
        traffic_intensity = (volume * AHT/3600)
        current_sl = calculate_service_level(headcount, traffic_intensity, AHT, ASA_target)
        
        # Display current state
        st.subheader("📊 Current State")
        
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Occupancy", f"{current_occ*100:.1f}%")
            st.metric("Volume", f"{volume} calls/hr")
        with mcol2:
            st.metric("Service Level", f"{current_sl*100:.1f}%")
            st.metric("AHT", f"{AHT}s")
        with mcol3:
            st.metric("Headcount", f"{headcount:.1f}")
            st.metric("Traffic Intensity", f"{traffic_intensity:.2f} Erlangs")
        
        # Store in session state
        st.session_state.tradeoff_params = {
            'volume': volume,
            'AHT': AHT,
            'ASA_target': ASA_target,
            'headcount': headcount,
            'target_occ': target_occ,
            'interval_minutes': interval_minutes
        }
        
        # Simple plot example
        if st.button("Generate Simple Analysis", type="primary"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot 1: Occupancy vs Headcount
            hc_range = np.linspace(max(1, headcount * 0.5), headcount * 2, 50)
            occ_values = [calculate_occupancy(volume, AHT, hc, interval_seconds) * 100 for hc in hc_range]
            
            ax1.plot(hc_range, occ_values, 'b-', linewidth=2)
            ax1.axvline(x=headcount, color='r', linestyle='--', label=f'Current HC: {headcount}')
            ax1.set_xlabel('Headcount')
            ax1.set_ylabel('Occupancy (%)')
            ax1.set_title('Headcount vs Occupancy')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Service Level vs Headcount
            sl_values = [calculate_service_level(hc, traffic_intensity, AHT, ASA_target) * 100 for hc in hc_range]
            
            ax2.plot(hc_range, sl_values, 'g-', linewidth=2)
            ax2.axvline(x=headcount, color='r', linestyle='--', label=f'Current HC: {headcount}')
            ax2.axhline(y=90, color='darkgreen', linestyle=':', alpha=0.5, label='90% Target')
            ax2.set_xlabel('Headcount')
            ax2.set_ylabel('Service Level (%)')
            ax2.set_title('Headcount vs Service Level')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab2:
        st.header("⚙️ DYNAMIC OPTIMIZATION TOOL")
        
        col1, col2 = st.columns(2)
        
        with col1:
            opt_volume = st.slider("Volume:", 1, 200, 40, 1, key="opt_volume")
            opt_AHT = st.slider("AHT (s):", 60, 1200, 390, 10, key="opt_AHT")
            opt_ASA = st.slider("ASA (s):", 5, 180, 30, 5, key="opt_ASA")
        
        with col2:
            opt_target_sla = st.slider("Target SLA (%):", 50, 100, 90, 1, key="opt_target_sla")
            opt_target_occ = st.slider("Target Occupancy:", 0.1, 1.0, 0.8, 0.01, format="%.0%%", key="opt_target_occ")
            opt_interval = st.selectbox("Interval (min):", [15, 30, 60], index=2, key="opt_interval")
        
        if st.button("Run Optimization", type="primary"):
            st.info("Optimization would run here with the complete functions")
    
    with tab3:
        st.header("📐 MATHEMATICAL CURVE ANALYSIS")
        
        if 'tradeoff_params' in st.session_state:
            params = st.session_state.tradeoff_params
            default_volume = params['volume']
            default_AHT = params['AHT']
        else:
            default_volume = 40
            default_AHT = 390
        
        col1, col2 = st.columns(2)
        
        with col1:
            math_volume = st.number_input("Volume:", min_value=1, max_value=200, value=default_volume, key="math_volume")
            math_AHT = st.number_input("AHT (s):", min_value=60, max_value=1200, value=default_AHT, key="math_AHT")
        
        with col2:
            math_ASA = st.number_input("ASA (s):", min_value=5, max_value=180, value=30, key="math_ASA")
            math_target_occ = st.slider("Target Occupancy:", 0.1, 1.0, 0.8, 0.01, format="%.0%%", key="math_target_occ")
        
        if st.button("Run Mathematical Analysis", type="primary"):
            st.success("Mathematical analysis would display here")
    
    with tab4:
        st.header("🔍 MATHEMATICAL ANALYSIS RESULTS")
        
        if 'tradeoff_params' not in st.session_state:
            st.warning("Please run the Trade-off Analysis first to get parameters.")
        else:
            params = st.session_state.tradeoff_params
            st.write(f"**Current Parameters:**")
            st.write(f"- Volume: {params['volume']} calls/hr")
            st.write(f"- AHT: {params['AHT']} seconds")
            st.write(f"- ASA Target: {params['ASA_target']} seconds")
            st.write(f"- Headcount: {params['headcount']:.1f}")
            st.write(f"- Target Occupancy: {params['target_occ']*100:.0f}%")
            
            # Calculate some metrics
            interval_seconds = params['interval_minutes'] * 60
            occ = calculate_occupancy(params['volume'], params['AHT'], params['headcount'], interval_seconds)
            traffic_intensity = (params['volume'] * params['AHT']/3600)
            sl = calculate_service_level(params['headcount'], traffic_intensity, params['AHT'], params['ASA_target'])
            
            st.metric("Current Occupancy", f"{occ*100:.1f}%")
            st.metric("Current Service Level", f"{sl*100:.1f}%")
            st.metric("Traffic Intensity", f"{traffic_intensity:.2f} Erlangs")
    
    # Footer
    st.markdown("---")
    st.caption("Call Center Occupancy Analysis Tool v1.0 | Based on Erlang C Calculations")

# This is the critical part - make sure main() is called
if __name__ == "__main__":
    main()
