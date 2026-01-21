import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.optimize import minimize_scalar, fsolve
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set page configuration FIRST
st.set_page_config(
    page_title="Call Center Occupancy Analysis",
    page_icon="📊",
    layout="wide"
)

# ========================
# ENHANCED CORE CALCULATION FUNCTIONS
# ========================

def erlang_c_probability_wait(N, A):
    """Calculate probability of wait using Erlang C formula with improved numerical stability"""
    if A >= N:
        return 1.0
    
    try:
        sum_term = 0
        # Calculate sum more efficiently
        for i in range(int(N)):
            sum_term += (A**i) / factorial(i)
        
        numerator = (A**N / factorial(N)) * (N / (N - A))
        denominator = sum_term + numerator
        C = numerator / denominator if denominator != 0 else 1.0
        return min(1.0, max(0.0, C))
    except:
        return 1.0

def calculate_service_level(N, A, AHT, target_time):
    """Calculate service level (% answered within target_time) with bounds"""
    if A >= N or N <= 0 or A <= 0:
        return 0.0
    
    try:
        P_wait = erlang_c_probability_wait(N, A)
        exponent = -(N - A) * target_time / AHT
        SL = 1 - P_wait * np.exp(exponent if exponent < 100 else 100)
        return min(1.0, max(0.0, SL))
    except:
        return 0.0

def calculate_occupancy(volume, AHT, headcount, interval_seconds):
    """Calculate occupancy percentage with safety checks"""
    if headcount <= 0 or interval_seconds <= 0:
        return 0.0
    occupancy = (volume * AHT) / (headcount * interval_seconds)
    return min(1.0, max(0.0, occupancy))

def calculate_required_headcount(volume, AHT, target_occupancy, interval_seconds):
    """Calculate headcount needed to achieve target occupancy"""
    if target_occupancy <= 0:
        return float('inf')
    required = (volume * AHT) / (target_occupancy * interval_seconds)
    return max(1.0, required)

def optimize_headcount_for_sla(volume, AHT, target_sla_percent, ASA_target, interval_seconds=3600):
    """Find minimum headcount that meets SLA target"""
    traffic_intensity = (volume * AHT / 3600)
    
    def sla_objective(N):
        if N <= traffic_intensity:
            return 1000  # Penalty for insufficient headcount
        sla = calculate_service_level(N, traffic_intensity, AHT, ASA_target) * 100
        # Return negative of SLA (we want to maximize, but minimize negative)
        return -(sla - target_sla_percent)**2
    
    # Search for optimal headcount
    lower_bound = max(1, int(traffic_intensity) + 1)
    upper_bound = lower_bound + 50
    
    try:
        result = minimize_scalar(
            sla_objective,
            bounds=(lower_bound, upper_bound),
            method='bounded',
            options={'xatol': 0.1, 'maxiter': 100}
        )
        
        optimal_N = max(1, np.ceil(result.x))
        return optimal_N, calculate_service_level(optimal_N, traffic_intensity, AHT, ASA_target) * 100
    except:
        # Fallback: linear search
        for N in range(lower_bound, upper_bound + 1):
            sla = calculate_service_level(N, traffic_intensity, AHT, ASA_target) * 100
            if sla >= target_sla_percent:
                return N, sla
        return lower_bound, calculate_service_level(lower_bound, traffic_intensity, AHT, ASA_target) * 100

# ========================
# MAIN APP WITH FULL FUNCTIONALITY
# ========================

def main():
    st.title("📞 Equiserve Call Center Occupancy Analysis Tool")
    st.markdown("""
    This tool helps analyze and optimize the trade-off between agent occupancy and service level (SLA) 
    in call center operations using Erlang C calculations.
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dynamic Trade-off Analysis",
        "⚙️ Optimization Engine", 
        "📐 Mathematical Analysis",
        "📊 Results Dashboard"
    ])
    
    with tab1:
        st.header("📈 DYNAMIC TRADE-OFF ANALYSIS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            volume = st.slider("Volume (calls/hr):", 1, 500, 45, 1, key="tradeoff_volume")
            AHT = st.slider("AHT (seconds):", 60, 1200, 390, 10, key="tradeoff_aht")
            ASA_target = st.slider("ASA Target (seconds):", 5, 300, 30, 5, key="tradeoff_asa")
        
        with col2:
            headcount = st.slider("Headcount:", 1.0, 100.0, 7.1, 0.5, key="tradeoff_hc")
            target_occ = st.slider("Target Occupancy:", 0.1, 1.0, 0.8, 0.01, format="%.0f%%", key="tradeoff_occ")
            interval_minutes = st.selectbox("Interval (minutes):", [15, 30, 60], index=2, key="tradeoff_interval")
        
        # Convert interval to seconds
        interval_seconds = interval_minutes * 60
        
        # Calculate current metrics
        current_occ = calculate_occupancy(volume, AHT, headcount, interval_seconds)
        traffic_intensity = (volume * AHT / 3600)
        current_sl = calculate_service_level(headcount, traffic_intensity, AHT, ASA_target)
        
        # Display current state
        st.subheader("📊 Current Performance Metrics")
        
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Occupancy", f"{current_occ*100:.1f}%", 
                     delta=f"{(current_occ - target_occ)*100:+.1f}% vs target" if abs(current_occ - target_occ) > 0.01 else "On target")
            st.metric("Volume", f"{volume} calls/hr")
        with mcol2:
            st.metric("Service Level", f"{current_sl*100:.1f}%", 
                     delta="✓ ≥ 80%" if current_sl >= 0.8 else "⚠️ < 80%")
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
            'interval_minutes': interval_minutes,
            'current_occ': current_occ,
            'current_sl': current_sl
        }
        
        # Generate comprehensive analysis
        if st.button("Generate Comprehensive Analysis", type="primary", key="gen_analysis"):
            with st.spinner("Generating analysis..."):
                # Create figure with 3 subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                # Plot 1: Occupancy vs Headcount
                hc_range = np.linspace(max(1, headcount * 0.5), headcount * 2, 50)
                occ_values = [calculate_occupancy(volume, AHT, hc, interval_seconds) * 100 for hc in hc_range]
                
                axes[0].plot(hc_range, occ_values, 'b-', linewidth=2.5, label='Occupancy')
                axes[0].axvline(x=headcount, color='r', linestyle='--', linewidth=2, label=f'Current: {headcount:.1f}')
                axes[0].axhline(y=target_occ*100, color='orange', linestyle=':', linewidth=2, label=f'Target: {target_occ*100:.0f}%')
                axes[0].fill_between(hc_range, occ_values, target_occ*100, where=np.array(occ_values) >= target_occ*100, 
                                     alpha=0.2, color='green', label='Above Target')
                axes[0].set_xlabel('Headcount', fontsize=11, fontweight='bold')
                axes[0].set_ylabel('Occupancy (%)', fontsize=11, fontweight='bold')
                axes[0].set_title('Occupancy vs Headcount', fontsize=12, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(loc='best')
                
                # Plot 2: Service Level vs Headcount
                sl_values = [calculate_service_level(hc, traffic_intensity, AHT, ASA_target) * 100 for hc in hc_range]
                
                axes[1].plot(hc_range, sl_values, 'g-', linewidth=2.5, label='Service Level')
                axes[1].axvline(x=headcount, color='r', linestyle='--', linewidth=2, label=f'Current: {headcount:.1f}')
                axes[1].axhline(y=80, color='darkgreen', linestyle=':', linewidth=2, label='80% SLA Target')
                axes[1].fill_between(hc_range, sl_values, 80, where=np.array(sl_values) >= 80, 
                                     alpha=0.2, color='lightgreen', label='Above 80%')
                axes[1].set_xlabel('Headcount', fontsize=11, fontweight='bold')
                axes[1].set_ylabel('Service Level (%)', fontsize=11, fontweight='bold')
                axes[1].set_title('Service Level vs Headcount', fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(loc='best')
                
                # Plot 3: Trade-off Curve (Occupancy vs Service Level)
                axes[2].plot(occ_values, sl_values, 'purple', linewidth=2.5, label='Trade-off Curve')
                axes[2].scatter([current_occ*100], [current_sl*100], color='red', s=100, zorder=5, 
                               label=f'Current: ({current_occ*100:.1f}%, {current_sl*100:.1f}%)')
                
                # Add target zones
                axes[2].axvline(x=target_occ*100, color='orange', linestyle=':', alpha=0.7, label=f'Target Occ: {target_occ*100:.0f}%')
                axes[2].axhline(y=80, color='darkgreen', linestyle=':', alpha=0.7, label='80% SLA')
                
                # Shade optimal quadrant
                x_optimal = target_occ*100
                y_optimal = 80
                axes[2].fill_between([x_optimal, 100], [y_optimal, y_optimal], 100, alpha=0.1, color='green', label='Optimal Zone')
                
                axes[2].set_xlabel('Occupancy (%)', fontsize=11, fontweight='bold')
                axes[2].set_ylabel('Service Level (%)', fontsize=11, fontweight='bold')
                axes[2].set_title('Occupancy vs Service Level Trade-off', fontsize=12, fontweight='bold')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend(loc='best')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Data table
                st.subheader("📋 Detailed Analysis Data")
                analysis_df = pd.DataFrame({
                    'Headcount': hc_range,
                    'Occupancy (%)': occ_values,
                    'Service Level (%)': sl_values,
                    'Above SLA Target': [sl >= 80 for sl in sl_values],
                    'Above Occupancy Target': [occ >= target_occ*100 for occ in occ_values]
                })
                st.dataframe(analysis_df.style.format({
                    'Headcount': '{:.1f}',
                    'Occupancy (%)': '{:.1f}%',
                    'Service Level (%)': '{:.1f}%'
                }), use_container_width=True)
    
    with tab2:
        st.header("⚙️ OPTIMIZATION ENGINE")
        
        st.markdown("""
        ### Find optimal staffing levels balancing occupancy and service level targets.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            opt_volume = st.slider("Call Volume:", 1, 500, 40, 1, key="opt_volume")
            opt_AHT = st.slider("Average Handle Time (s):", 60, 1200, 390, 10, key="opt_AHT")
            opt_ASA = st.slider("ASA Target (s):", 5, 300, 30, 5, key="opt_ASA")
        
        with col2:
            opt_target_sla = st.slider("Target Service Level (%):", 50, 99, 90, 1, key="opt_target_sla")
            opt_target_occ = st.slider("Target Occupancy:", 0.1, 1.0, 0.8, 0.01, format="%.0f%%", key="opt_target_occ")
            opt_interval = st.selectbox("Interval Duration:", [15, 30, 60], index=2, key="opt_interval")
        
        if st.button("🚀 Run Optimization Analysis", type="primary", key="run_opt"):
            with st.spinner("Running optimization analysis..."):
                # Calculate traffic intensity
                traffic_intensity = (opt_volume * opt_AHT / 3600)
                interval_seconds = opt_interval * 60
                
                # Find optimal headcount for SLA
                optimal_hc_sla, achieved_sla = optimize_headcount_for_sla(
                    opt_volume, opt_AHT, opt_target_sla, opt_ASA, interval_seconds
                )
                
                # Calculate occupancy at optimal headcount for SLA
                occ_at_optimal_sla = calculate_occupancy(opt_volume, opt_AHT, optimal_hc_sla, interval_seconds)
                
                # Find headcount for target occupancy
                required_hc_occ = calculate_required_headcount(opt_volume, opt_AHT, opt_target_occ, interval_seconds)
                sl_at_target_occ = calculate_service_level(required_hc_occ, traffic_intensity, opt_AHT, opt_ASA) * 100
                
                # Find balanced solution (closest to both targets)
                def balanced_objective(N):
                    occ = calculate_occupancy(opt_volume, opt_AHT, N, interval_seconds)
                    sla = calculate_service_level(N, traffic_intensity, opt_AHT, opt_ASA) * 100
                    
                    # Weighted penalty function
                    occ_penalty = (occ - opt_target_occ) ** 2
                    sla_penalty = max(0, opt_target_sla - sla) ** 2 * 2  # Heavier penalty for missing SLA
                    
                    return occ_penalty + sla_penalty
                
                # Search for balanced solution
                lower_bound = max(1, int(traffic_intensity) + 1)
                upper_bound = max(lower_bound + 20, int(required_hc_occ * 1.5))
                
                try:
                    result = minimize_scalar(
                        balanced_objective,
                        bounds=(lower_bound, upper_bound),
                        method='bounded',
                        options={'xatol': 0.1}
                    )
                    balanced_hc = max(1, np.round(result.x))
                except:
                    balanced_hc = (optimal_hc_sla + required_hc_occ) / 2
                
                balanced_occ = calculate_occupancy(opt_volume, opt_AHT, balanced_hc, interval_seconds) * 100
                balanced_sla = calculate_service_level(balanced_hc, traffic_intensity, opt_AHT, opt_ASA) * 100
                
                # Display results
                st.subheader("🎯 Optimization Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("SLA-Optimized", 
                             f"{optimal_hc_sla:.1f} agents",
                             f"SLA: {achieved_sla:.1f}%",
                             delta_color="normal")
                    st.caption(f"Occupancy: {occ_at_optimal_sla*100:.1f}%")
                
                with col2:
                    st.metric("Occupancy-Targeted", 
                             f"{required_hc_occ:.1f} agents",
                             f"Occupancy: {opt_target_occ*100:.0f}%",
                             delta_color="normal")
                    st.caption(f"Service Level: {sl_at_target_occ:.1f}%")
                
                with col3:
                    st.metric("Balanced Solution", 
                             f"{balanced_hc:.1f} agents",
                             f"Occ: {balanced_occ:.1f}%, SLA: {balanced_sla:.1f}%",
                             delta_color="normal")
                    st.caption("Best trade-off")
                
                # Recommendations
                st.subheader("📋 Recommendations")
                
                if balanced_sla >= opt_target_sla and balanced_occ/100 >= opt_target_occ:
                    st.success(f"✅ **Recommended**: Use **{balanced_hc:.1f} agents** - achieves both targets (SLA: {balanced_sla:.1f}%, Occupancy: {balanced_occ:.1f}%)")
                elif achieved_sla >= opt_target_sla:
                    st.warning(f"⚠️ **Consider**: **{optimal_hc_sla:.1f} agents** - meets SLA target but occupancy is {occ_at_optimal_sla*100:.1f}%")
                else:
                    st.error(f"❌ **Challenge**: Cannot meet both targets simultaneously. Consider adjusting volume, AHT, or targets.")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Generate curve
                hc_range = np.linspace(max(1, traffic_intensity + 1), max(traffic_intensity * 3, 20), 100)
                occ_values = [calculate_occupancy(opt_volume, opt_AHT, hc, interval_seconds) * 100 for hc in hc_range]
                sla_values = [calculate_service_level(hc, traffic_intensity, opt_AHT, opt_ASA) * 100 for hc in hc_range]
                
                # Plot trade-off curve
                ax.plot(occ_values, sla_values, 'b-', linewidth=2, label='Trade-off Curve')
                
                # Plot optimal points
                ax.scatter([occ_at_optimal_sla*100], [achieved_sla], color='red', s=150, 
                          label=f'SLA-Optimized ({optimal_hc_sla:.1f} agents)', zorder=5)
                ax.scatter([opt_target_occ*100], [sl_at_target_occ], color='green', s=150,
                          label=f'Occ-Targeted ({required_hc_occ:.1f} agents)', zorder=5)
                ax.scatter([balanced_occ], [balanced_sla], color='purple', s=200, marker='*',
                          label=f'Balanced ({balanced_hc:.1f} agents)', zorder=6)
                
                # Add target lines
                ax.axvline(x=opt_target_occ*100, color='orange', linestyle='--', alpha=0.7, label=f'Target Occupancy')
                ax.axhline(y=opt_target_sla, color='darkgreen', linestyle='--', alpha=0.7, label=f'Target SLA')
                
                # Formatting
                ax.set_xlabel('Occupancy (%)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Service Level (%)', fontsize=12, fontweight='bold')
                ax.set_title('Optimization Analysis: Occupancy vs Service Level', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Store results
                st.session_state.optimization_results = {
                    'optimal_hc_sla': optimal_hc_sla,
                    'achieved_sla': achieved_sla,
                    'required_hc_occ': required_hc_occ,
                    'sl_at_target_occ': sl_at_target_occ,
                    'balanced_hc': balanced_hc,
                    'balanced_occ': balanced_occ,
                    'balanced_sla': balanced_sla
                }
    
    with tab3:
        st.header("📐 MATHEMATICAL ANALYSIS")
        
        # Use parameters from trade-off analysis if available
        if 'tradeoff_params' in st.session_state:
            params = st.session_state.tradeoff_params
            default_volume = params['volume']
            default_AHT = params['AHT']
        else:
            default_volume = 40
            default_AHT = 390
        
        col1, col2 = st.columns(2)
        
        with col1:
            math_volume = st.number_input("Call Volume:", min_value=1, max_value=500, value=default_volume, key="math_volume")
            math_AHT = st.number_input("AHT (seconds):", min_value=60, max_value=1200, value=default_AHT, key="math_AHT")
        
        with col2:
            math_ASA = st.number_input("ASA Target (seconds):", min_value=5, max_value=300, value=30, key="math_ASA")
            math_target_occ = st.slider("Occupancy Target:", 0.1, 1.0, 0.8, 0.01, format="%.0f%%", key="math_target_occ")
        
        if st.button("Run Mathematical Analysis", type="primary", key="run_math"):
            with st.spinner("Performing calculations..."):
                # Calculate Erlang C probabilities
                traffic_intensity = (math_volume * math_AHT / 3600)
                
                # Create analysis for different headcounts
                st.subheader("📊 Erlang C Probability Analysis")
                
                # Headcount range from below to above traffic intensity
                min_hc = max(1, int(traffic_intensity * 0.5))
                max_hc = int(traffic_intensity * 2) + 5
                headcounts = np.arange(min_hc, max_hc + 1)
                
                # Calculate probabilities
                p_wait_list = []
                sla_list = []
                occupancy_list = []
                
                for N in headcounts:
                    p_wait = erlang_c_probability_wait(N, traffic_intensity)
                    sla = calculate_service_level(N, traffic_intensity, math_AHT, math_ASA) * 100
                    occ = calculate_occupancy(math_volume, math_AHT, N, 3600) * 100  # 1-hour interval
                    
                    p_wait_list.append(p_wait * 100)
                    sla_list.append(sla)
                    occupancy_list.append(occ)
                
                # Create DataFrame
                analysis_df = pd.DataFrame({
                    'Headcount': headcounts,
                    'Traffic Intensity (Erlangs)': traffic_intensity,
                    'P(Wait) %': p_wait_list,
                    'Service Level %': sla_list,
                    'Occupancy %': occupancy_list,
                    'Utilization Ratio': [N/traffic_intensity if traffic_intensity > 0 else 0 for N in headcounts]
                })
                
                # Format the DataFrame
                styled_df = analysis_df.style.format({
                    'Headcount': '{:.0f}',
                    'Traffic Intensity (Erlangs)': '{:.3f}',
                    'P(Wait) %': '{:.2f}%',
                    'Service Level %': '{:.2f}%',
                    'Occupancy %': '{:.2f}%',
                    'Utilization Ratio': '{:.3f}'
                }).background_gradient(subset=['Service Level %'], cmap='RdYlGn')
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Mathematical insights
                st.subheader("🔬 Key Mathematical Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **Erlang C Formula:**
                    ```
                    P(wait) = (Aᴺ/N!) × (N/(N-A))
                           / Σ(Aⁱ/i!) + (Aᴺ/N!) × (N/(N-A))
                    ```
                    Where:
                    - A = Traffic Intensity = {traffic_intensity:.3f} Erlangs
                    - N = Number of agents
                    """)
                
                with col2:
                    st.info(f"""
                    **Service Level Formula:**
                    ```
                    SLA = 1 - P(wait) × exp(-(N-A) × T/AHT)
                    ```
                    Where:
                    - T = ASA Target = {math_ASA} seconds
                    - AHT = {math_AHT} seconds
                    - N-A = Agent surplus = N - {traffic_intensity:.2f}
                    """)
                
                # Visualize mathematical relationships
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Plot 1: P(Wait) vs Headcount
                axes[0,0].plot(headcounts, p_wait_list, 'r-', linewidth=2)
                axes[0,0].axvline(x=traffic_intensity, color='k', linestyle='--', alpha=0.5, label=f'A={traffic_intensity:.1f}')
                axes[0,0].set_xlabel('Headcount')
                axes[0,0].set_ylabel('P(Wait) %')
                axes[0,0].set_title('Probability of Waiting vs Headcount')
                axes[0,0].grid(True, alpha=0.3)
                axes[0,0].legend()
                
                # Plot 2: Service Level vs Headcount
                axes[0,1].plot(headcounts, sla_list, 'g-', linewidth=2)
                axes[0,1].axhline(y=80, color='darkgreen', linestyle=':', label='80% Target')
                axes[0,1].set_xlabel('Headcount')
                axes[0,1].set_ylabel('Service Level %')
                axes[0,1].set_title('Service Level vs Headcount')
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].legend()
                
                # Plot 3: Occupancy vs Headcount
                axes[1,0].plot(headcounts, occupancy_list, 'b-', linewidth=2)
                axes[1,0].axhline(y=math_target_occ*100, color='orange', linestyle=':', label=f'Target: {math_target_occ*100:.0f}%')
                axes[1,0].set_xlabel('Headcount')
                axes[1,0].set_ylabel('Occupancy %')
                axes[1,0].set_title('Occupancy vs Headcount')
                axes[1,0].grid(True, alpha=0.3)
                axes[1,0].legend()
                
                # Plot 4: All three together
                axes[1,1].plot(headcounts, p_wait_list, 'r-', label='P(Wait)')
                axes[1,1].plot(headcounts, sla_list, 'g-', label='Service Level')
                axes[1,1].plot(headcounts, occupancy_list, 'b-', label='Occupancy')
                axes[1,1].set_xlabel('Headcount')
                axes[1,1].set_ylabel('Percentage')
                axes[1,1].set_title('Combined View')
                axes[1,1].grid(True, alpha=0.3)
                axes[1,1].legend()
                
                plt.tight_layout()
                st.pyplot(fig)
    
    with tab4:
        st.header("📊 RESULTS DASHBOARD")
        
        # Check if we have analysis data
        if 'tradeoff_params' not in st.session_state:
            st.warning("Please run the Trade-off Analysis first to get parameters.")
        else:
            params = st.session_state.tradeoff_params
            
            st.subheader("Current Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Volume", f"{params['volume']} calls/hr")
                st.metric("AHT", f"{params['AHT']} seconds")
                st.metric("ASA Target", f"{params['ASA_target']} seconds")
            
            with col2:
                st.metric("Headcount", f"{params['headcount']:.1f}")
                st.metric("Target Occupancy", f"{params['target_occ']*100:.0f}%")
                st.metric("Interval", f"{params['interval_minutes']} minutes")
            
            # Performance summary
            st.subheader("Performance Summary")
            
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            
            with perf_col1:
                occ_gap = params['current_occ'] - params['target_occ']
                st.metric("Occupancy Gap", 
                         f"{occ_gap*100:+.1f}%",
                         "Above target" if occ_gap > 0 else "Below target")
            
            with perf_col2:
                sla_status = "✓ Good" if params['current_sl'] >= 0.8 else "⚠️ Needs attention"
                st.metric("SLA Status", sla_status)
            
            with perf_col3:
                traffic_intensity = (params['volume'] * params['AHT']/3600)
                agent_surplus = params['headcount'] - traffic_intensity
                st.metric("Agent Surplus", f"{agent_surplus:.2f}")
            
            with perf_col4:
                efficiency_score = (params['current_occ'] * params['current_sl']) * 100
                st.metric("Efficiency Score", f"{efficiency_score:.1f}/100")
            
            # Recommendations based on current state
            st.subheader("📋 Actionable Recommendations")
            
            if params['current_sl'] < 0.8 and params['current_occ'] > 0.85:
                st.error("""
                **❌ CRITICAL ISSUE**: High occupancy but low service level.
                **Action**: Increase headcount immediately to improve service level.
                """)
            elif params['current_sl'] >= 0.9 and params['current_occ'] < 0.7:
                st.success("""
                **✅ EXCELLENT**: High service level with comfortable occupancy.
                **Action**: Consider slight volume increase or cross-training opportunities.
                """)
            elif params['current_sl'] >= 0.8 and params['current_occ'] >= params['target_occ']:
                st.success("""
                **✅ ON TARGET**: Meeting both occupancy and SLA targets.
                **Action**: Maintain current staffing levels.
                """)
            else:
                st.warning("""
                **⚠️ SUBOPTIMAL**: Room for improvement in either occupancy or service level.
                **Action**: Use the Optimization Engine to find better staffing levels.
                """)
            
            # Show optimization results if available
            if 'optimization_results' in st.session_state:
                st.subheader("Optimization Results")
                opt_results = st.session_state.optimization_results
                
                opt_df = pd.DataFrame([
                    {
                        'Strategy': 'SLA-Optimized',
                        'Agents': opt_results['optimal_hc_sla'],
                        'Service Level': f"{opt_results['achieved_sla']:.1f}%",
                        'Occupancy': f"{opt_results['optimal_hc_sla']:.1f}%"
                    },
                    {
                        'Strategy': 'Occupancy-Targeted',
                        'Agents': opt_results['required_hc_occ'],
                        'Service Level': f"{opt_results['sl_at_target_occ']:.1f}%",
                        'Occupancy': f"{params['target_occ']*100:.0f}%"
                    },
                    {
                        'Strategy': 'Balanced',
                        'Agents': opt_results['balanced_hc'],
                        'Service Level': f"{opt_results['balanced_sla']:.1f}%",
                        'Occupancy': f"{opt_results['balanced_occ']:.1f}%"
                    }
                ])
                
                st.dataframe(opt_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Call Center Occupancy Analysis Tool v2.0 | Based on Erlang C Queueing Theory</p>
        <p><small>Note: Results are estimates based on mathematical models. Real-world factors may vary.</small></p>
    </div>
    """, unsafe_allow_html=True)

# This is the critical part - make sure main() is called
if __name__ == "__main__":
    main()
