import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.optimize import fsolve
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Call Center Occupancy Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CORE CALCULATION FUNCTIONS (same as original)
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

def calculate_callsmaxocc(headcount, AHT, target_occ, interval_seconds):
    """Calculate maximum calls for target occupancy"""
    return (target_occ * headcount * interval_seconds) / AHT

def calculate_maxcap_per_agent(AHT, interval_seconds, target_occ=0.8):
    """Calculate productive capacity per agent (calls/hour)"""
    return (target_occ * interval_seconds) / AHT

def calculate_required_hc_for_occupancy(volume, AHT, target_occ, interval_seconds):
    """Calculate headcount needed to achieve target occupancy"""
    if target_occ <= 0:
        return float('inf')
    return (volume * AHT) / (target_occ * interval_seconds)

def sla_function(N, A, AHT, ASA):
    """Calculate service level using Erlang C formula"""
    if A >= N:
        return 0.0

    # Calculate probability of waiting (Erlang C)
    sum_term = 0
    for i in range(int(N)):
        sum_term += (A**i) / factorial(i)

    P_wait = (A**N / factorial(N)) * (N / (N - A)) / (sum_term + (A**N / factorial(N)) * (N / (N - A)))

    # Calculate service level
    SL = 1 - P_wait * np.exp(-(N - A) * ASA / AHT)
    return max(0, min(1, SL))

# ========================
# TAB 1: DYNAMIC TRADE-OFF CURVE GENERATOR
# ========================

def tradeoff_tab():
    st.header("DYNAMIC TRADE-OFF CURVE GENERATOR")
    st.markdown("""
    Adjust sliders to simulate different scenarios:
    - **Volume:** Calls arriving per hour
    - **AHT:** Average Handle Time (seconds)
    - **ASA Target:** Acceptable wait time for service level
    - **Headcount:** Number of agents
    - **Target Occupancy:** Desired utilization level
    - **Interval:** Time period for calculations
    """)
    
    # Create columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        volume = st.slider("Volume (calls/hr):", 1, 200, 45, 1)
        AHT = st.slider("AHT (seconds):", 60, 1200, 390, 10)
        ASA_target = st.slider("ASA Target (seconds):", 5, 180, 30, 5)
    
    with col2:
        headcount = st.slider("Headcount:", 1.0, 50.0, 7.1, 0.5)
        target_occ = st.slider("Target Occupancy:", 0.1, 1.0, 0.8, 0.01, format="%.0%%")
        interval_minutes = st.selectbox("Interval (minutes):", [15, 30, 60], index=2)
    
    # Convert interval to seconds
    interval_seconds = interval_minutes * 60
    
    # Calculate current metrics
    current_occ = calculate_occupancy(volume, AHT, headcount, interval_seconds)
    traffic_intensity = (volume * AHT/3600)
    current_sl = calculate_service_level(headcount, traffic_intensity, AHT, ASA_target)
    callsmaxocc_current = calculate_callsmaxocc(headcount, AHT, target_occ, interval_seconds)
    maxcap_per_agent = calculate_maxcap_per_agent(AHT, interval_seconds, target_occ)
    total_capacity = maxcap_per_agent * headcount
    
    # Display current state
    st.subheader("Current State")
    st.info(f"""
    **Volume:** {volume} calls/hr | **AHT:** {AHT}s | **HC:** {headcount:.1f}
    **Occupancy:** {current_occ*100:.1f}% | **SLA:** {current_sl*100:.1f}%
    **CallsMaxOcc:** {callsmaxocc_current:.1f} calls/hr
    **MaxCap per agent:** {maxcap_per_agent:.2f} calls/hr
    **Total capacity:** {total_capacity:.1f} calls/hr
    """)
    
    # Store values in session state for other tabs
    st.session_state.tradeoff_params = {
        'volume': volume,
        'AHT': AHT,
        'ASA_target': ASA_target,
        'headcount': headcount,
        'target_occ': target_occ,
        'interval_minutes': interval_minutes
    }
    
    # Generate plots
    if st.button("Generate Trade-off Analysis", type="primary"):
        plot_tradeoff_curve(volume, AHT, ASA_target, headcount, target_occ, interval_minutes)

def plot_tradeoff_curve(volume, AHT, ASA_target, headcount, target_occ, interval_minutes):
    """Plot all trade-off curves (adapted from original)"""
    interval_seconds = interval_minutes * 60
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # SUBPLOT 1: Occupancy vs Headcount Trade-off
    ax1 = plt.subplot(2, 3, 1)
    hc_range = np.linspace(max(1, headcount * 0.5), headcount * 2, 50)
    occ_values = []
    sl_values = []
    
    for hc in hc_range:
        occ = calculate_occupancy(volume, AHT, hc, interval_seconds)
        occ_values.append(occ)
        traffic_intensity = (volume * AHT/3600)
        sl = calculate_service_level(hc, traffic_intensity, AHT, ASA_target)
        sl_values.append(sl * 100)
    
    ax1.plot(hc_range, np.array(occ_values) * 100, 'b-', linewidth=2, label='Occupancy')
    ax1.axhline(y=target_occ * 100, color='r', linestyle='--', alpha=0.5, label=f'Target ({target_occ*100:.0f}%)')
    ax1.axvline(x=headcount, color='g', linestyle='--', alpha=0.5, label=f'Current HC')
    ax1.fill_between(hc_range, 0, target_occ * 100, alpha=0.1, color='red')
    ax1.set_xlabel('Headcount')
    ax1.set_ylabel('Occupancy (%)')
    ax1.set_title('Headcount vs Occupancy')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_ylim([0, 110])
    current_occ = calculate_occupancy(volume, AHT, headcount, interval_seconds)
    ax1.plot(headcount, current_occ * 100, 'ro', markersize=10)
    
    # SUBPLOT 2: Service Level vs Headcount
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(hc_range, sl_values, 'g-', linewidth=2, label='Service Level')
    ax2.axhline(y=90, color='darkgreen', linestyle='--', alpha=0.5, label='90% SLA Target')
    ax2.axvline(x=headcount, color='g', linestyle='--', alpha=0.5)
    ax2.fill_between(hc_range, 90, 100, alpha=0.1, color='green')
    ax2.set_xlabel('Headcount')
    ax2.set_ylabel('Service Level (%)')
    ax2.set_title('Headcount vs Service Level')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_ylim([0, 105])
    current_sl = calculate_service_level(headcount, traffic_intensity, AHT, ASA_target)
    ax2.plot(headcount, current_sl * 100, 'go', markersize=10)
    
    # SUBPLOT 3: Occupancy vs Service Level Trade-off Curve
    ax3 = plt.subplot(2, 3, 3)
    tradeoff_occ = []
    tradeoff_sl = []
    
    for hc in np.linspace(max(1, headcount * 0.3), headcount * 3, 100):
        occ = calculate_occupancy(volume, AHT, hc, interval_seconds)
        traffic_intensity = (volume * AHT/3600)
        sl = calculate_service_level(hc, traffic_intensity, AHT, ASA_target)
        if occ <= 1.0:
            tradeoff_occ.append(occ * 100)
            tradeoff_sl.append(sl * 100)
    
    scatter = ax3.scatter(tradeoff_occ, tradeoff_sl, c=np.linspace(0.3, 3, len(tradeoff_occ)),
                         cmap='viridis', s=50, alpha=0.7, edgecolors='black')
    ax3.plot(current_occ * 100, current_sl * 100, 'ro', markersize=12, label='Current Position')
    ax3.axvline(x=target_occ * 100, color='red', linestyle='--', alpha=0.5, label=f'Target Occ ({target_occ*100:.0f}%)')
    ax3.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% SLA')
    ax3.fill_betweenx([85, 100], 70, 85, alpha=0.1, color='gold', label='Optimal Zone')
    ax3.set_xlabel('Occupancy (%)')
    ax3.set_ylabel('Service Level (%)')
    ax3.set_title('Occupancy vs Service Level Trade-off')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower left', fontsize=9)
    ax3.set_xlim([0, 105])
    ax3.set_ylim([0, 105])
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Headcount Ratio')
    
    # SUBPLOT 4: Capacity Analysis
    ax4 = plt.subplot(2, 3, 4)
    callsmaxocc_current = calculate_callsmaxocc(headcount, AHT, target_occ, interval_seconds)
    maxcap_per_agent = calculate_maxcap_per_agent(AHT, interval_seconds, target_occ)
    total_capacity = maxcap_per_agent * headcount
    
    metrics = ['Volume', 'CallsMaxOcc', 'MaxCap (total)']
    values = [volume, callsmaxocc_current, total_capacity]
    colors = ['blue', 'red', 'green']
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.axhline(y=volume, color='blue', linestyle=':', alpha=0.5)
    ax4.set_ylabel('Calls per Hour')
    ax4.set_title('Capacity Analysis')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # SUBPLOT 5: AHT Impact Analysis
    ax5 = plt.subplot(2, 3, 5)
    aht_range = np.linspace(AHT * 0.5, AHT * 1.5, 50)
    aht_occ = []
    aht_sl = []
    aht_capacity = []
    
    for aht in aht_range:
        occ = calculate_occupancy(volume, aht, headcount, interval_seconds)
        aht_occ.append(occ * 100)
        traffic_intensity = (volume * aht/3600)
        sl = calculate_service_level(headcount, traffic_intensity, aht, ASA_target)
        aht_sl.append(sl * 100)
        capacity = calculate_maxcap_per_agent(aht, interval_seconds, target_occ)
        aht_capacity.append(capacity)
    
    ax5.plot(aht_range, aht_occ, 'b-', linewidth=2, label='Occupancy')
    ax5.plot(aht_range, aht_sl, 'g-', linewidth=2, label='Service Level')
    ax5.axvline(x=AHT, color='black', linestyle='--', alpha=0.5, label='Current AHT')
    ax5.set_xlabel('AHT (seconds)')
    ax5.set_ylabel('Percentage (%)')
    ax5.set_title('AHT Impact on Performance')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best')
    
    ax5_twin = ax5.twinx()
    ax5_twin.plot(aht_range, aht_capacity, 'r--', linewidth=2, label='Capacity/Agent')
    ax5_twin.set_ylabel('Calls per Agent per Hour', color='red')
    ax5_twin.tick_params(axis='y', labelcolor='red')
    
    # SUBPLOT 6: Cost of SLA Improvement
    ax6 = plt.subplot(2, 3, 6)
    sla_targets = np.arange(50, 101, 5)
    additional_hc_needed = []
    
    current_traffic = (volume * AHT/3600)
    current_sl = calculate_service_level(headcount, current_traffic, AHT, ASA_target) * 100
    
    for sla_target in sla_targets:
        hc_low = headcount * 0.5
        hc_high = headcount * 3
        for _ in range(20):
            hc_mid = (hc_low + hc_high) / 2
            sl_mid = calculate_service_level(hc_mid, current_traffic, AHT, ASA_target) * 100
            if sl_mid >= sla_target:
                hc_high = hc_mid
            else:
                hc_low = hc_mid
        hc_needed = (hc_low + hc_high) / 2
        additional_hc = max(0, hc_needed - headcount)
        additional_hc_needed.append(additional_hc)
    
    ax6.bar(sla_targets, additional_hc_needed, width=3, alpha=0.7, color='purple')
    ax6.axvline(x=current_sl, color='red', linestyle='--', linewidth=2, label=f'Current SLA ({current_sl:.1f}%)')
    ax6.set_xlabel('Service Level Target (%)')
    ax6.set_ylabel('Additional Headcount Needed')
    ax6.set_title('Cost of SLA Improvement')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper left')
    
    plt.suptitle('Dynamic Trade-off Analysis: Occupancy vs Service Level Optimization',
                fontsize=14, fontweight='bold', y=1.20)
    plt.tight_layout()
    st.pyplot(fig)

# ========================
# TAB 2: DYNAMIC OPTIMIZATION TOOL
# ========================

def optimization_tab():
    st.header("DYNAMIC OPTIMIZATION TOOL")
    st.markdown("Find the optimal headcount that balances your occupancy and SLA targets:")
    
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
        run_optimization(opt_volume, opt_AHT, opt_ASA, opt_target_sla, opt_target_occ, opt_interval)

def find_optimal_headcount(volume, AHT, ASA_target, target_sla, target_occ, interval_minutes=60):
    """Find optimal headcount (same as original)"""
    interval_seconds = interval_minutes * 60
    traffic_intensity = (volume * AHT/3600)
    
    best_hc = None
    best_score = -float('inf')
    results = []
    
    for hc in np.arange(1, 50, 0.5):
        occ = calculate_occupancy(volume, AHT, hc, interval_seconds)
        sl = calculate_service_level(hc, traffic_intensity, AHT, ASA_target)
        
        if occ > 1.0 or sl < 0:
            continue
        
        occ_score = -abs(occ - target_occ)
        sl_score = max(0, (sl * 100 - target_sla))
        score = occ_score * 0.4 + sl_score * 0.6
        
        results.append({
            'headcount': hc,
            'occupancy': occ * 100,
            'service_level': sl * 100,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_hc = hc
    
    results_df = pd.DataFrame(results)
    feasible = results_df[
        (results_df['occupancy'] >= target_occ * 100 * 0.95) &
        (results_df['service_level'] >= target_sla)
    ]
    
    if len(feasible) > 0:
        optimal = feasible.loc[feasible['score'].idxmax()]
    else:
        optimal = results_df.loc[results_df['score'].idxmax()]
    
    return optimal, results_df

def run_optimization(volume, AHT, ASA, target_sla, target_occ, interval_minutes):
    """Run optimization and display results"""
    interval_seconds = interval_minutes * 60
    
    # Find optimal headcount
    optimal_result, all_results = find_optimal_headcount(
        volume, AHT, ASA, target_sla, target_occ, interval_minutes
    )
    
    # Calculate metrics
    callsmaxocc_at_target = calculate_callsmaxocc(
        optimal_result['headcount'], AHT, target_occ, interval_seconds
    )
    callsmaxocc_at_result = calculate_callsmaxocc(
        optimal_result['headcount'], AHT, optimal_result['occupancy']/100, interval_seconds
    )
    maxcap_per_agent = calculate_maxcap_per_agent(AHT, interval_seconds, target_occ)
    
    # Display results
    st.subheader("OPTIMIZATION RESULTS")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Optimal Headcount", f"{optimal_result['headcount']:.1f} agents")
        st.metric("Resulting Occupancy", f"{optimal_result['occupancy']:.1f}%")
        st.metric("Resulting Service Level", f"{optimal_result['service_level']:.1f}%")
    
    with col2:
        st.metric(f"CallsMaxOcc ({target_occ*100:.0f}% Occ)", f"{callsmaxocc_at_target:.1f} calls/hour")
        st.metric("MaxCap per agent", f"{maxcap_per_agent:.2f} calls/hour")
        st.metric("Total team capacity", f"{maxcap_per_agent * optimal_result['headcount']:.1f} calls/hour")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Headcount vs Performance metrics
    hc_values = np.linspace(1, max(20, optimal_result['headcount'] * 1.5), 100)
    occ_values = []
    sla_values = []
    
    for hc in hc_values:
        occ = calculate_occupancy(volume, AHT, hc, interval_seconds)
        occ_values.append(occ * 100)
        traffic = (volume * AHT/3600)
        sla = calculate_service_level(hc, traffic, AHT, ASA) * 100
        sla_values.append(sla)
    
    ax1.plot(hc_values, occ_values, 'b-', linewidth=2, label='Occupancy')
    ax1.plot(hc_values, sla_values, 'g-', linewidth=2, label='Service Level')
    ax1.axvline(x=optimal_result['headcount'], color='red', linestyle='--',
               alpha=0.7, label=f'Optimal HC ({optimal_result["headcount"]:.1f})')
    ax1.axhline(y=target_occ*100, color='blue', alpha=0.3, linestyle='--')
    ax1.axhline(y=target_sla, color='green', alpha=0.3, linestyle='--')
    ax1.set_xlabel('Headcount')
    ax1.set_ylabel('Performance (%)')
    ax1.set_title('Headcount Optimization: Occupancy vs SLA')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_ylim([0, 105])
    
    # Plot 2: Capacity comparison
    capacity_metrics = ['Volume', f'CallsMaxOcc\n({target_occ*100:.0f}% Occ)', 'Team Capacity']
    capacity_values = [volume, callsmaxocc_at_target, maxcap_per_agent * optimal_result['headcount']]
    colors = ['blue', 'red', 'green']
    
    bars = ax2.bar(capacity_metrics, capacity_values, color=colors, alpha=0.7)
    ax2.axhline(y=volume, color='blue', linestyle=':', alpha=0.3)
    
    for bar, val in zip(bars, capacity_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(capacity_values)*0.05,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Calls per Hour')
    ax2.set_title(f'Capacity Comparison\n(Optimal HC: {optimal_result["headcount"]:.1f})')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Summary table
    st.subheader("Summary Table")
    summary_data = {
        'Metric': [
            'Optimal Headcount',
            'Resulting Occupancy',
            'Resulting SLA',
            f'CallsMaxOcc ({target_occ*100:.0f}% Occ)',
            'Volume vs CallsMaxOcc Ratio',
            'MaxCap per agent'
        ],
        'Value': [
            f"{optimal_result['headcount']:.1f}",
            f"{optimal_result['occupancy']:.1f}%",
            f"{optimal_result['service_level']:.1f}%",
            f"{callsmaxocc_at_target:.1f}",
            f"{volume/callsmaxocc_at_target:.2f}x",
            f"{maxcap_per_agent:.2f}"
        ],
        'Note': [
            'Balances both targets',
            '',
            '',
            'Max calls for target Occ',
            'Ratio',
            'Calls/hour at target Occ'
        ]
    }
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Key insight
    st.subheader("Key Insight")
    if volume > callsmaxocc_at_target:
        st.warning(f"• You have MORE volume than needed for {target_occ*100:.0f}% occupancy")
        st.info("  → To maintain occupancy target, you need to increase HC or reduce AHT")
    else:
        st.success(f"• You have LESS volume than needed for {target_occ*100:.0f}% occupancy")
        st.info("  → To maintain occupancy target, you need to decrease HC or increase AHT")

# ========================
# TAB 3: MATHEMATICAL CURVE ANALYSIS
# ========================

def math_curve_analysis_tab():
    st.header("MATHEMATICAL CURVE ANALYSIS")
    st.markdown("Explore the mathematical relationships between Occupancy and SLA curves")
    
    # Use parameters from Tab 1 if available, otherwise use defaults
    if 'tradeoff_params' in st.session_state:
        params = st.session_state.tradeoff_params
        default_volume = params['volume']
        default_AHT = params['AHT']
        default_ASA = params['ASA_target']
        default_target_occ = params['target_occ']
    else:
        default_volume = 40
        default_AHT = 390
        default_ASA = 30
        default_target_occ = 0.8
    
    # Allow user to adjust if needed
    col1, col2 = st.columns(2)
    
    with col1:
        math_volume = st.number_input("Volume:", min_value=1, max_value=200, value=default_volume, key="math_volume")
        math_AHT = st.number_input("AHT (s):", min_value=60, max_value=1200, value=default_AHT, key="math_AHT")
    
    with col2:
        math_ASA = st.number_input("ASA (s):", min_value=5, max_value=180, value=default_ASA, key="math_ASA")
        math_target_occ = st.slider("Target Occupancy:", 0.1, 1.0, default_target_occ, 0.01, format="%.0%%", key="math_target_occ")
        math_target_sla = st.slider("Target SLA:", 0.5, 1.0, 0.9, 0.01, format="%.0%%", key="math_target_sla")
    
    if st.button("Run Mathematical Analysis", type="primary"):
        plot_mathematical_analysis(math_volume, math_AHT, math_ASA, math_target_occ, math_target_sla)

# ========================
# TAB 4: MATHEMATICAL ANALYSIS RESULTS
# ========================

def math_results_tab():
    st.header("MATHEMATICAL ANALYSIS RESULTS")
    
    if 'tradeoff_params' not in st.session_state:
        st.warning("Please run the Trade-off Analysis first to get parameters.")
        return
    
    params = st.session_state.tradeoff_params
    
    # Extract parameters
    volume = params['volume']
    AHT = params['AHT']
    ASA = params['ASA_target']
    target_occ = params['target_occ']
    target_sla = 0.9  # Default SLA target
    
    # Run the analysis
    interval = 3600
    A = volume * (AHT/3600)
    k_occ = (volume * AHT) / interval
    
    # Calculate HC values
    hc_for_target_occ = k_occ / target_occ
    
    # Find HC for target SLA
    hc_for_target_sla = None
    def sla_eq(hc):
        if hc <= A:
            return -target_sla
        return sla_function(hc, A, AHT, ASA) - target_sla
    
    try:
        for guess in [A*0.5, A, A*1.5, A*2]:
            try:
                result = fsolve(sla_eq, guess, full_output=True)
                if result[2] == 1:
                    hc_val = result[0][0]
                    if hc_val > A:
                        hc_for_target_sla = hc_val
                        break
            except:
                continue
    except:
        pass
    
    # Display results
    st.subheader("Mathematical Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Traffic Intensity (A)", f"{A:.3f} Erlangs")
        st.metric("Constant (k)", f"{k_occ:.3f}")
    
    with col2:
        st.metric(f"HC for {target_occ*100:.0f}% Occ", f"{hc_for_target_occ:.2f}")
        if hc_for_target_sla:
            sla_at_hc = sla_function(hc_for_target_occ, A, AHT, ASA) * 100
            st.metric(f"SLA at this HC", f"{sla_at_hc:.1f}%")
    
    with col3:
        if hc_for_target_sla:
            st.metric(f"HC for {target_sla*100:.0f}% SLA", f"{hc_for_target_sla:.2f}")
            occ_at_hc = k_occ / hc_for_target_sla
            st.metric(f"Occ at this HC", f"{occ_at_hc*100:.1f}%")
    
    # Display equation forms
    st.subheader("Curve Equations")
    st.code(f"""
    1. Occupancy(HC) = k / HC
       where k = (Volume × AHT) / Interval
       k = ({volume} × {AHT}) / 3600 = {k_occ:.3f}
       ∴ Occ(HC) = {k_occ:.3f} / HC
    
    2. SLA(HC) = 1 - P_wait × exp(-(HC - A) × ASA / AHT)
       where A = Volume × (AHT/3600) = {A:.3f} Erlangs
       P_wait = Erlang_C_Probability(HC, A)
    """)
    
    # AHT adjustment analysis
    st.subheader("AHT Adjustment Analysis")
    
    # Calculate AHT for perfect balance
    if hc_for_target_sla:
        gap = abs(hc_for_target_occ - hc_for_target_sla)
        st.info(f"Gap between targets: {gap:.2f} HC")
        
        # Calculate required AHT adjustments
        aht_for_occ80 = (target_occ * hc_for_target_occ * 3600) / volume
        aht_for_sla90 = (target_occ * hc_for_target_sla * 3600) / volume
        
        st.write("**Required AHT adjustments:**")
        st.write(f"1. To get {target_occ*100:.0f}% Occ at HC={hc_for_target_occ:.1f}:")
        st.write(f"   → AHT = {aht_for_occ80:.0f}s ({(aht_for_occ80/AHT - 1)*100:+.1f}% change)")
        
        st.write(f"2. To get {target_occ*100:.0f}% Occ at HC={hc_for_target_sla:.1f}:")
        st.write(f"   → AHT = {aht_for_sla90:.0f}s ({(aht_for_sla90/AHT - 1)*100:+.1f}% change)")

def plot_mathematical_analysis(volume, AHT, ASA, target_occ, target_sla):
    """Plot mathematical analysis (simplified version)"""
    interval = 3600
    A = volume * (AHT/3600)
    k_occ = (volume * AHT) / interval
    
    # Generate headcount range
    hc_min = max(0.1, A * 0.1)
    hc_max = max(30, A * 3)
    hc_range = np.linspace(hc_min, hc_max, 200)
    
    # Calculate curves
    occ_curve = []
    sla_curve = []
    
    for hc in hc_range:
        occ = k_occ / hc if hc > 0 else 1.0
        occ_curve.append(occ)
        
        if hc <= A:
            sla = 0.0
        else:
            sla = sla_function(hc, A, AHT, ASA)
        sla_curve.append(sla)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Curves
    ax1.plot(hc_range, occ_curve, 'b-', linewidth=3, label='Occupancy Curve')
    ax1.plot(hc_range, sla_curve, 'g-', linewidth=3, label='Service Level Curve')
    ax1.axhline(y=target_occ, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=target_sla, color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Headcount (HC)')
    ax1.set_ylabel('Value (0-1 scale)')
    ax1.set_title('Mathematical Curves: Occupancy vs Service Level')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xlim([hc_min, hc_max])
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Difference curve
    diff_curve = np.array(occ_curve)
