# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Call Center Occupancy Optimizer")

# ========================
# CALCULATION FUNCTIONS
# ========================

def calculate_occupancy(volume, AHT, headcount, interval_seconds):
    """Calculate occupancy rate"""
    if headcount == 0:
        return 0
    return (volume * AHT / 3600) / (headcount * interval_seconds / 3600)

def calculate_service_level(headcount, traffic_intensity, AHT, ASA):
    """Calculate Erlang C service level"""
    if headcount <= traffic_intensity:
        return 0.0
    
    # Simplified Erlang C approximation
    # In production, use proper Erlang C formula
    from math import exp
    
    # Probability of waiting
    Pw = traffic_intensity ** headcount / (np.math.factorial(headcount) * 
          (headcount - traffic_intensity) * sum([traffic_intensity ** i / np.math.factorial(i) for i in range(headcount)]))
    
    # Service level
    sla = 1 - Pw * exp(-(headcount - traffic_intensity) * ASA / AHT)
    return max(0.0, min(1.0, sla))

def calculate_callsmaxocc(headcount, AHT, target_occ, interval_seconds):
    """Calculate maximum calls per hour at target occupancy"""
    return (headcount * interval_seconds / AHT) * target_occ

def calculate_maxcap_per_agent(AHT, interval_seconds, target_occ):
    """Calculate maximum capacity per agent"""
    return (interval_seconds / AHT) * target_occ

def find_optimal_headcount(volume, AHT, ASA, target_sla, target_occ, interval_minutes):
    """Find headcount that best balances occupancy and SLA targets"""
    interval_seconds = interval_minutes * 60
    
    # Search for optimal headcount
    best_score = float('inf')
    best_result = None
    all_results = []
    
    for hc in np.arange(1, 50, 0.5):
        # Calculate occupancy and SLA
        occ = calculate_occupancy(volume, AHT, hc, interval_seconds) * 100
        traffic = volume * AHT / 3600
        sla = calculate_service_level(hc, traffic, AHT, ASA) * 100
        
        # Calculate how far from targets
        occ_distance = abs(occ - target_occ*100)
        sla_distance = abs(sla - target_sla)
        
        # Combined score (weighted)
        score = occ_distance + sla_distance * 2  # SLA is more important
        
        all_results.append({
            'headcount': hc,
            'occupancy': occ,
            'service_level': sla,
            'score': score
        })
        
        if score < best_score:
            best_score = score
            best_result = all_results[-1]
    
    return best_result, all_results

# ========================
# STREAMLIT UI FUNCTIONS
# ========================

def create_optimization_widget():
    """Create interactive optimization widget using Streamlit"""
    
    st.header("🔧 Dynamic Optimization Tool")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        volume = st.slider(
            "Call Volume (calls/hour)",
            min_value=1.0,
            max_value=200.0,
            value=40.0,
            step=1.0,
            help="Number of calls arriving per hour"
        )
        
        AHT = st.slider(
            "Average Handle Time (seconds)",
            min_value=60.0,
            max_value=1200.0,
            value=390.0,
            step=10.0,
            help="Average time to handle a call"
        )
        
        ASA = st.slider(
            "ASA Target (seconds)",
            min_value=5.0,
            max_value=180.0,
            value=30.0,
            step=5.0,
            help="Acceptable wait time for service level calculation"
        )
    
    with col2:
        target_sla = st.slider(
            "Target Service Level (%)",
            min_value=50.0,
            max_value=100.0,
            value=90.0,
            step=1.0,
            help="Desired service level percentage"
        )
        
        target_occ = st.slider(
            "Target Occupancy",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.01,
            format="%.0f%%",
            help="Desired occupancy rate"
        )
        
        interval = st.selectbox(
            "Interval (minutes)",
            options=[15, 30, 60],
            index=2,
            help="Time period for calculations"
        )
    
    # Run optimization button
    if st.button("🚀 Run Optimization", type="primary"):
        return run_optimization_analysis(volume, AHT, ASA, target_sla, target_occ, interval)
    
    return None

def run_optimization_analysis(volume, AHT, ASA, target_sla, target_occ, interval_minutes):
    """Run the optimization analysis and display results"""
    
    interval_seconds = interval_minutes * 60
    
    # Find optimal headcount
    optimal_result, all_results = find_optimal_headcount(
        volume, AHT, ASA, target_sla, target_occ, interval_minutes
    )
    
    # Calculate additional metrics
    callsmaxocc_at_target = calculate_callsmaxocc(
        optimal_result['headcount'], AHT, target_occ, interval_seconds
    )
    
    callsmaxocc_at_result = calculate_callsmaxocc(
        optimal_result['headcount'], AHT, optimal_result['occupancy']/100, interval_seconds
    )
    
    maxcap_per_agent = calculate_maxcap_per_agent(AHT, interval_seconds, target_occ)
    
    # Calculate headcount needed for target occupancy only
    hc_for_target_occ = (volume * AHT / 3600) / (target_occ * interval_seconds / 3600)
    
    # Display results
    st.success("✅ Optimization Complete!")
    
    # Key Metrics
    st.subheader("📊 Optimization Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Optimal Headcount", f"{optimal_result['headcount']:.1f}")
    
    with col2:
        st.metric("Resulting Occupancy", f"{optimal_result['occupancy']:.1f}%")
    
    with col3:
        st.metric("Resulting Service Level", f"{optimal_result['service_level']:.1f}%")
    
    with col4:
        st.metric("Team Capacity", f"{maxcap_per_agent * optimal_result['headcount']:.1f} calls/hour")
    
    # Detailed Analysis
    st.subheader("📈 Detailed Analysis")
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Headcount vs Performance
    hc_values = np.linspace(1, max(20, optimal_result['headcount'] * 1.5), 100)
    occ_values = []
    sla_values = []
    
    for hc in hc_values:
        occ = calculate_occupancy(volume, AHT, hc, interval_seconds) * 100
        occ_values.append(occ)
        
        traffic = volume * AHT / 3600
        sla = calculate_service_level(hc, traffic, AHT, ASA) * 100
        sla_values.append(sla)
    
    ax1.plot(hc_values, occ_values, 'b-', linewidth=2, label='Occupancy')
    ax1.plot(hc_values, sla_values, 'g-', linewidth=2, label='Service Level')
    
    # Mark key points
    ax1.axvline(x=optimal_result['headcount'], color='red', linestyle='--',
               alpha=0.7, label=f'Optimal HC ({optimal_result["headcount"]:.1f})')
    ax1.axhline(y=target_occ*100, color='blue', alpha=0.3, linestyle='--')
    ax1.axhline(y=target_sla, color='green', alpha=0.3, linestyle='--')
    
    ax1.set_xlabel('Headcount')
    ax1.set_ylabel('Performance (%)')
    ax1.set_title('Headcount Optimization: Occupancy vs SLA')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 105])
    
    # Plot 2: Capacity comparison
    capacity_metrics = ['Current Volume', f'Max at {target_occ*100:.0f}% Occ', 'Team Capacity']
    capacity_values = [volume, callsmaxocc_at_target, maxcap_per_agent * optimal_result['headcount']]
    colors = ['blue', 'red', 'green']
    
    bars = ax2.bar(capacity_metrics, capacity_values, color=colors, alpha=0.7)
    ax2.set_ylabel('Calls per Hour')
    ax2.set_title('Capacity Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, capacity_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(capacity_values)*0.05,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
    
    # Insights
    st.subheader("💡 Key Insights")
    
    if volume > callsmaxocc_at_target:
        st.warning(f"""
        **Occupancy Warning:** You have MORE volume ({volume:.1f}) than needed for {target_occ*100:.0f}% occupancy ({callsmaxocc_at_target:.1f}).
        
        **Recommendation:** To maintain occupancy target, consider:
        - Increasing headcount to {hc_for_target_occ:.1f} agents
        - Reducing AHT through process improvements
        - Implementing call deflection strategies
        """)
    else:
        st.info(f"""
        **Capacity Available:** You have LESS volume ({volume:.1f}) than needed for {target_occ*100:.0f}% occupancy ({callsmaxocc_at_target:.1f}).
        
        **Opportunity:** You can handle additional volume while maintaining targets.
        """)
    
    # Summary Table
    st.subheader("📋 Summary Table")
    
    summary_data = pd.DataFrame({
        'Metric': [
            'Optimal Headcount',
            'Resulting Occupancy',
            'Resulting Service Level',
            f'Max Calls at {target_occ*100:.0f}% Occ',
            'Volume to Max Capacity Ratio',
            'Capacity per Agent'
        ],
        'Value': [
            f"{optimal_result['headcount']:.1f} agents",
            f"{optimal_result['occupancy']:.1f}%",
            f"{optimal_result['service_level']:.1f}%",
            f"{callsmaxocc_at_target:.1f} calls/hour",
            f"{volume/callsmaxocc_at_target:.2f}x",
            f"{maxcap_per_agent:.2f} calls/hour"
        ],
        'Note': [
            'Balances both occupancy and SLA',
            f'Target was {target_occ*100:.0f}%',
            f'Target was {target_sla}%',
            'Maximum calls for target occupancy',
            'How close to capacity',
            'At target occupancy'
        ]
    })
    
    st.dataframe(summary_data, use_container_width=True)
    
    return optimal_result

# ========================
# MAIN APP
# ========================

def main():
    """Main Streamlit app"""
    st.title("📞 Call Center Occupancy & Service Level Optimizer")
    
    st.markdown("""
    This tool helps you find the optimal balance between:
    - **Occupancy:** Agent utilization (higher = more efficient)
    - **Service Level:** Customer wait time (higher = better service)
    
    Adjust the parameters in the sidebar to simulate different scenarios.
    """)
    
    # Sidebar for additional controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("About")
        st.info("""
        This tool uses Erlang C calculations to model call center performance.
        
        **Key Metrics:**
        - **Volume:** Calls per hour
        - **AHT:** Average Handle Time
        - **ASA:** Acceptable Service Time
        - **Occupancy:** Agent busy time
        - **Service Level:** % answered within ASA
        """)
        
        if st.button("🔄 Reset All", type="secondary"):
            st.rerun()
    
    # Run the optimization widget
    results = create_optimization_widget()
    
    if results:
        # Option to download results
        st.divider()
        st.subheader("📥 Export Results")
        
        if st.button("Download Results as CSV"):
            results_df = pd.DataFrame([results])
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name="optimization_results.csv",
                mime="text/csv"
            )

# ========================
# ENTRY POINT
# ========================

if __name__ == "__main__":
    main()
