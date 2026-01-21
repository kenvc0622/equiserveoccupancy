import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.optimize import minimize_scalar
import pandas as pd
import warnings
import base64
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Equiserve Occupancy Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# GLOBAL SETTINGS & SESSION STATE
# ========================

# Initialize session state for global parameters
if 'global_target_sla' not in st.session_state:
    st.session_state.global_target_sla = 90  # Default 90%

if 'global_target_occupancy' not in st.session_state:
    st.session_state.global_target_occupancy = 80  # Default 80%

# ========================
# ENHANCED CORE FUNCTIONS
# ========================

def erlang_c_probability_wait(N, A):
    """Calculate probability of wait using Erlang C formula"""
    if A >= N:
        return 1.0
    
    try:
        sum_term = 0
        for i in range(int(N)):
            sum_term += (A**i) / factorial(i)
        
        numerator = (A**N / factorial(N)) * (N / (N - A))
        denominator = sum_term + numerator
        C = numerator / denominator if denominator != 0 else 1.0
        return min(1.0, max(0.0, C))
    except:
        return 1.0

def calculate_service_level(N, A, AHT, target_time):
    """Calculate service level (% answered within target_time)"""
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
    """Calculate occupancy percentage"""
    if headcount <= 0 or interval_seconds <= 0:
        return 0.0
    occupancy = (volume * AHT) / (headcount * interval_seconds)
    return min(1.0, max(0.0, occupancy))

def calculate_required_hc_for_sla(volume, AHT, target_sla_pct, asa_target, interval_seconds=3600):
    """Calculate headcount required to achieve target SLA"""
    traffic_intensity = (volume * AHT / 3600)
    
    def sla_objective(N):
        if N <= traffic_intensity:
            return 1000  # Penalty
        sla = calculate_service_level(N, traffic_intensity, AHT, asa_target) * 100
        return abs(sla - target_sla_pct)
    
    lower_bound = max(1, int(traffic_intensity) + 1)
    upper_bound = lower_bound + 30
    
    try:
        result = minimize_scalar(
            sla_objective,
            bounds=(lower_bound, upper_bound),
            method='bounded',
            options={'xatol': 0.1}
        )
        return max(1, np.ceil(result.x))
    except:
        # Fallback search
        for N in range(lower_bound, upper_bound + 1):
            sla = calculate_service_level(N, traffic_intensity, AHT, asa_target) * 100
            if sla >= target_sla_pct:
                return N
        return lower_bound

def calculate_shrinkage_adjusted_hc(base_hc, shrinkage_pct):
    """Adjust headcount for shrinkage"""
    if shrinkage_pct >= 100:
        return float('inf')
    return base_hc / (1 - shrinkage_pct/100)

def classify_risk(sla_percent, occupancy_percent, target_sla, target_occ):
    """Classify hour into risk categories"""
    
    sla_buffer = sla_percent - target_sla
    
    # Classification logic
    if sla_percent >= target_sla + 5:  # Comfortable buffer
        if occupancy_percent >= target_occ - 5:
            return "✅ Optimal", "Low"
        else:
            return "✅ Good SLA", "Low"
    
    elif target_sla <= sla_percent < target_sla + 5:  # Tight but okay
        if occupancy_percent >= target_occ:
            return "⚠️ Marginal", "Medium"
        else:
            return "⚠️ Low Occupancy", "Medium"
    
    elif target_sla - 10 <= sla_percent < target_sla:  # Below target
        return "⚠️ Below Target", "High"
    
    else:  # Significantly below target
        return "❌ Critical", "Severe"

# ========================
# SIDEBAR FOR GLOBAL SETTINGS
# ========================

with st.sidebar:
    st.header("⚙️ Global Settings")
    
    # Global SLA Target Slider
    st.session_state.global_target_sla = st.slider(
        "Target Service Level (%)",
        min_value=70,
        max_value=99,
        value=st.session_state.global_target_sla,
        step=1,
        help="Target SLA percentage applied across all tabs"
    )
    
    # Global Occupancy Target Slider (0-100%)
    st.session_state.global_target_occupancy = st.slider(
        "Target Occupancy (%)",
        min_value=0,
        max_value=100,
        value=st.session_state.global_target_occupancy,
        step=1,
        help="Target occupancy percentage applied across all tabs"
    )
    
    st.markdown("---")
    st.caption("Changes apply to all analysis tabs")

# ========================
# MAIN APP
# ========================

def main():
    st.title("📞 Equiserve Occupancy Analysis Tool")
    st.markdown("""
    This tool helps analyze and optimize the trade-off between agent occupancy and service level (SLA) 
    in call center operations using Erlang C calculations.
    """)
    
    # Create tabs - ADDING NEW TERMINOLOGY TAB
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Dynamic Trade-off Analysis",
        "⚙️ Optimization Engine", 
        "📐 Mathematical Analysis",
        "📊 Results Dashboard",
        "📅 Hour-by-Hour Staffing",  # NEW
        "📖 Terminology Guide"       # NEW
    ])
    
    # ========================
    # TAB 1: DYNAMIC TRADE-OFF ANALYSIS (Updated with global targets)
    # ========================
    
    with tab1:
        st.header("📈 DYNAMIC TRADE-OFF ANALYSIS")
        
        # Display current global targets
        st.info(f"**Current Targets:** SLA ≥ {st.session_state.global_target_sla}%, Occupancy ≥ {st.session_state.global_target_occupancy}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            volume = st.slider("Volume (calls/hr):", 1, 500, 45, 1, key="tradeoff_volume")
            AHT = st.slider("AHT (seconds):", 60, 1200, 390, 10, key="tradeoff_aht")
            ASA_target = st.slider("ASA Target (seconds):", 5, 300, 30, 5, key="tradeoff_asa")
        
        with col2:
            headcount = st.slider("Headcount:", 1.0, 100.0, 7.1, 0.5, key="tradeoff_hc")
            # Using global target occupancy instead of separate slider
            target_occ_pct = st.session_state.global_target_occupancy
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
            occ_diff = (current_occ*100 - target_occ_pct)
            delta_text = f"{occ_diff:+.1f}%" if abs(occ_diff) > 0.1 else "On target"
            st.metric("Occupancy", f"{current_occ*100:.1f}%", delta=delta_text)
            st.metric("Volume", f"{volume} calls/hr")
        with mcol2:
            sla_diff = (current_sl*100 - st.session_state.global_target_sla)
            delta_text = f"{sla_diff:+.1f}%" if abs(sla_diff) > 0.1 else "On target"
            st.metric("Service Level", f"{current_sl*100:.1f}%", delta=delta_text)
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
            'target_occ_pct': target_occ_pct,
            'interval_minutes': interval_minutes,
            'current_occ': current_occ,
            'current_sl': current_sl
        }
        
        # Generate analysis
        if st.button("Generate Comprehensive Analysis", type="primary", key="gen_analysis"):
            with st.spinner("Generating analysis..."):
                # Create figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                
                # Plot 1: Occupancy vs Headcount
                hc_range = np.linspace(max(1, headcount * 0.5), headcount * 2, 50)
                occ_values = [calculate_occupancy(volume, AHT, hc, interval_seconds) * 100 for hc in hc_range]
                
                axes[0].plot(hc_range, occ_values, 'b-', linewidth=2.5, label='Occupancy')
                axes[0].axvline(x=headcount, color='r', linestyle='--', linewidth=2, label=f'Current: {headcount:.1f}')
                axes[0].axhline(y=target_occ_pct, color='orange', linestyle=':', linewidth=2, label=f'Target: {target_occ_pct}%')
                axes[0].fill_between(hc_range, occ_values, target_occ_pct, where=np.array(occ_values) >= target_occ_pct, 
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
                axes[1].axhline(y=st.session_state.global_target_sla, color='darkgreen', linestyle=':', linewidth=2, 
                               label=f'Target: {st.session_state.global_target_sla}%')
                axes[1].fill_between(hc_range, sl_values, st.session_state.global_target_sla, 
                                     where=np.array(sl_values) >= st.session_state.global_target_sla, 
                                     alpha=0.2, color='lightgreen', label=f'Above {st.session_state.global_target_sla}%')
                axes[1].set_xlabel('Headcount', fontsize=11, fontweight='bold')
                axes[1].set_ylabel('Service Level (%)', fontsize=11, fontweight='bold')
                axes[1].set_title('Service Level vs Headcount', fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(loc='best')
                
                # Plot 3: Trade-off Curve
                axes[2].plot(occ_values, sl_values, 'purple', linewidth=2.5, label='Trade-off Curve')
                axes[2].scatter([current_occ*100], [current_sl*100], color='red', s=100, zorder=5, 
                               label=f'Current: ({current_occ*100:.1f}%, {current_sl*100:.1f}%)')
                
                # Add target zones
                axes[2].axvline(x=target_occ_pct, color='orange', linestyle=':', alpha=0.7, label=f'Target Occ')
                axes[2].axhline(y=st.session_state.global_target_sla, color='darkgreen', linestyle=':', alpha=0.7, label=f'Target SLA')
                
                # Shade optimal quadrant
                axes[2].fill_between([target_occ_pct, 100], [st.session_state.global_target_sla, st.session_state.global_target_sla], 
                                     100, alpha=0.1, color='green', label='Optimal Zone')
                
                axes[2].set_xlabel('Occupancy (%)', fontsize=11, fontweight='bold')
                axes[2].set_ylabel('Service Level (%)', fontsize=11, fontweight='bold')
                axes[2].set_title('Occupancy vs Service Level Trade-off', fontsize=12, fontweight='bold')
                axes[2].grid(True, alpha=0.3)
                axes[2].legend(loc='best')
                
                plt.tight_layout()
                st.pyplot(fig)
        # ========================
    # TAB 2: OPTIMIZATION ENGINE (Updated with global targets)
    # ========================
    
    with tab2:
        st.header("⚙️ OPTIMIZATION ENGINE")
        st.info(f"**Optimizing for:** SLA ≥ {st.session_state.global_target_sla}%, Occupancy ≥ {st.session_state.global_target_occupancy}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            opt_volume = st.slider("Call Volume:", 1, 500, 40, 1, key="opt_volume")
            opt_AHT = st.slider("Average Handle Time (s):", 60, 1200, 390, 10, key="opt_AHT")
            opt_ASA = st.slider("ASA Target (s):", 5, 300, 30, 5, key="opt_ASA")
        
        with col2:
            # Use global targets instead of separate sliders
            opt_target_sla = st.session_state.global_target_sla
            opt_target_occ = st.session_state.global_target_occupancy
            opt_interval = st.selectbox("Interval Duration:", [15, 30, 60], index=2, key="opt_interval")
        
        if st.button("🚀 Run Optimization Analysis", type="primary", key="run_opt"):
            with st.spinner("Running optimization analysis..."):
                # Calculate traffic intensity
                traffic_intensity = (opt_volume * opt_AHT) / 3600
                
                # Find optimal headcount range
                min_hc = max(1, int(traffic_intensity) + 1)
                max_hc = min_hc + 20
                
                # Analyze different headcounts
                results = []
                for hc in range(min_hc, max_hc + 1):
                    # Calculate metrics
                    sla = calculate_service_level(hc, traffic_intensity, opt_AHT, opt_ASA) * 100
                    occupancy = calculate_occupancy(opt_volume, opt_AHT, hc, opt_interval * 60) * 100
                    
                    # Calculate efficiency score (balanced approach)
                    sla_score = min(100, sla / opt_target_sla * 100) if opt_target_sla > 0 else 0
                    occ_score = min(100, occupancy / opt_target_occ * 100) if opt_target_occ > 0 else 0
                    efficiency_score = (sla_score * 0.6 + occ_score * 0.4)  # Weighted average
                    
                    # Determine optimization strategy
                    if sla >= opt_target_sla and occupancy >= opt_target_occ:
                        strategy = "✅ Balanced"
                    elif sla >= opt_target_sla:
                        strategy = "📈 SLA-Optimized"
                    elif occupancy >= opt_target_occ:
                        strategy = "📊 Occupancy-Targeted"
                    else:
                        strategy = "⚠️ Suboptimal"
                    
                    results.append({
                        'Headcount': hc,
                        'SLA %': f"{sla:.1f}%",
                        'Occupancy %': f"{occupancy:.1f}%",
                        'Efficiency Score': f"{efficiency_score:.1f}",
                        'Strategy': strategy
                    })
                
                # Create DataFrame
                results_df = pd.DataFrame(results)
                
                # Find optimal solutions
                sla_optimal = results_df[results_df['SLA %'].str.replace('%', '').astype(float) >= opt_target_sla].iloc[0] if not results_df[results_df['SLA %'].str.replace('%', '').astype(float) >= opt_target_sla].empty else None
                occ_optimal = results_df[results_df['Occupancy %'].str.replace('%', '').astype(float) >= opt_target_occ].iloc[0] if not results_df[results_df['Occupancy %'].str.replace('%', '').astype(float) >= opt_target_occ].empty else None
                balanced_optimal = results_df.sort_values('Efficiency Score', ascending=False).iloc[0]
                
                # Display results
                st.subheader("📊 Optimization Results")
                
                # Display optimal solutions
                opt_col1, opt_col2, opt_col3 = st.columns(3)
                
                with opt_col1:
                    if sla_optimal is not None:
                        st.metric(
                            "SLA-Optimized",
                            f"{sla_optimal['Headcount']} agents",
                            f"{sla_optimal['SLA %']} SLA"
                        )
                    else:
                        st.metric("SLA-Optimized", "Not achieved", "Below target")
                
                with opt_col2:
                    if occ_optimal is not None:
                        st.metric(
                            "Occupancy-Targeted",
                            f"{occ_optimal['Headcount']} agents",
                            f"{occ_optimal['Occupancy %']} Occupancy"
                        )
                    else:
                        st.metric("Occupancy-Targeted", "Not achieved", "Below target")
                
                with opt_col3:
                    st.metric(
                        "Balanced Solution",
                        f"{balanced_optimal['Headcount']} agents",
                        f"Score: {balanced_optimal['Efficiency Score']}"
                    )
                
                # Display detailed table
                st.subheader("📈 Detailed Analysis")
                
                # Color function for strategies
                def color_strategy(val):
                    if val == "✅ Balanced":
                        return 'background-color: #d4edda; color: #155724;'
                    elif val == "📈 SLA-Optimized":
                        return 'background-color: #cce5ff; color: #004085;'
                    elif val == "📊 Occupancy-Targeted":
                        return 'background-color: #fff3cd; color: #856404;'
                    else:
                        return 'background-color: #f8d7da; color: #721c24;'
                
                styled_df = results_df.style.applymap(color_strategy, subset=['Strategy'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Prepare data
                hc_values = results_df['Headcount'].tolist()
                sla_values = [float(x.replace('%', '')) for x in results_df['SLA %']]
                occ_values = [float(x.replace('%', '')) for x in results_df['Occupancy %']]
                eff_values = [float(x) for x in results_df['Efficiency Score']]
                
                # Plot SLA and Occupancy
                ax.plot(hc_values, sla_values, 'g-', linewidth=2.5, label='SLA %')
                ax.plot(hc_values, occ_values, 'b-', linewidth=2.5, label='Occupancy %')
                
                # Add target lines
                ax.axhline(y=opt_target_sla, color='darkgreen', linestyle=':', linewidth=2, label=f'Target SLA ({opt_target_sla}%)')
                ax.axhline(y=opt_target_occ, color='darkblue', linestyle=':', linewidth=2, label=f'Target Occ ({opt_target_occ}%)')
                
                # Highlight optimal points
                if sla_optimal is not None:
                    ax.plot(sla_optimal['Headcount'], float(sla_optimal['SLA %'].replace('%', '')), 
                           'go', markersize=10, label='SLA-Optimal')
                
                if occ_optimal is not None:
                    ax.plot(occ_optimal['Headcount'], float(occ_optimal['Occupancy %'].replace('%', '')), 
                           'bo', markersize=10, label='Occ-Optimal')
                
                ax.plot(balanced_optimal['Headcount'], float(balanced_optimal['SLA %'].replace('%', '')), 
                       'ro', markersize=10, label='Balanced')
                
                ax.set_xlabel('Headcount', fontsize=11, fontweight='bold')
                ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
                ax.set_title('Optimization Analysis: SLA vs Occupancy Trade-off', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Efficiency score plot
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                
                ax2.plot(hc_values, eff_values, 'purple', linewidth=2.5, label='Efficiency Score')
                ax2.plot(balanced_optimal['Headcount'], float(balanced_optimal['Efficiency Score']), 
                        'ro', markersize=10, label='Optimal Point')
                ax2.set_xlabel('Headcount', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Efficiency Score', fontsize=11, fontweight='bold')
                ax2.set_title('Efficiency Score vs Headcount', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='best')
                
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.info("""
                    **Based on your targets:**
                    - **Target SLA:** {}%
                    - **Target Occupancy:** {}%
                    
                    **Traffic Intensity:** {:.2f} Erlangs
                    **Minimum Agents Required:** {} (N > A)
                    """.format(opt_target_sla, opt_target_occ, traffic_intensity, min_hc))
                
                with rec_col2:
                    st.success("""
                    **Recommended Staffing:**
                    
                    **{} agents** (Balanced Solution)
                    - Expected SLA: {}
                    - Expected Occupancy: {}
                    - Efficiency Score: {}
                    
                    **Trade-off:** {}
                    """.format(
                        balanced_optimal['Headcount'],
                        balanced_optimal['SLA %'],
                        balanced_optimal['Occupancy %'],
                        balanced_optimal['Efficiency Score'],
                        balanced_optimal['Strategy']
                    ))
                
                st.success("✅ Optimization complete!")
    # ========================
    # TAB 3: MATHEMATICAL ANALYSIS (Updated with global targets)
    # ========================
    
    with tab3:
        st.header("📐 MATHEMATICAL ANALYSIS")
        st.info(f"**Analysis Parameters:** SLA Target = {st.session_state.global_target_sla}%, Occupancy Target = {st.session_state.global_target_occupancy}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            math_volume = st.slider("Call Volume:", 1, 500, 37, 1, key="math_volume")
            math_AHT = st.slider("Average Handle Time (s):", 60, 1200, 390, 10, key="math_AHT")
            math_ASA = st.slider("ASA Target (s):", 5, 300, 30, 5, key="math_ASA")
        
        with col2:
            math_headcount_start = st.slider("Starting Headcount:", 0, 10, 0, 1, key="math_start_hc")
            math_headcount_end = st.slider("Ending Headcount:", 10, 50, 10, 1, key="math_end_hc")
            math_interval = st.selectbox("Interval (minutes):", [15, 30, 60], index=2, key="math_interval")
        
        if st.button("🧮 Generate Mathematical Analysis", type="primary", key="gen_math"):
            with st.spinner("Calculating Erlang C probabilities..."):
                # Calculate traffic intensity
                traffic_intensity = (math_volume * math_AHT) / 3600
                
                # Generate analysis table
                analysis_data = []
                headcount_range = range(math_headcount_start, math_headcount_end + 1)
                
                for N in headcount_range:
                    if N == 0:
                        # Special case for N=0
                        p_wait = 1.0
                        sla = 0.0
                        occupancy = 0.0
                        utilization = 0.0
                    else:
                        # Calculate metrics
                        p_wait = erlang_c_probability_wait(N, traffic_intensity) * 100
                        sla = calculate_service_level(N, traffic_intensity, math_AHT, math_ASA) * 100
                        occupancy = calculate_occupancy(math_volume, math_AHT, N, math_interval*60) * 100
                        utilization = (traffic_intensity / N) * 100 if N > 0 else 0
                    
                    analysis_data.append({
                        'Headcount': N,
                        'Traffic Intensity (Erlangs)': f"{traffic_intensity:.3f}",
                        'P(Wait) %': f"{p_wait:.2f}%",
                        'Service Level %': f"{sla:.2f}%",
                        'Occupancy %': f"{occupancy:.2f}%",
                        'Utilization Ratio': f"{utilization:.2f}%"
                    })
                
                # Create DataFrame
                analysis_df = pd.DataFrame(analysis_data)
                
                # Display results
                st.subheader("Erlang C Probability Analysis")
                st.dataframe(analysis_df, use_container_width=True)
                
                # Key mathematical insights
                st.subheader("Key Mathematical Insights")
                
                col_math1, col_math2 = st.columns(2)
                
                with col_math1:
                    st.markdown(f"""
                    ### Erlang C Formula:
                    
                    $$P(\\text{{wait}}) = \\frac{{(A^N / N!) \\times (N/(N-A))}}{{\\sum_{{i=0}}^{{N-1}}(A^i / i!) + (A^N / N!) \\times (N/(N-A))}}$$
                    
                    Where:  
                    - $A$ = Traffic Intensity = {traffic_intensity:.3f} Erlangs  
                    - $N$ = Number of agents
                    
                    ### Service Level Formula:
                    
                    $$SLA = 1 - P(\\text{{wait}}) \\times \\exp(-(N-A) \\times T / AHT)$$
                    
                    Where:  
                    - $T$ = ASA Target = {math_ASA} seconds  
                    - $AHT$ = {math_AHT} seconds  
                    - $N-A$ = Agent surplus
                    """)
                    
                with col_math2:
                    st.markdown(f"""
                    ### Key Relationships:
                    
                    1. **Traffic Intensity (A)**:  
                       $$A = \\frac{{\\text{{Volume}} \\times AHT}}{{3600}}$$
                    
                    2. **Occupancy**:  
                       $$\\text{{Occ}} = \\frac{{\\text{{Volume}} \\times AHT}}{{N \\times \\text{{Interval}}}}$$
                    
                    3. **Utilization Ratio**:  
                       $$U = \\frac{{A}}{{N}} \\times 100\\%$$
                    
                    4. **Agent Requirements**:  
                       Minimum agents needed: $N > A$
                    """)
                    
                # Create visualizations
                st.subheader("Mathematical Relationships")
                
                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Prepare data for plotting
                N_values = list(headcount_range)
                p_wait_values = [float(d['P(Wait) %'].strip('%')) for d in analysis_data]
                sla_values = [float(d['Service Level %'].strip('%')) for d in analysis_data]
                occ_values = [float(d['Occupancy %'].strip('%')) for d in analysis_data]
                
                # Plot 1: Probability of Wait vs Headcount
                axes[0, 0].plot(N_values, p_wait_values, 'b-', linewidth=2.5, marker='o')
                axes[0, 0].set_xlabel('Headcount', fontsize=11, fontweight='bold')
                axes[0, 0].set_ylabel('P(Wait) %', fontsize=11, fontweight='bold')
                axes[0, 0].set_title('Probability of Waiting vs Headcount', fontsize=12, fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].fill_between(N_values, p_wait_values, 0, alpha=0.2, color='blue')
                
                # Plot 2: Service Level vs Headcount
                axes[0, 1].plot(N_values, sla_values, 'g-', linewidth=2.5, marker='s')
                axes[0, 1].axhline(y=st.session_state.global_target_sla, color='darkgreen', linestyle=':', linewidth=2)
                axes[0, 1].set_xlabel('Headcount', fontsize=11, fontweight='bold')
                axes[0, 1].set_ylabel('Service Level %', fontsize=11, fontweight='bold')
                axes[0, 1].set_title('Service Level vs Headcount', fontsize=12, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].fill_between(N_values, sla_values, st.session_state.global_target_sla, 
                                       where=np.array(sla_values) >= st.session_state.global_target_sla, 
                                       alpha=0.2, color='lightgreen')
                
                # Plot 3: Occupancy vs Headcount
                axes[1, 0].plot(N_values, occ_values, 'r-', linewidth=2.5, marker='^')
                axes[1, 0].axhline(y=st.session_state.global_target_occupancy, color='orange', linestyle=':', linewidth=2)
                axes[1, 0].set_xlabel('Headcount', fontsize=11, fontweight='bold')
                axes[1, 0].set_ylabel('Occupancy %', fontsize=11, fontweight='bold')
                axes[1, 0].set_title('Occupancy vs Headcount', fontsize=12, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].fill_between(N_values, occ_values, st.session_state.global_target_occupancy, 
                                       where=np.array(occ_values) >= st.session_state.global_target_occupancy, 
                                       alpha=0.2, color='lightcoral')
                
                # Plot 4: Combined View
                axes[1, 1].plot(N_values, p_wait_values, 'b-', linewidth=2, label='P(Wait)')
                axes[1, 1].plot(N_values, sla_values, 'g-', linewidth=2, label='SLA')
                axes[1, 1].plot(N_values, occ_values, 'r-', linewidth=2, label='Occupancy')
                axes[1, 1].set_xlabel('Headcount', fontsize=11, fontweight='bold')
                axes[1, 1].set_ylabel('Percentage', fontsize=11, fontweight='bold')
                axes[1, 1].set_title('Combined View', fontsize=12, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].legend(loc='best')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Performance thresholds
                st.subheader("Performance Thresholds")
                
                # Find key thresholds
                threshold_90_sla = next((N for N, sla in zip(N_values, sla_values) if sla >= 90), None)
                threshold_target_sla = next((N for N, sla in zip(N_values, sla_values) if sla >= st.session_state.global_target_sla), None)
                threshold_target_occ = next((N for N, occ in zip(N_values, occ_values) if occ >= st.session_state.global_target_occupancy), None)
                
                col_thresh1, col_thresh2, col_thresh3 = st.columns(3)
                
                with col_thresh1:
                    if threshold_90_sla is not None:
                        st.metric("Agents for 90% SLA", f"{threshold_90_sla}")
                    else:
                        st.metric("Agents for 90% SLA", "Not achieved")
                
                with col_thresh2:
                    if threshold_target_sla is not None:
                        st.metric(f"Agents for {st.session_state.global_target_sla}% SLA", f"{threshold_target_sla}")
                    else:
                        st.metric(f"Agents for {st.session_state.global_target_sla}% SLA", "Not achieved")
                
                with col_thresh3:
                    if threshold_target_occ is not None:
                        st.metric(f"Agents for {st.session_state.global_target_occupancy}% Occupancy", f"{threshold_target_occ}")
                    else:
                        st.metric(f"Agents for {st.session_state.global_target_occupancy}% Occupancy", "Not achieved")
    
    # ========================
    # TAB 4: RESULTS DASHBOARD
    # ========================
    
    with tab4:
        st.header("📊 RESULTS DASHBOARD")
        
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
                st.metric("Target Occupancy", f"{params['target_occ_pct']}%")
                st.metric("Target SLA", f"{st.session_state.global_target_sla}%")
            
            # Performance summary
            st.subheader("Performance Summary")
            
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            
            with perf_col1:
                occ_gap = params['current_occ']*100 - params['target_occ_pct']
                st.metric("Occupancy Gap", 
                         f"{occ_gap:+.1f}%",
                         "Above target" if occ_gap > 0 else "Below target")
            
            with perf_col2:
                sla_gap = params['current_sl']*100 - st.session_state.global_target_sla
                sla_status = "✓ Good" if sla_gap >= 0 else "⚠️ Needs attention"
                st.metric("SLA Status", sla_status, f"{sla_gap:+.1f}%")
            
            # ... [rest of existing dashboard code] ...
    
    # ========================
    # TAB 5: HOUR-BY-HOUR STAFFING (NEW)
    # ========================
    
    with tab5:
        st.header("📅 Hour-by-Hour Staffing Plan")
        st.markdown("""
        Upload your hourly forecast CSV and generate a detailed staffing plan with risk analysis.
        """)
        
        # File upload section
        uploaded_file = st.file_uploader(
            "📂 Upload Hourly Forecast CSV", 
            type=['csv'],
            help="Upload CSV with columns: Hour, Forecasted_Calls, AHT_Seconds, Shrinkage_Pct"
        )
        
        if uploaded_file is not None:
            try:
                # Read and preview CSV
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_cols = ['Hour', 'Forecasted_Calls', 'AHT_Seconds', 'Shrinkage_Pct']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                    st.write("Current columns:", df.columns.tolist())
                else:
                    st.success("✅ CSV loaded successfully!")
                    
                    # Preview data
                    with st.expander("📋 Preview Uploaded Data", expanded=True):
                        st.dataframe(df.style.format({
                            'Forecasted_Calls': '{:.0f}',
                            'AHT_Seconds': '{:.0f}',
                            'Shrinkage_Pct': '{:.1f}%'
                        }), use_container_width=True)
                    
                    # Analysis parameters
                    st.subheader("⚙️ Analysis Parameters")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        hbh_target_sla = st.slider(
                            "Target SLA %", 
                            70, 99, st.session_state.global_target_sla, 1,
                            key="hbh_sla"
                        )
                    with col2:
                        hbh_target_occ = st.slider(
                            "Target Occupancy %",
                            0, 100, st.session_state.global_target_occupancy, 1,
                            key="hbh_occ"
                        )
                    with col3:
                        hbh_asa = st.slider(
                            "ASA Target (seconds)",
                            5, 300, 30, 5,
                            key="hbh_asa"
                        )
                    
                    if st.button("📊 Generate Hour-by-Hour Analysis", type="primary"):
                        with st.spinner("Calculating staffing requirements..."):
                            # Process each hour
                            results = []
                            precarious_hours = []
                            
                            for _, row in df.iterrows():
                                # Calculate required HC
                                required_hc = calculate_required_hc_for_sla(
                                    row['Forecasted_Calls'],
                                    row['AHT_Seconds'],
                                    hbh_target_sla,
                                    hbh_asa
                                )
                                
                                # Calculate proposed scheduled HC
                                prop_sched_hc = calculate_shrinkage_adjusted_hc(
                                    required_hc,
                                    row['Shrinkage_Pct']
                                )
                                
                                # Calculate metrics
                                occupancy = calculate_occupancy(
                                    row['Forecasted_Calls'],
                                    row['AHT_Seconds'],
                                    prop_sched_hc,
                                    3600  # 1-hour intervals
                                ) * 100
                                
                                traffic_intensity = (row['Forecasted_Calls'] * row['AHT_Seconds'] / 3600)
                                sla = calculate_service_level(
                                    prop_sched_hc,
                                    traffic_intensity,
                                    row['AHT_Seconds'],
                                    hbh_asa
                                ) * 100
                                
                                # Classify risk
                                status, risk = classify_risk(
                                    sla, 
                                    occupancy, 
                                    hbh_target_sla, 
                                    hbh_target_occ
                                )
                                
                                result_row = {
                                    'Hour': f"{int(row['Hour']):02d}:00",
                                    'Forecasted_Calls': row['Forecasted_Calls'],
                                    'AHT_Seconds': row['AHT_Seconds'],
                                    'Shrinkage_Pct': f"{row['Shrinkage_Pct']:.1f}%",
                                    'Required_HC': round(required_hc, 1),
                                    'Prop_Sched_HC': round(prop_sched_hc, 1),
                                    'Occupancy_Pct': round(occupancy, 1),
                                    'SLA_Pct': round(sla, 1),
                                    'Status': status,
                                    'Risk_Level': risk
                                }
                                
                                results.append(result_row)
                                
                                # Track precarious hours
                                if risk in ['High', 'Severe']:
                                    precarious_hours.append({
                                        'Hour': f"{int(row['Hour']):02d}:00",
                                        'SLA_Pct': round(sla, 1),
                                        'Risk_Level': risk,
                                        'Prop_Sched_HC': round(prop_sched_hc, 1),
                                        'Additional_HC_Needed': max(0, round(prop_sched_hc * 1.1 - prop_sched_hc, 1))
                                    })
                            
                            # Create results dataframe
                            results_df = pd.DataFrame(results)
                            
                            # Display main results table
                            st.subheader("📊 Hour-by-Hour Staffing Plan")
                            
                            # Color function for risk levels
                            def color_risk(val):
                                if val == 'Low':
                                    return 'background-color: #d4edda; color: #155724;'
                                elif val == 'Medium':
                                    return 'background-color: #fff3cd; color: #856404;'
                                elif val == 'High':
                                    return 'background-color: #f8d7da; color: #721c24;'
                                elif val == 'Severe':
                                    return 'background-color: #dc3545; color: white; font-weight: bold;'
                                return ''
                            
                            # Format and display table
                            styled_df = results_df.style.applymap(
                                color_risk, subset=['Risk_Level']
                            ).format({
                                'Occupancy_Pct': '{:.1f}%',
                                'SLA_Pct': '{:.1f}%'
                            })
                            
                            st.dataframe(styled_df, use_container_width=True, height=400)
                            
                            # SUMMARY SECTION
                            st.subheader("📈 Summary Statistics")
                            
                            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                            
                            with summary_col1:
                                total_calls = results_df['Forecasted_Calls'].sum()
                                avg_calls = results_df['Forecasted_Calls'].mean()
                                st.metric("Total Calls", f"{total_calls:.0f}")
                                st.caption(f"Avg: {avg_calls:.1f}/hr")
                            
                            with summary_col2:
                                max_hc = results_df['Prop_Sched_HC'].max()
                                avg_hc = results_df['Prop_Sched_HC'].mean()
                                st.metric("Max Staffing", f"{max_hc:.1f}")
                                st.caption(f"Avg: {avg_hc:.1f}")
                            
                            with summary_col3:
                                avg_occ = results_df['Occupancy_Pct'].mean()
                                avg_sla = results_df['SLA_Pct'].mean()
                                st.metric("Avg Occupancy", f"{avg_occ:.1f}%")
                                st.caption(f"Avg SLA: {avg_sla:.1f}%")
                            
                            with summary_col4:
                                risk_hours = len([r for r in results if r['Risk_Level'] in ['High', 'Severe']])
                                total_hours = len(results)
                                st.metric("Risk Hours", f"{risk_hours}/{total_hours}")
                                st.caption(f"{risk_hours/total_hours*100:.1f}% of hours")

                            # ---- START PLOTLY CODE ----
                            
                            # Create interactive Plotly charts
                            fig = make_subplots(
                                rows=2, cols=2,                
                                subplot_titles=('Volume vs Staffing', 'Occupancy Trend', 'SLA Performance', 'Risk Heatmap'),
                                vertical_spacing=0.15,
                                horizontal_spacing=0.15
                            )
                            
                            # Chart 1: Volume vs Staffing
                            fig.add_trace(
                                go.Bar(
                                    x=results_df['Hour'],
                                    y=results_df['Forecasted_Calls'],
                                    name='Volume',
                                    marker_color='lightblue'
                                ),
                                row=1, col=1
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=results_df['Hour'],
                                    y=results_df['Prop_Sched_HC'],
                                    name='Staffing',
                                    yaxis='y2',
                                    line=dict(color='red', width=2)
                                ),
                                row=1, col=1
                            )
                            
                            # Add secondary y-axis for chart 1
                            fig.update_yaxes(title_text="Volume", row=1, col=1)
                            fig.update_yaxes(title_text="Staffing", secondary_y=True, row=1, col=1)
                            
                            # Chart 2: Occupancy Trend
                            fig.add_trace(
                                go.Scatter(
                                    x=results_df['Hour'],
                                    y=results_df['Occupancy_Pct'],
                                    name='Occupancy',
                                    line=dict(color='green', width=3),
                                    fill='tozeroy'
                                ),
                                row=1, col=2
                            )
                            fig.add_hline(y=hbh_target_occ, line_dash="dash", line_color="orange", row=1, col=2)
                            fig.update_yaxes(title_text="Occupancy (%)", row=1, col=2)
                            
                            # Chart 3: SLA Performance
                            fig.add_trace(
                                go.Scatter(
                                    x=results_df['Hour'],
                                    y=results_df['SLA_Pct'],
                                    name='SLA',
                                    line=dict(color='purple', width=3),
                                    mode='lines+markers'
                                ),
                                row=2, col=1
                            )
                            fig.add_hline(y=hbh_target_sla, line_dash="dash", line_color="darkgreen", row=2, col=1)
                            fig.update_yaxes(title_text="SLA (%)", row=2, col=1)
                            
                            # Chart 4: Risk Heatmap
                            risk_colors = {'Low': 0, 'Medium': 1, 'High': 2, 'Severe': 3}
                            risk_numeric = [risk_colors[r] for r in results_df['Risk_Level']]
                            
                            fig.add_trace(
                                go.Heatmap(
                                    x=results_df['Hour'],
                                    y=['Risk'],
                                    z=[risk_numeric],  # 2D array for heatmap
                                    colorscale=[[0, 'green'], [0.3, 'yellow'], [0.6, 'orange'], [1, 'red']],
                                    showscale=True,
                                    colorbar=dict(title="Risk Level", tickvals=[0, 1, 2, 3], ticktext=['Low', 'Medium', 'High', 'Severe']),
                                    hovertext=results_df['Risk_Level'],
                                    hoverinfo='text'
                                ),
                                row=2, col=2
                            )
                            fig.update_yaxes(title_text="Risk", row=2, col=2)
                            
                            # Update layout
                            fig.update_layout(
                                height=600, 
                                showlegend=False,
                                title_text="Hour-by-Hour Analysis Dashboard",
                                title_x=0.5
                            )
                            
                            # Update x-axis labels for all subplots
                            fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
                            fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
                            fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
                            fig.update_xaxes(title_text="Hour of Day", row=1, col=2)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # END PLOTLY CODE ----
                            
                            # PRECARIOUS HOURS RECOMMENDATION PANEL
                            if precarious_hours:
                                st.subheader("⚠️ Precarious Hours - Action Required")
                                
                                # Create DataFrame for precarious hours
                                precarious_df = pd.DataFrame(precarious_hours)
                                
                                # Display with recommendations
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.dataframe(
                                        precarious_df.style.applymap(
                                            lambda x: 'background-color: #f8d7da; color: #721c24; font-weight: bold;' 
                                            if x == 'Severe' else 'background-color: #fff3cd; color: #856404;',
                                            subset=['Risk_Level']
                                        ),
                                        use_container_width=True
                                    )
                                
                                with col2:
                                    st.info("""
                                    **💡 Recommendations:**
                                    - **Severe Risk**: Add 10-15% more agents
                                    - **High Risk**: Add 5-10% more agents
                                    - **Medium Risk**: Monitor closely
                                    - Consider shift adjustments
                                    - Review break schedules
                                    """)
                                
                                # Detailed recommendations by hour
                                st.markdown("#### 📋 Hour-by-Hour Action Plan")
                                
                                for hour in precarious_hours:
                                    if hour['Risk_Level'] == 'Severe':
                                        st.error(f"""
                                        **❌ {hour['Hour']} - SEVERE RISK**
                                        - Current SLA: {hour['SLA_Pct']}% (Critical)
                                        - Recommended: Add {hour['Additional_HC_Needed']} agents
                                        - Alternative: Reduce AHT by 10-15%
                                        """)
                                    elif hour['Risk_Level'] == 'High':
                                        st.warning(f"""
                                        **⚠️ {hour['Hour']} - HIGH RISK**
                                        - Current SLA: {hour['SLA_Pct']}% (Below Target)
                                        - Recommended: Add {hour['Additional_HC_Needed']} agents
                                        - Consider: Cross-training or queue prioritization
                                        """)
                            
                            # EXPORT SECTION
                            st.subheader("📤 Export Options")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Export to CSV
                                csv = results_df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="hourly_staffing_plan.csv" class="button">📥 Download Full CSV</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                                if precarious_hours:
                                    precarious_csv = pd.DataFrame(precarious_hours).to_csv(index=False)
                                    b64_precarious = base64.b64encode(precarious_csv.encode()).decode()
                                    href_precarious = f'<a href="data:file/csv;base64,{b64_precarious}" download="precarious_hours.csv" class="button">📥 Download Precarious Hours CSV</a>'
                                    st.markdown(href_precarious, unsafe_allow_html=True)
                            
                            with col2:
                                # Summary export
                                summary_data = {
                                    'Metric': ['Total Calls', 'Avg Calls/Hour', 'Max Staffing', 'Avg Staffing', 'Avg Occupancy', 'Avg SLA', 'Risk Hours', 'Target SLA', 'Target Occupancy'],
                                    'Value': [total_calls, avg_calls, max_hc, avg_hc, avg_occ, avg_sla, f"{risk_hours}/{total_hours}", f"{hbh_target_sla}%", f"{hbh_target_occ}%"]
                                }
                                summary_df = pd.DataFrame(summary_data)
                                summary_csv = summary_df.to_csv(index=False)
                                b64_summary = base64.b64encode(summary_csv.encode()).decode()
                                href_summary = f'<a href="data:file/csv;base64,{b64_summary}" download="analysis_summary.csv">📥 Download Summary CSV</a>'
                                st.markdown(href_summary, unsafe_allow_html=True)
                            
                            st.caption("Note: Download files for detailed analysis and reporting")
                            
            except Exception as e: 
                st.error(f"Error in analysis: {e}")
        
        else:
            # Show upload instructions when no file is uploaded
            st.info("""
            ### 📋 Expected CSV Format:
            
            Create a CSV file with these columns:
            
            ```
            Hour,Forecasted_Calls,AHT_Seconds,Shrinkage_Pct
            8,45,390,15
            9,67,390,15
            10,89,390,15
            11,102,390,15
            12,95,400,20
            13,87,410,15
            14,110,380,10
            15,98,390,15
            16,76,395,15
            17,54,400,20
            ```
            
            **Column Definitions:**
            - **Hour**: Hour of day (0-23)
            - **Forecasted_Calls**: Expected call volume for that hour
            - **AHT_Seconds**: Average Handle Time in seconds
            - **Shrinkage_Pct**: Percentage of time agents are unavailable (breaks, meetings, etc.)
            
            [Download Sample CSV](https://example.com/sample.csv)
            """)
    
    # ========================
    # TAB 6: TERMINOLOGY GUIDE (NEW)
    # ========================
    
    with tab6:
        st.header("📖 Terminology Guide")
        st.markdown("""
        This guide explains all terms used throughout the analysis tool.
        """)
        
        # Create tabs within terminology guide
        term_tab1, term_tab2, term_tab3 = st.tabs([
            "📊 General Metrics",
            "📅 Hour-by-Hour Analysis",
            "⚙️ Optimization Terms"
        ])
        
        with term_tab1:
            st.subheader("General Call Center Metrics")
            
            terms_general = {
                "AHT (Average Handle Time)": "The average duration of a call from start to finish, including talk time and after-call work.",
                "ASA (Average Speed of Answer)": "The average time callers wait in queue before being answered by an agent.",
                "SLA (Service Level Agreement)": "The percentage of calls answered within a specified time threshold (e.g., 90% within 30 seconds).",
                "Occupancy": "The percentage of time agents are actively handling calls versus waiting for calls.",
                "Shrinkage": "The percentage of paid time when agents are not available to handle calls (breaks, meetings, training, etc.).",
                "Erlang": "A unit of telecommunications traffic measurement. One Erlang = 60 minutes of call traffic.",
                "Traffic Intensity": "The volume of call traffic expressed in Erlangs. Calculated as (Calls × AHT) / 3600.",
                "Probability of Wait (P-wait)": "The likelihood that an incoming call will have to wait in queue before being answered."
            }
            
            for term, definition in terms_general.items():
                st.markdown(f"**{term}**")
                st.markdown(f"*{definition}*")
                st.markdown("---")
        
        with term_tab2:
            st.subheader("Hour-by-Hour Analysis Terms")
            
            st.markdown("""
            ### CSV Upload Format Terms:
            """)
            
            csv_terms = {
                "Hour": "The hour of day (0-23) for which forecasts apply. Example: '8' represents 8:00-9:00 AM.",
                "Forecasted_Calls": "The expected number of incoming calls during that hour.",
                "AHT_Seconds": "The expected Average Handle Time for calls during that hour, in seconds.",
                "Shrinkage_Pct": "The expected percentage of time agents will be unavailable during that hour (entered as 15 for 15%)."
            }
            
            for term, definition in csv_terms.items():
                st.markdown(f"**{term}**")
                st.markdown(f"*{definition}*")
                st.markdown("---")
            
            st.markdown("""
            ### Analysis Output Terms:
            """)
            
            output_terms = {
                "Required HC": "The minimum headcount calculated from Erlang C formula to handle the forecasted volume at target SLA.",
                "Prop Sched HC": "Proposed Scheduled Headcount - the actual number of agents to schedule, adjusted for shrinkage.",
                "Occupancy %": "The calculated utilization percentage of scheduled agents for that hour.",
                "SLA %": "The predicted service level percentage based on the proposed staffing.",
                "Status": "Visual indicator of performance: ✅ Optimal, ⚠️ Marginal, ❌ Critical.",
                "Risk Level": "Classification of risk: Low, Medium, High, or Severe based on SLA and occupancy targets."
            }
            
            for term, definition in output_terms.items():
                st.markdown(f"**{term}**")
                st.markdown(f"*{definition}*")
                st.markdown("---")
            
            st.markdown("""
            ### Risk Classification:
            """)
            
            risk_table = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High', 'Severe'],
                'SLA Range': ['≥ Target + 5%', 'Target to Target + 5%', 'Target - 10% to Target', '< Target - 10%'],
                'Action Required': ['None - Optimal', 'Monitor - Tight', 'Add Staff - Below Target', 'Immediate Action - Critical']
            })
            
            st.table(risk_table)
        
        with term_tab3:
            st.subheader("Optimization Terms")
            
            opt_terms = {
                "Target SLA": "The service level percentage you aim to achieve across all intervals.",
                "Target Occupancy": "The optimal utilization percentage for agents, balancing efficiency with service quality.",
                "Optimization Strategy": [
                    "**SLA-Optimized**: Prioritizes meeting service level targets, may result in lower occupancy.",
                    "**Occupancy-Targeted**: Prioritizes achieving target occupancy, may compromise on service level.",
                    "**Balanced Solution**: Finds the best trade-off between SLA and occupancy targets."
                ],
                "Precarious Hours": "Hours where staffing levels are marginal or insufficient to meet targets, requiring attention.",
                "Agent Surplus/Deficit": "The difference between scheduled agents and the traffic intensity (Erlangs).",
                "Efficiency Score": "A composite score (0-100) balancing both occupancy and service level performance."
            }
            
            for term, definition in opt_terms.items():
                st.markdown(f"**{term}**")
                if isinstance(definition, list):
                    for item in definition:
                        st.markdown(f"• {item}")
                else:
                    st.markdown(f"*{definition}*")
                st.markdown("---")
            
            st.markdown("""
            ### Mathematical Formulas:
            """)
            
            formulas = {
                "Erlang C Probability": "P(wait) = (Aᴺ/N!) × (N/(N-A)) / [Σ(Aⁱ/i!) + (Aᴺ/N!) × (N/(N-A))]",
                "Service Level": "SLA = 1 - P(wait) × exp(-(N-A) × T/AHT)",
                "Occupancy": "Occ = (Volume × AHT) / (Headcount × Interval)",
                "Shrinkage Adjustment": "Scheduled HC = Required HC / (1 - Shrinkage%)"
            }
            
            for formula_name, formula in formulas.items():
                st.markdown(f"**{formula_name}:**")
                st.code(formula, language='latex')
                st.markdown("---")
        
        # Quick reference
        # Quick reference
        st.markdown("""
        ---
        ### 🚀 Quick Reference
        
        **Typical Trade-off Ranges:**
        
        | Service Level Target | Typical Occupancy Range | Use Case |
        |----------------------|-------------------------|----------|
        | 90-95% (High Quality) | 60-75% | Premium/Service-focused centers |
        | 85-90% (Balanced) | 70-80% | Standard call centers |
        | 80-85% (Efficient) | 75-85% | Cost-sensitive operations |
        
        **Warning Signs:**
        - SLA < 80% with Occupancy > 85% → Overworked agents, poor service
        - SLA > 95% with Occupancy < 60% → Underutilized resources, high costs
        
        **Key Ratios:**
        - Agent-to-Traffic: Aim for N/A ≈ 1.1-1.3 for balanced operations
        - Service Factor: SLA × Occupancy ÷ 100 (aim for 60-75)
        """)
    
    # ========================
    # FOOTER
    # ========================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Equiserve Occupancy Analysis Tool v2.5 | Based on Erlang C Queueing Theory</p>
        <p><small>Note: Results are estimates based on mathematical models. Real-world factors may vary.</small></p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
