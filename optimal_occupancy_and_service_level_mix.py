# optimal_occupancy_and_service_level_mix.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import math

st.set_page_config(layout="wide", page_title="Call Center Optimization Suite")

# ========================
# CALCULATION ENGINE - FIXED
# ========================

def calculate_occupancy(volume, AHT, headcount, interval_seconds):
    """Calculate occupancy rate"""
    if headcount == 0:
        return 0
    return (volume * AHT / 3600) / (headcount * interval_seconds / 3600)

def calculate_erlang_c(m, a, ASA, AHT):
    """Calculate Erlang C formula for service level - FIXED VERSION"""
    # Convert m to integer for range calculation
    m_int = int(math.ceil(m))  # Round up to nearest integer
    
    if m_int <= a:
        return 0.0
    
    try:
        # Calculate Erlang C probability using logarithms for numerical stability
        sum_term = 0
        for i in range(m_int):
            # Use logarithms to avoid large numbers
            log_term = i * math.log(a) - math.lgamma(i + 1)
            sum_term += math.exp(log_term)
        
        # Calculate numerator with logarithms
        log_numerator = m_int * math.log(a) - math.lgamma(m_int + 1)
        numerator = math.exp(log_numerator)
        
        denominator = (numerator * m_int / (m_int - a)) + sum_term
        
        if denominator == 0:
            return 0.0
            
        pw = numerator / denominator
        
        # Service level calculation
        if (m_int - a) <= 0:
            return 0.0
            
        exponent = -(m_int - a) * ASA / AHT
        sla = 1 - pw * math.exp(exponent)
        return max(0.0, min(1.0, sla))
    
    except (OverflowError, ValueError):
        # Fallback to approximation for extreme values
        if m_int > 100:  # For large headcounts, use approximation
            return min(1.0, 1 - math.exp(-(m_int - a) * ASA / AHT))
        return 0.5  # Default fallback

def calculate_service_level(headcount, volume, AHT, ASA):
    """Calculate service level using Erlang C"""
    traffic_intensity = volume * AHT / 3600
    return calculate_erlang_c(headcount, traffic_intensity, ASA, AHT)

def calculate_callsmaxocc(headcount, AHT, target_occ, interval_seconds):
    """Calculate maximum calls per hour at target occupancy"""
    return (headcount * interval_seconds / AHT) * target_occ

def calculate_maxcap_per_agent(AHT, interval_seconds, target_occ):
    """Calculate maximum capacity per agent"""
    return (interval_seconds / AHT) * target_occ

def find_optimal_headcount(volume, AHT, ASA, target_sla, target_occ, interval_minutes):
    """Find headcount that best balances occupancy and SLA targets"""
    interval_seconds = interval_minutes * 60
    
    best_score = float('inf')
    best_result = None
    all_results = []
    
    # Search through headcounts (using integer steps for better performance)
    for hc in np.arange(1, 50, 0.5):
        # Calculate occupancy and SLA
        occ = calculate_occupancy(volume, AHT, hc, interval_seconds) * 100
        sla = calculate_service_level(hc, volume, AHT, ASA) * 100
        
        # Handle NaN or None values
        if sla is None or np.isnan(sla):
            sla = 0.0
            
        # Calculate how far from targets
        occ_distance = abs(occ - target_occ * 100)
        sla_distance = abs(sla - target_sla)
        
        # Combined score (SLA weighted more heavily)
        score = occ_distance + sla_distance * 2
        
        all_results.append({
            'headcount': hc,
            'occupancy': occ,
            'service_level': sla,
            'score': score,
            'occ_distance': occ_distance,
            'sla_distance': sla_distance
        })
        
        if score < best_score:
            best_score = score
            best_result = all_results[-1]
    
    return best_result, all_results

# ========================
# VISUALIZATION FUNCTIONS
# ========================

def create_headcount_tradeoff_plot(volume, AHT, ASA, interval_minutes, optimal_hc, hc_for_occ, hc_for_sla, target_occ, target_sla):
    """Create headcount vs performance tradeoff plot"""
    interval_seconds = interval_minutes * 60
    
    hc_values = np.linspace(1, max(30, optimal_hc * 1.5), 100)
    occ_values = []
    sla_values = []
    
    for hc in hc_values:
        occ = calculate_occupancy(volume, AHT, hc, interval_seconds) * 100
        occ_values.append(occ)
        
        sla = calculate_service_level(hc, volume, AHT, ASA) * 100
        if sla is None or np.isnan(sla):
            sla = 0.0
        sla_values.append(sla)
    
    fig = go.Figure()
    
    # Add occupancy line
    fig.add_trace(go.Scatter(
        x=hc_values, y=occ_values,
        mode='lines',
        name='Occupancy (%)',
        line=dict(color='blue', width=3),
        hovertemplate='Headcount: %{x:.1f}<br>Occupancy: %{y:.1f}%'
    ))
    
    # Add service level line
    fig.add_trace(go.Scatter(
        x=hc_values, y=sla_values,
        mode='lines',
        name='Service Level (%)',
        line=dict(color='green', width=3),
        hovertemplate='Headcount: %{x:.1f}<br>Service Level: %{y:.1f}%'
    ))
    
    # Add target lines
    fig.add_hline(y=target_occ, line_dash="dash", line_color="blue", 
                  opacity=0.5, annotation_text=f"Target Occupancy: {target_occ}%")
    fig.add_hline(y=target_sla, line_dash="dash", line_color="green", 
                  opacity=0.5, annotation_text=f"Target SLA: {target_sla}%")
    
    # Add optimal point
    optimal_occ = calculate_occupancy(volume, AHT, optimal_hc, interval_seconds) * 100
    optimal_sla = calculate_service_level(optimal_hc, volume, AHT, ASA) * 100
    if optimal_sla is None or np.isnan(optimal_sla):
        optimal_sla = 0.0
    
    fig.add_trace(go.Scatter(
        x=[optimal_hc], y=[optimal_occ],
        mode='markers',
        name='Optimal Point',
        marker=dict(color='red', size=15, symbol='star'),
        hovertemplate=f'Optimal: {optimal_hc:.1f} agents<br>Occ: {optimal_occ:.1f}%<br>SLA: {optimal_sla:.1f}%'
    ))
    
    # Update layout
    fig.update_layout(
        title="Headcount Optimization: Occupancy vs Service Level Trade-off",
        xaxis_title="Headcount (Agents)",
        yaxis_title="Performance (%)",
        hovermode="x unified",
        height=500,
        template="plotly_white"
    )
    
    return fig

def create_capacity_comparison_plot(volume, optimal_hc, AHT, target_occ, interval_minutes, callsmaxocc_at_target):
    """Create capacity comparison bar chart"""
    interval_seconds = interval_minutes * 60
    maxcap_per_agent = calculate_maxcap_per_agent(AHT, interval_seconds, target_occ/100)
    team_capacity = maxcap_per_agent * optimal_hc
    
    categories = ['Current Volume', f'Max at {target_occ}% Occ', 'Team Capacity']
    values = [volume, callsmaxocc_at_target, team_capacity]
    
    # Calculate occupancy at each point
    occ_at_volume = calculate_occupancy(volume, AHT, optimal_hc, interval_seconds) * 100
    occ_at_target = target_occ
    occ_at_capacity = calculate_occupancy(team_capacity, AHT, optimal_hc, interval_seconds) * 100
    
    occupancy_values = [occ_at_volume, occ_at_target, occ_at_capacity]
    
    fig = go.Figure()
    
    # Add bars for capacity
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        name='Capacity (calls/hour)',
        marker_color=['blue', 'red', 'green'],
        text=[f'{v:.1f}<br>{o:.1f}% Occ' for v, o in zip(values, occupancy_values)],
        textposition='outside',
        hovertemplate='%{x}<br>Capacity: %{y:.1f} calls/hour<br>Occupancy: %{customdata:.1f}%',
        customdata=occupancy_values
    ))
    
    # Add current volume reference line
    fig.add_hline(y=volume, line_dash="dot", line_color="blue", opacity=0.3)
    
    fig.update_layout(
        title=f"Capacity Comparison (Optimal: {optimal_hc:.1f} agents)",
        yaxis_title="Calls per Hour",
        showlegend=False,
        height=500,
        template="plotly_white"
    )
    
    return fig

def create_sensitivity_heatmap(volume, AHT, ASA, interval_minutes, optimal_hc):
    """Create sensitivity analysis heatmap"""
    interval_seconds = interval_minutes * 60
    
    # Vary volume and AHT
    volume_range = np.linspace(max(1, volume * 0.5), volume * 1.5, 15)
    AHT_range = np.linspace(max(60, AHT * 0.7), AHT * 1.3, 15)
    
    occupancy_grid = np.zeros((len(AHT_range), len(volume_range)))
    sla_grid = np.zeros((len(AHT_range), len(volume_range)))
    
    for i, aht_val in enumerate(AHT_range):
        for j, vol_val in enumerate(volume_range):
            occupancy_grid[i, j] = calculate_occupancy(vol_val, aht_val, optimal_hc, interval_seconds) * 100
            sla_val = calculate_service_level(optimal_hc, vol_val, aht_val, ASA) * 100
            if sla_val is None or np.isnan(sla_val):
                sla_val = 0.0
            sla_grid[i, j] = sla_val
    
    # Create occupancy heatmap
    fig1 = go.Figure(data=go.Heatmap(
        z=occupancy_grid,
        x=volume_range,
        y=AHT_range,
        colorscale='RdBu',
        zmid=80,
        colorbar=dict(title="Occupancy %"),
        hovertemplate='Volume: %{x:.1f}<br>AHT: %{y:.1f}s<br>Occupancy: %{z:.1f}%<extra></extra>'
    ))
    
    fig1.update_layout(
        title=f"Occupancy Sensitivity (HC: {optimal_hc:.1f})",
        xaxis_title="Call Volume (calls/hour)",
        yaxis_title="AHT (seconds)",
        height=400,
        template="plotly_white"
    )
    
    # Create SLA heatmap
    fig2 = go.Figure(data=go.Heatmap(
        z=sla_grid,
        x=volume_range,
        y=AHT_range,
        colorscale='RdYlGn',
        zmid=90,
        colorbar=dict(title="SLA %"),
        hovertemplate='Volume: %{x:.1f}<br>AHT: %{y:.1f}s<br>SLA: %{z:.1f}%<extra></extra>'
    ))
    
    fig2.update_layout(
        title=f"Service Level Sensitivity (HC: {optimal_hc:.1f})",
        xaxis_title="Call Volume (calls/hour)",
        yaxis_title="AHT (seconds)",
        height=400,
        template="plotly_white"
    )
    
    return fig1, fig2

# ========================
# MAIN APP INTERFACE
# ========================

def main():
    st.title("📊 Advanced Call Center Optimization Suite")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("⚙️ Simulation Parameters")
        
        volume = st.slider(
            "Call Volume (calls/hour)",
            min_value=1,
            max_value=500,
            value=40,
            step=1,
            help="Number of calls arriving per hour"
        )
        
        AHT = st.slider(
            "Average Handle Time (seconds)",
            min_value=60,
            max_value=1800,
            value=390,
            step=10,
            help="Average time to handle a call including wrap-up"
        )
        
        ASA = st.slider(
            "ASA Target (seconds)",
            min_value=5,
            max_value=300,
            value=30,
            step=5,
            help="Acceptable Service Time for service level calculation"
        )
        
        target_sla = st.slider(
            "Target Service Level (%)",
            min_value=50,
            max_value=99,
            value=90,
            step=1,
            help="Desired percentage of calls answered within ASA"
        )
        
        target_occ = st.slider(
            "Target Occupancy (%)",
            min_value=50,
            max_value=95,
            value=80,
            step=1,
            help="Desired agent utilization rate"
        )
        
        interval = st.selectbox(
            "Interval Length (minutes)",
            options=[15, 30, 60],
            index=2,
            help="Time period for calculations"
        )
        
        st.divider()
        
        if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
            st.session_state.run_analysis = True
        
        if st.button("🔄 Reset", type="secondary", use_container_width=True):
            st.session_state.run_analysis = False
            st.rerun()
    
    # Main content area
    if st.session_state.get('run_analysis', False):
        # Run optimization
        interval_seconds = interval * 60
        
        with st.spinner("Running Erlang C calculations..."):
            try:
                optimal_result, all_results = find_optimal_headcount(
                    volume, AHT, ASA, target_sla, target_occ, interval
                )
                
                if optimal_result is None:
                    st.error("Could not find optimal solution. Please adjust parameters.")
                    return
                
                # Calculate additional metrics
                optimal_hc = optimal_result['headcount']
                callsmaxocc_at_target = calculate_callsmaxocc(
                    optimal_hc, AHT, target_occ/100, interval_seconds
                )
                
                # Calculate headcount for individual targets
                if target_occ > 0:
                    hc_for_target_occ = (volume * AHT / 3600) / ((target_occ/100) * interval_seconds / 3600)
                else:
                    hc_for_target_occ = 0
                
                # Find headcount for target SLA using binary search
                hc_low, hc_high = 1, 50
                for _ in range(25):
                    hc_mid = (hc_low + hc_high) / 2
                    sla_mid = calculate_service_level(hc_mid, volume, AHT, ASA) * 100
                    if sla_mid is None or np.isnan(sla_mid):
                        sla_mid = 0.0
                    if sla_mid >= target_sla:
                        hc_high = hc_mid
                    else:
                        hc_low = hc_mid
                hc_for_target_sla = (hc_low + hc_high) / 2
                
            except Exception as e:
                st.error(f"Calculation error: {str(e)}")
                st.info("Try adjusting parameters (especially reducing volume or increasing AHT)")
                return
        
        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Summary", "📈 Trade-off Analysis", "🔥 Sensitivity", 
            "📋 Detailed Results", "💡 Recommendations"
        ])
        
        with tab1:
            st.header("Optimization Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Optimal Headcount",
                    f"{optimal_hc:.1f}",
                    delta=f"{(optimal_hc - hc_for_target_occ):+.1f} vs Occ-only"
                )
            
            with col2:
                st.metric(
                    "Resulting Occupancy",
                    f"{optimal_result['occupancy']:.1f}%",
                    delta=f"{optimal_result['occupancy'] - target_occ:+.1f}%"
                )
            
            with col3:
                sla_value = optimal_result['service_level']
                if sla_value is None or np.isnan(sla_value):
                    sla_value = 0.0
                st.metric(
                    "Resulting Service Level",
                    f"{sla_value:.1f}%",
                    delta=f"{sla_value - target_sla:+.1f}%"
                )
            
            with col4:
                if hc_for_target_occ > 0:
                    efficiency = optimal_hc / max(hc_for_target_occ, hc_for_target_sla) * 100
                else:
                    efficiency = 100
                st.metric(
                    "Solution Efficiency",
                    f"{efficiency:.1f}%",
                    help="How close to ideal scenario"
                )
            
            # Key insights box
            sla_display = sla_value if sla_value is not None else 0.0
            st.info(f"""
            **Key Finding:** {optimal_hc:.1f} agents provides the best balance:
            - **{optimal_result['occupancy']:.1f}% occupancy** (Target: {target_occ}%)
            - **{sla_display:.1f}% service level** (Target: {target_sla}%)
            
            This solution is **{efficiency:.1f}% efficient** compared to meeting each target individually.
            """)
        
        with tab2:
            st.header("Trade-off Analysis")
            
            try:
                # Trade-off plot
                fig_tradeoff = create_headcount_tradeoff_plot(
                    volume, AHT, ASA, interval, optimal_hc, 
                    hc_for_target_occ, hc_for_target_sla, target_occ, target_sla
                )
                st.plotly_chart(fig_tradeoff, use_container_width=True)
                
                # Capacity comparison
                fig_capacity = create_capacity_comparison_plot(
                    volume, optimal_hc, AHT, target_occ, interval, callsmaxocc_at_target
                )
                st.plotly_chart(fig_capacity, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Visualization error: {str(e)}")
                st.info("Some charts may not render properly with extreme parameters")
            
            # Detailed comparison table
            occ_sla_only = calculate_occupancy(volume, AHT, hc_for_target_sla, interval_seconds) * 100
            sla_occ_only = calculate_service_level(hc_for_target_occ, volume, AHT, ASA) * 100
            if sla_occ_only is None or np.isnan(sla_occ_only):
                sla_occ_only = 0.0
            
            comparison_data = pd.DataFrame({
                'Scenario': ['Optimal Balance', 'Occupancy-Focused', 'SLA-Focused'],
                'Headcount': [optimal_hc, hc_for_target_occ, hc_for_target_sla],
                'Occupancy': [
                    optimal_result['occupancy'],
                    target_occ,
                    occ_sla_only
                ],
                'Service Level': [
                    sla_display,
                    sla_occ_only,
                    target_sla
                ],
                'Efficiency Score': [
                    optimal_result['score'],
                    abs(target_occ - target_occ) + abs(sla_occ_only - target_sla)*2,
                    abs(occ_sla_only - target_occ) + abs(target_sla - target_sla)*2
                ]
            })
            
            st.dataframe(comparison_data.style.format({
                'Headcount': '{:.1f}',
                'Occupancy': '{:.1f}%',
                'Service Level': '{:.1f}%',
                'Efficiency Score': '{:.1f}'
            }).highlight_min(subset=['Efficiency Score'], color='lightgreen'), 
            use_container_width=True)
        
        with tab3:
            st.header("Sensitivity Analysis")
            
            try:
                # Heatmaps
                fig_occ_heat, fig_sla_heat = create_sensitivity_heatmap(
                    volume, AHT, ASA, interval, optimal_hc
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_occ_heat, use_container_width=True)
                
                with col2:
                    st.plotly_chart(fig_sla_heat, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate heatmaps: {str(e)}")
                st.info("Try with less extreme parameter ranges")
            
            # What-if scenarios
            st.subheader("What-if Scenarios")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                volume_change = st.slider("Volume Change (%)", -50, 50, 0, 5, key="vol_change")
                new_volume = volume * (1 + volume_change/100)
                new_occ = calculate_occupancy(new_volume, AHT, optimal_hc, interval_seconds) * 100
                new_sla = calculate_service_level(optimal_hc, new_volume, AHT, ASA) * 100
                if new_sla is None or np.isnan(new_sla):
                    new_sla = 0.0
                st.metric("New Occupancy", f"{new_occ:.1f}%", delta=f"{new_occ - optimal_result['occupancy']:+.1f}%")
            
            with col2:
                aht_change = st.slider("AHT Change (%)", -30, 30, 0, 5, key="aht_change")
                new_aht = AHT * (1 + aht_change/100)
                new_occ = calculate_occupancy(volume, new_aht, optimal_hc, interval_seconds) * 100
                new_sla = calculate_service_level(optimal_hc, volume, new_aht, ASA) * 100
                if new_sla is None or np.isnan(new_sla):
                    new_sla = 0.0
                st.metric("New Service Level", f"{new_sla:.1f}%", delta=f"{new_sla - sla_display:+.1f}%")
            
            with col3:
                hc_change = st.slider("Headcount Change", -5, 5, 0, 1, key="hc_change")
                new_hc = optimal_hc + hc_change
                new_occ = calculate_occupancy(volume, AHT, new_hc, interval_seconds) * 100
                new_sla = calculate_service_level(new_hc, volume, AHT, ASA) * 100
                if new_sla is None or np.isnan(new_sla):
                    new_sla = 0.0
                st.metric("New Efficiency", f"{(new_occ/100)*(new_sla/100)*100:.1f}%")
        
        with tab4:
            st.header("Detailed Simulation Results")
            
            # Convert all results to DataFrame
            results_df = pd.DataFrame(all_results)
            results_df.columns = ['Headcount', 'Occupancy (%)', 'Service Level (%)', 'Score', 'Occ Distance', 'SLA Distance']
            
            # Format for display
            display_df = results_df.copy()
            display_df['Headcount'] = display_df['Headcount'].round(1)
            display_df['Occupancy (%)'] = display_df['Occupancy (%)'].round(1)
            display_df['Service Level (%)'] = display_df['Service Level (%)'].round(1)
            display_df['Score'] = display_df['Score'].round(1)
            
            # Highlight optimal row
            def highlight_optimal(row):
                if abs(row['Headcount'] - optimal_hc) < 0.1:
                    return ['background-color: lightgreen'] * len(row)
                return [''] * len(row)
            
            st.dataframe(display_df.style.apply(highlight_optimal, axis=1), 
                        use_container_width=True, height=400)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with tab5:
            st.header("Actionable Recommendations")
            
            # Generate insights based on analysis
            if callsmaxocc_at_target > 0:
                volume_ratio = volume / callsmaxocc_at_target
            else:
                volume_ratio = 1
            
            if volume_ratio > 1.1:
                st.error(f"""
                ## ⚠️ **CRITICAL: Overcapacity Situation**
                
                **Current volume ({volume}) exceeds capacity at target occupancy ({callsmaxocc_at_target:.1f}) by {(volume_ratio-1)*100:.0f}%**
                
                ### 🎯 **Immediate Actions:**
                1. **Increase headcount** to at least {hc_for_target_occ:.1f} agents
                2. **Reduce AHT** by {(AHT - (callsmaxocc_at_target * AHT / volume)):.0f} seconds
                3. **Implement call deflection** to reduce volume by {volume - callsmaxocc_at_target:.1f} calls/hour
                
                ### 📊 **Impact if no action:**
                - Occupancy will exceed {target_occ + 10}% (agent burnout risk)
                - Service level will drop below {max(50, target_sla - 20)}% (customer dissatisfaction)
                """)
            elif volume_ratio < 0.9:
                st.success(f"""
                ## ✅ **Opportunity: Underutilized Capacity**
                
                **You have {(callsmaxocc_at_target - volume):.1f} calls/hour of available capacity**
                
                ### 🎯 **Growth Opportunities:**
                1. **Handle additional volume** up to {callsmaxocc_at_target:.1f} calls/hour
                2. **Improve service level** by reducing headcount to {hc_for_target_sla:.1f} agents
                3. **Cross-train agents** for additional skills
                
                ### 💰 **Potential Revenue Impact:**
                - Additional {(callsmaxocc_at_target - volume) * 50:.0f} revenue/hour (at $50/call)
                - {((callsmaxocc_at_target - volume) / volume * 100):.1f}% growth potential
                """)
            else:
                st.info(f"""
                ## ⚖️ **Optimal Balance Achieved**
                
                **Your current configuration is well-balanced for targets**
                
                ### 🎯 **Maintenance Actions:**
                1. **Monitor volume fluctuations** within ±20% of current
                2. **Keep AHT stable** at {AHT} ±30 seconds
                3. **Maintain headcount** at {optimal_hc:.1f} ±1 agents
                
                ### 🔄 **Continuous Improvement:**
                - Regularly review ASA target (current: {ASA} seconds)
                - Consider gradual occupancy increase to {min(85, target_occ + 5)}%
                - Explore AHT reduction opportunities
                """)
            
            # ROI calculation
            st.divider()
            st.subheader("💰 ROI Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                revenue_per_call = st.number_input("Revenue per call ($)", 10, 200, 50, 10, key="revenue")
            
            with col2:
                cost_per_agent_hour = st.number_input("Cost per agent/hour ($)", 20, 100, 40, 5, key="cost")
            
            with col3:
                hours_per_day = st.number_input("Operating hours/day", 8, 24, 10, 1, key="hours")
            
            daily_revenue = volume * revenue_per_call * hours_per_day
            daily_cost = optimal_hc * cost_per_agent_hour * hours_per_day
            daily_profit = daily_revenue - daily_cost
            
            st.metric("Estimated Daily Profit", f"${daily_profit:,.0f}", 
                     delta=f"${(daily_profit/(daily_revenue+0.001)*100):.1f}% margin")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Call Center Optimization Suite
        
        This advanced tool helps you optimize the critical balance between:
        - **Agent Occupancy** (Efficiency & Cost)
        - **Service Level** (Customer Experience & Quality)
        
        ### 📊 **What this tool provides:**
        
        1. **Optimal Headcount Calculation** - Find the perfect number of agents
        2. **Trade-off Visualization** - See how occupancy and service level interact
        3. **Sensitivity Analysis** - Understand how changes impact performance
        4. **ROI Calculations** - Financial impact of your decisions
        5. **Actionable Recommendations** - Specific steps to improve
        
        ### 🎯 **How to use:**
        1. Adjust parameters in the **sidebar**
        2. Click **"Run Full Analysis"**
        3. Explore results across **5 detailed tabs**
        
        ### 📈 **Key Metrics Explained:**
        - **Call Volume**: Calls arriving per hour
        - **AHT**: Average Handle Time (seconds)
        - **ASA**: Acceptable Service Time (seconds)
        - **Occupancy**: % of time agents are busy
        - **Service Level**: % of calls answered within ASA
        
        ---
        
        **Ready to optimize your call center?** 👈 Adjust parameters in the sidebar and click **Run Full Analysis**
        """)
        
        # Quick example
        st.info("""
        **💡 Example Scenario:**
        - Volume: 40 calls/hour
        - AHT: 390 seconds
        - ASA: 30 seconds
        - Target SLA: 90%
        - Target Occupancy: 80%
        
        This typically requires ~12-15 agents for optimal balance.
        """)

# ========================
# RUN THE APP
# ========================

if __name__ == "__main__":
    # Initialize session state
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
    
    main()
