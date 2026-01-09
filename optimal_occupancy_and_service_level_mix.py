# optimal_occupancy_and_service_level_mix.py
import streamlit as st
import pandas as pd
import numpy as np

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
    """Calculate simplified service level"""
    if headcount <= traffic_intensity:
        return 0.0
    
    # Simplified approximation (for demo)
    base_sla = min(0.99, (headcount - traffic_intensity) / headcount * 1.5)
    adjustment = 1 - (ASA / AHT) * 0.3  # ASA impact
    
    return max(0.0, min(1.0, base_sla * adjustment))

# ========================
# STREAMLIT APP
# ========================

def main():
    st.title("📞 Call Center Occupancy & Service Level Optimizer")
    
    st.markdown("""
    Find the optimal balance between agent occupancy (efficiency) and service level (customer wait time).
    """)
    
    # Input parameters in columns
    col1, col2 = st.columns(2)
    
    with col1:
        volume = st.slider(
            "Call Volume (calls/hour)",
            min_value=1,
            max_value=200,
            value=40,
            step=1
        )
        
        AHT = st.slider(
            "Average Handle Time (seconds)",
            min_value=60,
            max_value=1200,
            value=390,
            step=10
        )
        
        ASA = st.slider(
            "ASA Target (seconds)",
            min_value=5,
            max_value=180,
            value=30,
            step=5
        )
    
    with col2:
        target_sla = st.slider(
            "Target Service Level (%)",
            min_value=50,
            max_value=100,
            value=90,
            step=1
        )
        
        target_occ = st.slider(
            "Target Occupancy (%)",
            min_value=10,
            max_value=100,
            value=80,
            step=1
        )
        
        interval = st.selectbox(
            "Interval (minutes)",
            options=[15, 30, 60],
            index=2
        )
    
    # Convert percentage to decimal
    target_occ_decimal = target_occ / 100
    interval_seconds = interval * 60
    
    # Calculate for different headcounts
    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Calculating optimal headcount..."):
            
            # Find optimal headcount
            best_hc = None
            best_score = float('inf')
            results = []
            
            for hc in range(1, 30):
                occ = calculate_occupancy(volume, AHT, hc, interval_seconds) * 100
                traffic = volume * AHT / 3600
                sla = calculate_service_level(hc, traffic, AHT, ASA) * 100
                
                # Score based on distance from targets
                occ_diff = abs(occ - target_occ)
                sla_diff = abs(sla - target_sla)
                score = occ_diff + sla_diff * 2
                
                results.append({
                    'Headcount': hc,
                    'Occupancy (%)': round(occ, 1),
                    'Service Level (%)': round(sla, 1),
                    'Score': round(score, 1)
                })
                
                if score < best_score:
                    best_score = score
                    best_hc = hc
            
            # Display optimal headcount
            st.success(f"✅ **Optimal Headcount: {best_hc} agents**")
            
            # Show results table
            st.subheader("📊 Simulation Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Performance Metrics
            st.subheader("📈 Performance at Optimal Headcount")
            
            optimal_result = results_df[results_df['Headcount'] == best_hc].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Headcount", best_hc)
            
            with col2:
                occ_value = optimal_result['Occupancy (%)']
                st.metric("Occupancy", f"{occ_value:.1f}%", 
                         delta=f"{occ_value - target_occ:.1f}% vs target")
            
            with col3:
                sla_value = optimal_result['Service Level (%)']
                st.metric("Service Level", f"{sla_value:.1f}%",
                         delta=f"{sla_value - target_sla:.1f}% vs target")
            
            with col4:
                capacity = (best_hc * interval_seconds / AHT) * (target_occ_decimal)
                st.metric("Capacity", f"{capacity:.1f} calls/hour")
            
            # Visualizations using Streamlit native charts
            st.subheader("📉 Performance Trends")
            
            # Create data for charts
            chart_data = pd.DataFrame({
                'Headcount': results_df['Headcount'],
                'Occupancy': results_df['Occupancy (%)'],
                'Service Level': results_df['Service Level (%)']
            }).set_index('Headcount')
            
            # Line chart
            st.line_chart(chart_data)
            
            # Highlight optimal point
            st.info(f"""
            **Analysis at {best_hc} agents:**
            - **Occupancy:** {optimal_result['Occupancy (%)']:.1f}% (Target: {target_occ}%)
            - **Service Level:** {optimal_result['Service Level (%)']:.1f}% (Target: {target_sla}%)
            - **Gap Score:** {optimal_result['Score']:.1f} (lower is better)
            """)
            
            # Capacity Analysis
            st.subheader("⚙️ Capacity Analysis")
            
            # Calculate capacity metrics
            max_calls_at_target = (best_hc * interval_seconds / AHT) * (target_occ_decimal)
            volume_ratio = volume / max_calls_at_target if max_calls_at_target > 0 else 0
            
            capacity_data = pd.DataFrame({
                'Metric': ['Current Volume', 'Max at Target Occupancy', 'Utilization'],
                'Value': [volume, round(max_calls_at_target, 1), f"{volume_ratio*100:.1f}%"]
            })
            
            st.table(capacity_data)
            
            # Insights
            st.subheader("💡 Recommendations")
            
            if volume > max_calls_at_target:
                st.warning(f"""
                **High Volume Alert:** Current volume ({volume}) exceeds capacity at target occupancy ({max_calls_at_target:.1f}).
                
                **Consider:**
                1. Increase headcount beyond {best_hc} agents
                2. Reduce AHT from {AHT} seconds
                3. Implement call deflection strategies
                """)
            else:
                st.success(f"""
                **Capacity Available:** You can handle {max_calls_at_target - volume:.1f} more calls/hour 
                while maintaining {target_occ}% occupancy with {best_hc} agents.
                """)
            
            # Export option
            st.subheader("📥 Export Results")
            
            # Convert to CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="occupancy_simulation.csv",
                mime="text/csv"
            )

# Run the app
if __name__ == "__main__":
    main()
