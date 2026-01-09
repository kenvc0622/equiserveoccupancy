
# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

"""# Final Occupancy Simulation

"""

# ========================
# DYNAMIC OPTIMIZATION WIDGET
# ========================

def create_optimization_widget():
    """Create interactive widget for optimization"""

    # Default values
    default_volume = 40
    default_AHT = 390
    default_ASA = 30
    default_target_sla = 90  # 90%
    default_target_occ = 0.8  # 80%
    default_interval = 60

    # Create widgets
    opt_volume = widgets.FloatSlider(
        value=default_volume,
        min=1,
        max=200,
        step=1,
        description='Volume:',
        continuous_update=False
    )

    opt_AHT = widgets.FloatSlider(
        value=default_AHT,
        min=60,
        max=1200,
        step=10,
        description='AHT (s):',
        continuous_update=False
    )

    opt_ASA = widgets.FloatSlider(
        value=default_ASA,
        min=5,
        max=180,
        step=5,
        description='ASA (s):',
        continuous_update=False
    )

    opt_target_sla = widgets.FloatSlider(
        value=default_target_sla,
        min=50,
        max=100,
        step=1,
        description='Target SLA (%):',
        continuous_update=False
    )

    opt_target_occ = widgets.FloatSlider(
        value=default_target_occ,
        min=0.1,
        max=1.0,
        step=0.01,
        description='Target Occ:',
        continuous_update=False,
        readout_format='.0%'
    )

    opt_interval = widgets.Dropdown(
        options=[15, 30, 60],
        value=default_interval,
        description='Interval (min):'
    )

    def run_optimization(volume, AHT, ASA, target_sla, target_occ, interval_minutes):
        # Convert interval to seconds
        interval_seconds = interval_minutes * 60

        # Find optimal headcount
        optimal_result, all_results = find_optimal_headcount(
            volume, AHT, ASA, target_sla, target_occ, interval_minutes
        )

        # Calculate CallsMaxOcc for TARGET OCCUPANCY with optimal headcount
        callsmaxocc_at_target = calculate_callsmaxocc(
            optimal_result['headcount'], AHT, target_occ, interval_seconds
        )

        # Calculate CallsMaxOcc for RESULTING OCCUPANCY
        callsmaxocc_at_result = calculate_callsmaxocc(
            optimal_result['headcount'], AHT, optimal_result['occupancy']/100, interval_seconds
        )

        # Calculate MaxCap per agent
        maxcap_per_agent = calculate_maxcap_per_agent(AHT, interval_seconds, target_occ)

        # Calculate headcount needed for target occupancy (ignoring SLA)
        hc_for_target_occ = calculate_required_hc_for_occupancy(
            volume, AHT, target_occ, interval_seconds
        )

        # Calculate headcount needed for target SLA (ignoring occupancy)
        # This requires finding headcount that achieves exactly target_sla
        hc_for_target_sla = None
        current_traffic = (volume * AHT/3600)

        # Bisection search for headcount that gives target_sla
        hc_low = 1
        hc_high = 50

        for _ in range(25):
            hc_mid = (hc_low + hc_high) / 2
            sl_mid = calculate_service_level(hc_mid, current_traffic, AHT, ASA) * 100

            if sl_mid >= target_sla:
                hc_high = hc_mid
            else:
                hc_low = hc_mid

        hc_for_target_sla = (hc_low + hc_high) / 2

        # Create output display
        print("="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)

        print(f"\nINPUT PARAMETERS:")
        print(f"• Volume: {volume} calls/hour")
        print(f"• AHT: {AHT} seconds")
        print(f"• ASA Target: {ASA} seconds")
        print(f"• Target Occupancy: {target_occ*100:.0f}%")
        print(f"• Target Service Level: {target_sla}%")
        print(f"• Interval: {interval_minutes} minutes")

        print(f"\nOPTIMAL SOLUTION (Balancing both targets):")
        print(f"• Recommended Headcount: {optimal_result['headcount']:.1f} agents")
        print(f"• Resulting Occupancy: {optimal_result['occupancy']:.1f}%")
        print(f"• Resulting Service Level: {optimal_result['service_level']:.1f}%")

        print(f"\nCAPACITY ANALYSIS:")
        print(f"• CallsMaxOcc for {target_occ*100:.0f}% Occ: {callsmaxocc_at_target:.1f} calls/hour")
        print(f"• CallsMaxOcc for {optimal_result['occupancy']:.1f}% Occ: {callsmaxocc_at_result:.1f} calls/hour")
        print(f"• MaxCap per agent ({target_occ*100:.0f}% Occ): {maxcap_per_agent:.2f} calls/hour")
        print(f"• Total team capacity: {maxcap_per_agent * optimal_result['headcount']:.1f} calls/hour")

        print(f"\nINDIVIDUAL TARGET REQUIREMENTS:")
        print(f"• Headcount for {target_occ*100:.0f}% Occ only: {hc_for_target_occ:.1f} agents")
        print(f"  → Would give SLA of {calculate_service_level(hc_for_target_occ, current_traffic, AHT, ASA)*100:.1f}%")
        print(f"• Headcount for {target_sla}% SLA only: {hc_for_target_sla:.1f} agents")
        print(f"  → Would give Occ of {calculate_occupancy(volume, AHT, hc_for_target_sla, interval_seconds)*100:.1f}%")

        print(f"\nTRADE-OFF ANALYSIS:")
        print(f"• Current volume vs CallsMaxOcc: ", end="")
        if volume > callsmaxocc_at_target:
            print(f"Volume ({volume}) > CallsMaxOcc ({callsmaxocc_at_target:.1f}) → Occupancy > {target_occ*100:.0f}%")
        elif volume < callsmaxocc_at_target:
            print(f"Volume ({volume}) < CallsMaxOcc ({callsmaxocc_at_target:.1f}) → Occupancy < {target_occ*100:.0f}%")
        else:
            print(f"Volume = CallsMaxOcc → Occupancy = {target_occ*100:.0f}%")

        # Create visualization of the trade-off
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

        # Mark key points
        ax1.axvline(x=optimal_result['headcount'], color='red', linestyle='--',
                   alpha=0.7, label=f'Optimal HC ({optimal_result["headcount"]:.1f})')
        ax1.axvline(x=hc_for_target_occ, color='blue', linestyle=':',
                   alpha=0.5, label=f'HC for {target_occ*100:.0f}% Occ ({hc_for_target_occ:.1f})')
        ax1.axvline(x=hc_for_target_sla, color='green', linestyle=':',
                   alpha=0.5, label=f'HC for {target_sla}% SLA ({hc_for_target_sla:.1f})')

        ax1.axhline(y=target_occ*100, color='blue', alpha=0.3, linestyle='--')
        ax1.axhline(y=target_sla, color='green', alpha=0.3, linestyle='--')

        ax1.set_xlabel('Headcount', fontweight='bold')
        ax1.set_ylabel('Performance (%)', fontweight='bold')
        ax1.set_title('Headcount Optimization: Occupancy vs SLA', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.set_ylim([0, 105])

        # Plot 2: Capacity comparison
        capacity_metrics = ['Volume', f'CallsMaxOcc\n({target_occ*100:.0f}% Occ)', 'Team Capacity']
        capacity_values = [volume, callsmaxocc_at_target, maxcap_per_agent * optimal_result['headcount']]
        colors = ['blue', 'red', 'green']

        bars = ax2.bar(capacity_metrics, capacity_values, color=colors, alpha=0.7)
        ax2.axhline(y=volume, color='blue', linestyle=':', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, capacity_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(capacity_values)*0.05,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

        # Add occupancy achieved at each capacity
        occ_at_volume = calculate_occupancy(volume, AHT, optimal_result['headcount'], interval_seconds)*100
        occ_at_callsmaxocc = target_occ*100  # By definition
        occ_at_team_capacity = calculate_occupancy(maxcap_per_agent * optimal_result['headcount'],
                                                  AHT, optimal_result['headcount'], interval_seconds)*100

        ax2.text(0, capacity_values[0]/2, f'{occ_at_volume:.1f}% Occ',
                ha='center', va='center', fontweight='bold', color='white', fontsize=9)
        ax2.text(1, capacity_values[1]/2, f'{occ_at_callsmaxocc:.0f}% Occ',
                ha='center', va='center', fontweight='bold', color='white', fontsize=9)
        ax2.text(2, capacity_values[2]/2, f'{occ_at_team_capacity:.1f}% Occ',
                ha='center', va='center', fontweight='bold', color='white', fontsize=9)

        ax2.set_ylabel('Calls per Hour', fontweight='bold')
        ax2.set_title(f'Capacity Comparison\n(Optimal HC: {optimal_result["headcount"]:.1f})',
                     fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

        # Print summary table
        print(f"\nSUMMARY TABLE:")
        print("-"*60)
        print(f"{'Metric':<25} {'Value':<15} {'Note':<20}")
        print("-"*60)
        print(f"{'Optimal Headcount':<25} {optimal_result['headcount']:<15.1f} {'Balances both targets':<20}")
        print(f"{'Resulting Occupancy':<25} {optimal_result['occupancy']:<15.1f}% {'':<20}")
        print(f"{'Resulting SLA':<25} {optimal_result['service_level']:<15.1f}% {'':<20}")
        print(f"{'CallsMaxOcc (80% Occ)':<25} {callsmaxocc_at_target:<15.1f} {'Max calls for target Occ':<20}")
        print(f"{'Volume vs CallsMaxOcc':<25} {volume/callsmaxocc_at_target:<15.2f}x {'Ratio':<20}")
        print(f"{'MaxCap per agent':<25} {maxcap_per_agent:<15.2f} {'Calls/hour at target Occ':<20}")
        print("-"*60)

        # Key insight
        print(f"\nKEY INSIGHT:")
        if volume > callsmaxocc_at_target:
            print(f"• You have MORE volume than needed for {target_occ*100:.0f}% occupancy")
            print(f"  → To maintain occupancy target, you need to increase HC or reduce AHT")
        else:
            print(f"• You have LESS volume than needed for {target_occ*100:.0f}% occupancy")
            print(f"  → To maintain occupancy target, you need to decrease HC or increase AHT")

    # Create interactive widget
    opt_interactive = widgets.interactive_output(
        run_optimization,
        {
            'volume': opt_volume,
            'AHT': opt_AHT,
            'ASA': opt_ASA,
            'target_sla': opt_target_sla,
            'target_occ': opt_target_occ,
            'interval_minutes': opt_interval
        }
    )

    # Layout
    controls_box = widgets.VBox([
        widgets.HBox([opt_volume, opt_AHT]),
        widgets.HBox([opt_ASA, opt_target_sla]),
        widgets.HBox([opt_target_occ, opt_interval]),
    ])

    return widgets.VBox([controls_box, opt_interactive])

# Add this before the error line
if 'create_interactive_tradeoff' not in globals():
    st.error("Function 'create_interactive_tradeoff' is not defined!")
    # Define it here or show instructions
    def create_interactive_tradeoff():
        return st.write("Placeholder function - define your actual function above")

# Update the main execution section
if __name__ == "__main__":
    print("="*80)
    print("DYNAMIC TRADE-OFF CURVE GENERATOR")
    print("="*80)
    print("\nAdjust sliders to simulate different scenarios:")
    print("• Volume: Calls arriving per hour")
    print("• AHT: Average Handle Time (seconds)")
    print("• ASA Target: Acceptable wait time for service level")
    print("• Headcount: Number of agents")
    print("• Target Occupancy: Desired utilization level")
    print("• Interval: Time period for calculations\n")

    # Create and display the interactive widget
    interactive_tool = create_interactive_tradeoff()
    display(interactive_tool)

    print("\n" + "="*80)
    print("DYNAMIC OPTIMIZATION TOOL")
    print("="*80)
    print("\nFind the optimal headcount that balances your occupancy and SLA targets:")

    # Create and display optimization widget
    optimization_tool = create_optimization_widget()
    display(optimization_tool)
def load_and_preprocess_data():
    """From your Colab data loading cell"""
    # Replace Colab paths
    # df = pd.read_csv('/content/data.csv')  # ❌ Colab path
    df = pd.read_csv('data.csv')  # ✅ Relative path
    return df

def calculate_optimal_mix(occupancy, service_level):
    """From your Colab calculation cell"""
    # Your calculation logic
    return optimal_values

def visualize_tradeoff(data):
    """From your Colab visualization cell"""
    fig = go.Figure()
    # Your plotting code
    return fig

# === MAIN APP FUNCTION ===
def create_interactive_tradeoff():
    """Main UI - combine all pieces"""
    st.title("Occupancy Optimization Dashboard")
    
    # Load data
    data = load_and_preprocess_data()
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        occupancy = st.slider("Target Occupancy", 0.65, 0.95, 0.85, 0.01)
    with col2:
        service_level = st.slider("Service Level", 0.80, 0.99, 0.95, 0.01)
    
    # Calculate
    results = calculate_optimal_mix(occupancy, service_level)
    
    # Visualize
    fig = visualize_tradeoff(results)
    st.plotly_chart(fig, use_container_width=True)
    
    return results

# === RUN THE APP ===
if __name__ == "__main__":
    # This is the entry point
    results = create_interactive_tradeoff()
    
    # Optional debug
    with st.expander("Raw Output"):
        st.write(results)




