import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ========================
# ASSUME HELPER FUNCTIONS ARE DEFINED HERE
# ========================
# Place definitions for the following functions here:
# - find_optimal_headcount(volume, AHT, ASA, target_sla, target_occ, interval_minutes)
# - calculate_callsmaxocc(headcount, AHT, occupancy, interval_seconds)
# - calculate_maxcap_per_agent(AHT, interval_seconds, target_occ)
# - calculate_required_hc_for_occupancy(volume, AHT, target_occ, interval_seconds)
# - calculate_service_level(headcount, traffic, AHT, ASA)
# - calculate_occupancy(volume, AHT, headcount, interval_seconds)
# These are referenced in the code but not provided in the snippet.

# ========================
# DYNAMIC OPTIMIZATION FUNCTION
# ========================

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
    st.write("="*80)
    st.write("OPTIMIZATION RESULTS")
    st.write("="*80)

    st.write("\nINPUT PARAMETERS:")
    st.write(f"• Volume: {volume} calls/hour")
    st.write(f"• AHT: {AHT} seconds")
    st.write(f"• ASA Target: {ASA} seconds")
    st.write(f"• Target Occupancy: {target_occ*100:.0f}%")
    st.write(f"• Target Service Level: {target_sla}%")
    st.write(f"• Interval: {interval_minutes} minutes")

    st.write("\nOPTIMAL SOLUTION (Balancing both targets):")
    st.write(f"• Recommended Headcount: {optimal_result['headcount']:.1f} agents")
    st.write(f"• Resulting Occupancy: {optimal_result['occupancy']:.1f}%")
    st.write(f"• Resulting Service Level: {optimal_result['service_level']:.1f}%")

    st.write("\nCAPACITY ANALYSIS:")
    st.write(f"• CallsMaxOcc for {target_occ*100:.0f}% Occ: {callsmaxocc_at_target:.1f} calls/hour")
    st.write(f"• CallsMaxOcc for {optimal_result['occupancy']:.1f}% Occ: {callsmaxocc_at_result:.1f} calls/hour")
    st.write(f"• MaxCap per agent ({target_occ*100:.0f}% Occ): {maxcap_per_agent:.2f} calls/hour")
    st.write(f"• Total team capacity: {maxcap_per_agent * optimal_result['headcount']:.1f} calls/hour")

    st.write("\nINDIVIDUAL TARGET REQUIREMENTS:")
    st.write(f"• Headcount for {target_occ*100:.0f}% Occ only: {hc_for_target_occ:.1f} agents")
    st.write(f"  → Would give SLA of {calculate_service_level(hc_for_target_occ, current_traffic, AHT, ASA)*100:.1f}%")
    st.write(f"• Headcount for {target_sla}% SLA only: {hc_for_target_sla:.1f} agents")
    st.write(f"  → Would give Occ of {calculate_occupancy(volume, AHT, hc_for_target_sla, interval_seconds)*100:.1f}%")

    st.write("\nTRADE-OFF ANALYSIS:")
    st.write(f"• Current volume vs CallsMaxOcc: ", end="")
    if volume > callsmaxocc_at_target:
        st.write(f"Volume ({volume}) > CallsMaxOcc ({callsmaxocc_at_target:.1f}) → Occupancy > {target_occ*100:.0f}%")
    elif volume < callsmaxocc_at_target:
        st.write(f"Volume ({volume}) < CallsMaxOcc ({callsmaxocc_at_target:.1f}) → Occupancy < {target_occ*100:.0f}%")
    else:
        st.write(f"Volume = CallsMaxOcc → Occupancy = {target_occ*100:.0f}%")

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
    st.pyplot(fig)

    # Print summary table
    st.write("\nSUMMARY TABLE:")
    st.write("-"*60)
    st.write(f"{'Metric':<25} {'Value':<15} {'Note':<20}")
    st.write("-"*60)
    st.write(f"{'Optimal Headcount':<25} {optimal_result['headcount']:<15.1f} {'Balances both targets':<20}")
    st.write(f"{'Resulting Occupancy':<25} {optimal_result['occupancy']:<15.1f}% {'':<20}")
    st.write(f"{'Resulting SLA':<25} {optimal_result['service_level']:<15.1f}% {'':<20}")
    st.write(f"{'CallsMaxOcc (80% Occ)':<25} {callsmaxocc_at_target:<15.1f} {'Max calls for target Occ':<20}")
    st.write(f"{'Volume vs CallsMaxOcc':<25} {volume/callsmaxocc_at_target:<15.2f}x {'Ratio':<20}")
    st.write(f"{'MaxCap per agent':<25} {maxcap_per_agent:<15.2f} {'Calls/hour at target Occ':<20}")
    st.write("-"*60)

    # Key insight
    st.write("\nKEY INSIGHT:")
    if volume > callsmaxocc_at_target:
        st.write(f"• You have MORE volume than needed for {target_occ*100:.0f}% occupancy")
        st.write(f"  → To maintain occupancy target, you need to increase HC or reduce AHT")
    else:
        st.write(f"• You have LESS volume than needed for {target_occ*100:.0f}% occupancy")
        st.write(f"  → To maintain occupancy target, you need to decrease HC or increase AHT")

# ========================
# MAIN STREAMLIT APP
# ========================

if __name__ == "__main__":
    st.write("="*80)
    st.write("DYNAMIC TRADE-OFF CURVE GENERATOR")
    st.write("="*80)
    st.write("\nAdjust sliders to simulate different scenarios:")
    st.write("• Volume: Calls arriving per hour")
    st.write("• AHT: Average Handle Time (seconds)")
    st.write("• ASA Target: Acceptable wait time for service level")
    st.write("• Target Occupancy: Desired utilization level")
    st.write("• Target Service Level: Desired SLA percentage")
    st.write("• Interval: Time period for calculations\n")

    # Note: create_interactive_tradeoff() was not provided in the original code snippet.
    # If you have its definition, add it here and call st.pyplot() for its output if it produces a figure.
    # For now, it's omitted.

    st.write("\n" + "="*80)
    st.write("DYNAMIC OPTIMIZATION TOOL")
    st.write("="*80)
    st.write("\nFind the optimal headcount that balances your occupancy and SLA targets:")

    # Create controls
    volume = st.slider('Volume:', min_value=1.0, max_value=200.0, value=40.0, step=1.0)
    AHT = st.slider('AHT (s):', min_value=60.0, max_value=1200.0, value=390.0, step=10.0)
    ASA = st.slider('ASA (s):', min_value=5.0, max_value=180.0, value=30.0, step=5.0)
    target_sla = st.slider('Target SLA (%):', min_value=50.0, max_value=100.0, value=90.0, step=1.0)
    target_occ = st.slider('Target Occ:', min_value=0.1, max_value=1.0, value=0.8, step=0.01, format="%.2f")
    interval_minutes = st.selectbox('Interval (min):', options=[15, 30, 60], index=2)

    # Run the optimization
    run_optimization(volume, AHT, ASA, target_sla, target_occ, interval_minutes)
