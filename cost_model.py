import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io, os, math

# ------------------ Cost model ------------------

def evaluate_annual_costs(sim_res, params, price_profile):
    """
    Compute annual cost given simulation results and economic params.
    price_profile: pandas Series (kWh price for each timestep)
    params: includes capex_per_kwh, capex_per_kw, lifetime_yr, discount_rate, opex_frac, feedin_factor
    """
    timeline = sim_res['timeline']
    dt = params.get('timestep_hours', 0.25)
    # grid energy import cost
    # price_profile is per kWh aligned with timeline index
    imported = timeline['import_kWh'].values  # kWh per timestep
    exported = timeline['export_kWh'].values

    # hier werden die Stromkosten erst berechnet von Strombezug aus dem Netz - das kann man hier lassen, weil hier der Preis irrelevant - wenn der restliche Bedarf offen ist, muss er vom Netz gedeckt werden
    import_cost = (imported * price_profile.values).sum()

    # revenue from export (assume some feed-in fraction)
    ############# MÖGLICHE ÄNDERUNG!!!!!!!!
    # das hier könnte man allerdings in battery simulation schon mitreinnehmen: also neue Logik die sagt "Strom wird lieber verkauft wenn Verkaufspreis >= x € anstatt in BESS gespeichert. wenn < x € dann erste Prio in BESS einspeichern."
    feedin_price = price_profile * params.get('feedin_fraction', 0.5)
    export_revenue = (exported * feedin_price.values).sum()

    # battery annualized cost
    capex_kwh = params.get('capex_per_kwh', 400.0)  # EUR per kWh
    capex_kw = params.get('capex_per_kw', 200.0)    # EUR per kW
    capex = capex_kwh * params['capacity_kwh'] + capex_kw * max(params['p_charge_kw'], params['p_discharge_kw'])
    lifetime = params.get('lifetime_yr', 15)
    dr = params.get('discount_rate', 0.07)
    # annualized capital cost using CRF
    if dr>0:
        crf = (dr*(1+dr)**lifetime)/((1+dr)**lifetime - 1)
    else:
        crf = 1.0/lifetime
    annualized_capex = capex * crf
    # opex, 5 ist ein Default-Wert!
    opex_annual = params.get('opex_per_kwhyr', 5.0) * params['capacity_kwh'] + params.get('opex_per_kwyr', 2.0)*max(params['p_charge_kw'],params['p_discharge_kw'])
    
    # # --- Realistische Degradationsberechnung ---
    # cap = params['capacity_kwh']
    # throughput_kwh = sim_res['throughput_kwh']
    
    # # 1. Zyklische Degradation: Wie viel Kapazität "verbrauchen" wir?
    # # Ein Zyklus entspricht 2 * Kapazität (Laden + Entladen)
    # cycles_done = throughput_kwh / (2 * cap) if cap > 0 else 0
    # degradation_cyclic = cycles_done / params.get('cycle_life', 6000)
    
    # # 2. Kalendarische Degradation (pro Jahr)
    # degradation_calendar = 1.0 / params.get('calendar_life_yr', 20)
    
    # # Die tatsächliche Degradation ist oft das Maximum oder die Summe beider Effekte
    # # Wir nutzen hier einen gängigen kombinierten Ansatz:
    # annual_soh_loss = max(degradation_cyclic, degradation_calendar) * (1 - params.get('eol_soh', 0.8))
    
    # # Wirtschaftlicher Wertverlust: Was kostet uns dieser Verschleiß?
    # # (Anteil am CAPEX, der durch die Nutzung in diesem Jahr "verbraucht" wurde)
    # capex_total = (params['capex_per_kwh'] * cap + 
    #                params['capex_per_kw'] * max(params['p_charge_kw'], params['p_discharge_kw']))
    
    # # Die Degradationskosten entsprechen dem Anteil am Wertverlust bis zum EoL
    # # Wenn 20% Kapazitätsverlust = 100% Wertverlust (EoL) bedeuten:
    # degradation_cost_annual = (annual_soh_loss / (1 - params.get('eol_soh', 0.8))) * capex_total / params.get('lifetime_yr', 15)
    
    # # Optional: Wir können auch einfach den Durchsatz bepreisen (z.B. 0.03 €/kWh Durchsatz)
    # # degradation_cost_annual = throughput_kwh * 0.03

    total_annual_cost = import_cost - export_revenue + annualized_capex + opex_annual #+ degradation_cost_annual

    # For comparison, compute cost without battery: everything unmet imported, generation exported at feed-in price
    # That is equivalent to run with capacity=0 -> all gen exported, imports = load - gen positive
    net = timeline['gen_kW'] - timeline['load_kW']
    import_no_batt = np.where(net < 0, -net * dt, 0.0)
    export_no_batt = np.where(net > 0, net * dt, 0.0)
    import_cost_no_batt = (import_no_batt * price_profile.values).sum()
    export_revenue_no_batt = (export_no_batt * feedin_price.values).sum()
    total_cost_no_batt = import_cost_no_batt - export_revenue_no_batt  # no capex obviously

    results = {
        'import_cost': import_cost,
        'export_revenue': export_revenue,
        'annualized_capex': annualized_capex,
        'opex_annual': opex_annual,
        'total_annual_cost': total_annual_cost,
        'total_cost_no_batt': total_cost_no_batt,
        'savings_vs_no_batt': total_cost_no_batt - total_annual_cost,
        #'annual_soh_loss_pct': annual_soh_loss * 100,
        #'degradation_cost_annual': degradation_cost_annual,
        #'equivalent_full_cycles': cycles_done
    }
    return results