import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io, os, math

# ------------------ Cost model ------------------

def evaluate_annual_costs(sim_res, params, price_profile):
    timeline = sim_res['timeline']
    dt = params.get('timestep_hours', 0.25)
    
    # --- NEU: Skalierungsfaktor berechnen ---
    # Wir messen die Dauer der Simulation in Tagen
    duration_days = (timeline.index[-1] - timeline.index[0]).total_seconds() / (24 * 3600)
    # Falls der Index keine Zeitstempel hat: duration_days = (len(timeline) * dt) / 24
    scaling_factor = 365.0 / duration_days if duration_days > 0 else 1.0

    # Variable Kosten (Summe über Simulationszeitraum)
    price_profile = price_profile.reindex(timeline.index).ffill()
    import_cost_sim = (timeline['import_kWh'] * price_profile).sum()

    feedin_price = price_profile * params.get('feedin_fraction', 0.5)
    export_rev_sim = (timeline['export_kWh'] * feedin_price.values).sum()

    # Annualisierung der variablen Kosten
    annualized_import_cost = import_cost_sim * scaling_factor
    annualized_export_revenue = export_rev_sim * scaling_factor

    # Fixkosten (bereits auf Jahresbasis)
    capex_kwh = params.get('capex_per_kwh', 400.0)
    capex_kw = params.get('capex_per_kw', 200.0)
    capex = capex_kwh * params['capacity_kwh'] + capex_kw * max(params['p_charge_kw'], params['p_discharge_kw'])
    
    lifetime = params.get('lifetime_yr', 15)
    dr = params.get('discount_rate', 0.07)
    crf = (dr*(1+dr)**lifetime)/((1+dr)**lifetime - 1) if dr > 0 else 1.0/lifetime
    annualized_capex = capex * crf
    
    opex_annual = (params.get('opex_per_kwhyr', 5.0) * params['capacity_kwh'] + 
                   params.get('opex_per_kwyr', 2.0) * max(params['p_charge_kw'], params['p_discharge_kw']))

    total_annual_cost = annualized_import_cost - annualized_export_revenue + annualized_capex + opex_annual

    # Vergleich ohne Batterie (ebenfalls annualisiert)
    net_no_batt = (timeline['gen_kW'] - timeline['load_kW']) * dt
    import_no_batt_series = np.where(net_no_batt < 0, -net_no_batt, 0.0)
    export_no_batt_series = np.where(net_no_batt > 0, net_no_batt, 0.0)
    
    cost_no_batt_sim = (import_no_batt_series * price_profile.values).sum()
    rev_no_batt_sim = (export_no_batt_series * (price_profile.values * params.get('feedin_fraction', 0.5))).sum()
    
    total_cost_no_batt = (cost_no_batt_sim - rev_no_batt_sim) * scaling_factor

    return {
        'import_cost_annual': annualized_import_cost,
        'export_revenue_annual': annualized_export_revenue,
        'annualized_capex': annualized_capex,
        'opex_annual': opex_annual,
        'total_annual_cost': total_annual_cost,
        'total_cost_no_batt': total_cost_no_batt,
        'savings_vs_no_batt': total_cost_no_batt - total_annual_cost,
        'scaling_factor_applied': scaling_factor
    }
