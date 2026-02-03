import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from battery_simulation import simulate_battery
from cost_model import evaluate_annual_costs

def optimize_battery_size(load_kW, gen_kW, price_profile, params_base,
                          cap_range_kwh=(0, 200),
                          p_cha_range_kw=(0, 100),
                          p_dis_range_kw=(0, 100),
                          maxiter=50, popsize=10, workers=1):
    """
    Optimiert Batteriegröße, Lade- und Entladeleistung mit globaler Optimierung.
    Nutzt scipy.optimize.differential_evolution parallelisiert.
    
    Returns:
        best: dict mit 'params', 'sim', 'capacity_kwh', 'p_charge_kw', 'p_discharge_kw'
        df_full: DataFrame mit allen ausgewerteten Punkten
    """

    records = []

    # --------------------------
    # Wrapper für DE (muss top-level sein für multiprocessing)
    # --------------------------
    def objective_wrapper(x, load_kW, gen_kW, price_profile, params_base):
        params = params_base.copy()
        params['capacity_kwh'] = float(x[0])
        params['p_charge_kw'] = float(x[1])
        params['p_discharge_kw'] = float(x[2])

        bat_results = simulate_battery(load_kW, gen_kW, params)
        cost_results = evaluate_annual_costs(bat_results, params, price_profile)

        autarky = 1.0 - bat_results['total_import_kwh'] / (load_kW.sum()*params.get('timestep_hours',0.25)) if load_kW.sum() > 0 else 0.0

        # Speichere Ergebnis
        records.append({
            'capacity_kwh': x[0],
            'p_charge_kw': x[1],
            'p_discharge_kw': x[2],
            'annual_cost': cost_results['total_annual_cost'],
            'autarky': autarky
        })

        return cost_results['total_annual_cost']

    # --------------------------
    # Bounds für DE
    # --------------------------
    bounds = [cap_range_kwh, p_cha_range_kw, p_dis_range_kw]

    # --------------------------
    # Differential Evolution
    # --------------------------
    result = differential_evolution(
        objective_wrapper, bounds,
        args=(load_kW, gen_kW, price_profile, params_base),
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-2,
        polish=True,
        seed=42,
        workers=workers  # -1 für alle Kerne
    )

    best_x = result.x
    best_params = params_base.copy()
    best_params['capacity_kwh'] = float(best_x[0])
    best_params['p_charge_kw'] = float(best_x[1])
    best_params['p_discharge_kw'] = float(best_x[2])

    # Finale Simulation für beste Lösung
    best_sim = simulate_battery(load_kW, gen_kW, best_params)

    best = {
        'params': best_params,
        'sim': best_sim,
        'capacity_kwh': best_x[0],
        'p_charge_kw': best_x[1],
        'p_discharge_kw': best_x[2]
    }

    df_full = pd.DataFrame.from_records(records)
    return best, df_full
