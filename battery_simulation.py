import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io, os, math
from numba import njit

# hier ist die Batterielogik gecodet - mit numba/@nijt in Maschinencode ausgeführt, für schnellere Berechnung
@njit
def battery_loop(load, gen, dt, cap, soc_min_frac, soc_max_frac, pch, pdis, eta_c, eta_d, initial_soc_frac, cyc_loss, cal_loss):
    n = len(load)
    soc = initial_soc_frac * cap
    soc_series = np.zeros(n)
    cap_series = np.zeros(n)
    charge_series = np.zeros(n)
    discharge_series = np.zeros(n)
    import_from_grid = np.zeros(n)
    export_to_grid = np.zeros(n)
    curtailed = np.zeros(n)
    throughput = 0.0
    calendar_loss = (cap*cal_loss)/8760
    cyclic_loss = cyc_loss / 5000
    # Sicherheitsabfrage: Wenn Kapazität 0 ist, sind auch die Bruchteile 0
    current_cap = cap  # Initialisierung der Arbeitsvariable

    for i in range(n):
        # 1. Kalendarische Alterung abziehen (pro Zeitschritt)
        soc_max = soc_max_frac * current_cap
        soc_min = soc_min_frac * current_cap   

        current_cap -= calendar_loss * dt

        net = gen[i] - load[i]

        if net >= 0:
            # Überschuss
            # hier ist die Logik klar 
            available_charge_power = min(pch, net)
            energy_offer = available_charge_power * dt
            space = soc_max - soc
            accepted_energy = min(space, energy_offer * eta_c)
            energy_drawn = accepted_energy / eta_c if eta_c > 0 else 0.0

            soc += accepted_energy
            # Hier throughput und cycle loss nur auf Entladung beziehen, scheinbar realistischer
            # throughput += accepted_energy
            # 2. Zyklische Alterung (basierend auf geladener Energie)
            # current_cap -= accepted_energy * cyclic_loss

            charge_series[i] = energy_drawn / dt
            remaining_surplus_kWh = net * dt - energy_drawn
            # hier Obergrenze definieren für Energieverkauf zum Grid?
            # hier auch Verluste?
            export_to_grid[i] = max(0.0, remaining_surplus_kWh)
            discharge_series[i] = 0.0
            import_from_grid[i] = 0.0
        else:
            # Defizit
            demand = -net
            available_discharge_power = min(pdis, demand)
            usable_soc = max(0.0, soc - soc_min)
            Eout = min(available_discharge_power * dt, usable_soc * eta_d)

            soc -= Eout / eta_d if eta_d > 0 else 0.0
            throughput += Eout
            # Optional: Auch Entladung lässt Kapazität sinken
            current_cap -= Eout * cyclic_loss

            discharge_series[i] = Eout / dt
            unmet = demand * dt - Eout
            import_from_grid[i] = max(0.0, unmet)
            charge_series[i] = 0.0
            export_to_grid[i] = 0.0

        # Bound SOC Absicherung für alle Fälle - wird aber eigentlich durch obigen Code schon abgesichert
        if soc < soc_min:
            soc = soc_min
        if soc > soc_max:
            curtailed[i] += soc - soc_max
            soc = soc_max

        soc_series[i] = soc
        cap_series[i] = current_cap

    return soc_series, cap_series, charge_series, discharge_series, import_from_grid, export_to_grid, curtailed, throughput



# Orchestrierung - "Vorbereitung & Verpackung" des Battery Loops
def simulate_battery(load_kW, gen_kW, params):
    assert load_kW.index.equals(gen_kW.index)
    dt = params.get('timestep_hours', 0.25)

    # Degradations-Parameter (Beispielwerte)
    # Annahme: 20% Verlust nach 5000 Vollzyklen bei 100 kWh -> 20 kWh / (5000 * 100)
    cyclic_loss = params.get('cyclic_loss', 0.2) 
    # Annahme: 2% kalendarischer Verlust pro Jahr
    calendar_loss = params.get('calendar_loss', 0.02)

    cap = params['capacity_kwh']
    soc_min_frac = params.get('soc_min_frac', 0.0) 
    soc_max_frac = params.get('soc_max_frac', 1.0) 
    pch = params['p_charge_kw']
    pdis = params['p_discharge_kw']
    eta_c = params.get('eta_charge', 0.95)
    eta_d = params.get('eta_discharge', 0.95)
    initial_soc_frac = params.get('initial_soc_frac', 0.1) 

    load = load_kW.values
    gen = gen_kW.values

    soc_series, cap_series, charge_series, discharge_series, import_from_grid, export_to_grid, curtailed, throughput = battery_loop(
        load, gen, dt, cap, soc_min_frac, soc_max_frac, pch, pdis, eta_c, eta_d, initial_soc_frac, cyclic_loss, calendar_loss
    )

    timeline = pd.DataFrame({
        'load_kW': load,
        'gen_kW': gen,
        'pv_to_load_kW': np.minimum(load, gen),
        'battery_to_load_kW': discharge_series,
        'charge_kW': charge_series,
        'discharge_kW': discharge_series,
        'soc_kWh': soc_series,
        'import_kWh': import_from_grid,
        'export_kWh': export_to_grid,
        'curtailed_kWh': curtailed
    }, index=load_kW.index)

    results = {
        'timeline': timeline,
        'soh_final_kWh': cap_series[-1],
        'throughput_kwh': throughput,
        'total_import_kwh': import_from_grid.sum(),
        'total_export_kwh': export_to_grid.sum(),
        'total_curtailed_kwh': curtailed.sum(),
    }
    return results





















