# Base parameters for battery and economics
params_base = {
    'capacity_kwh': 50.0,      # to be optimized

    'p_charge_kw': 15.0, # max. charging power of battery
    'p_discharge_kw': 15.0, # max. discharging power of battery

    'soc_min_frac': 0.1,
    'soc_max_frac': 0.9,
    'eta_charge': 0.92,
    'eta_discharge': 0.92,
    'initial_soc_frac': 0.0,
    'timestep_hours': 0.25,

    'capex_per_kwh': 300,   # EUR per kWh (battery)
    'capex_per_kw': 150,    # EUR per kW inverter/PCS

    'lifetime_yr': 15,
    'discount_rate': 0.07,

    'opex_per_kwhyr': 5.0,
    'opex_per_kwyr': 2.0,
    
    'feedin_fraction': 0.25,   # feed-in price as fraction of import price

    # 'eol_soh': 0.8,           # End of Life bei 80% Restkapazität
    # 'cycle_life': 6000,       # Volllastzyklen bis EoL
    # 'calendar_life_yr': 20,   # Jahre bis EoL nur durch Zeit

    'cyclic_loss': 0.2, # 20% Verlust in 5000 Zyklen
    'calendar_loss': 0.02
}
