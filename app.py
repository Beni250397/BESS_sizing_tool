

import streamlit as st
import os
import time
import pandas as pd
import numpy as np
from read_csv import load_profile, generate_price_profile
from battery_simulation import simulate_battery
from cost_model import evaluate_annual_costs
from optimization import optimize_battery_size
# Hier die neue Funktion importieren (stelle sicher, dass sie in plot.py so heißt)
from plot import plot_interactive_full_period 
from parameters import params_base

st.set_page_config(page_title="BESS Sizing Tool", layout="wide")

st.title("🔋 BESS-Dimensionierung")

# --- SIDEBAR: DATEI-AUSWAHL (DRAG & DROP) ---
st.sidebar.header("📂 Dateien hochladen")

# Drag & Drop für Lastprofil
uploaded_load = st.sidebar.file_uploader(
    "Lastprofil hochladen (CSV)", 
    type=["csv"], 
    help="Falls keine Datei gewählt wird, wird ein synthetisches Lastprofil berechnet."
)

# Logik: Wenn nichts hochgeladen wurde, MUSS die Variable None sein
load_csv_path = uploaded_load if uploaded_load is not None else None
# Hinweis in der Sidebar, was aktuell genutzt wird
if load_csv_path is None:
    st.sidebar.info("ℹ️ Last: Synthetisch (Default)")
else:
    st.sidebar.success(f"✅ Last: {uploaded_load.name}")

# Drag & Drop für Erzeugungsprofil
uploaded_gen = st.sidebar.file_uploader(
    "Erzeugungsprofil hochladen (CSV)", 
    type=["csv"], 
    help="Falls keine Datei gewählt wird, wird ein synthetisches Erzeugungsprofil berechnet."
)
gen_csv_path = uploaded_gen if uploaded_gen is not None else None
if gen_csv_path is None:
    st.sidebar.info("ℹ️ Erzeugung: Synthetisch (Default)")
else:
    st.sidebar.success(f"✅ Erzeugung: {uploaded_gen.name}")

# --- MAIN: PARAMETER ANPASSEN ---
st.header("⚙️ Parameter")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Kapazität & Leistung")
    cap = st.number_input('Max. Kapazität (kWh)', value=float(params_base['capacity_kwh']))
    p_charge = st.number_input('Max. Ladeleistung (kW)', value=float(params_base['p_charge_kw']))
    p_discharge = st.number_input('Max. Entladeleistung (kW)', value=float(params_base['p_discharge_kw']))
    lifetime = st.number_input('Lebensdauer (Jahre)', value=int(params_base['lifetime_yr']))

with col2:
    st.subheader("Effizienz & SOC")
    eta_c = st.slider('Lade-Wirkungsgrad', 0.0, 1.0, float(params_base['eta_charge']))
    eta_d = st.slider('Entlade-Wirkungsgrad', 0.0, 1.0, float(params_base['eta_discharge']))
    soc_min = st.slider('Min. SoC', 0.0, 1.0, float(params_base['soc_min_frac']))
    soc_max = st.slider('Max. SoC', 0.0, 1.0, float(params_base['soc_max_frac']))

with col3:
    st.subheader("Ökonomie")
    capex_kwh = st.number_input('CAPEX Energie (€/kWh)', value=float(params_base['capex_per_kwh']))
    capex_kw = st.number_input('CAPEX Leistung (€/kW)', value=float(params_base['capex_per_kw']))
    opex_kwh = st.number_input('OPEX Energie (€/kWh/Jahr)', value=float(params_base['opex_per_kwhyr']))
    opex_kw = st.number_input('OPEX Leistung (€/kW/Jahr)', value=float(params_base['opex_per_kwyr']))
    feedin = st.number_input('Einspeisefaktor (Anteil Importpreis)', value=float(params_base['feedin_fraction']))
    discount = st.number_input('Discount Rate (Zinssatz)', value=float(params_base['discount_rate']))

# Dictionary aktualisieren
params_base.update({
    'capacity_kwh': cap, 'p_charge_kw': p_charge, 'p_discharge_kw': p_discharge,
    'soc_min_frac': soc_min, 'soc_max_frac': soc_max, 'eta_charge': eta_c, 'eta_discharge': eta_d,
    'capex_per_kwh': capex_kwh, 'capex_per_kw': capex_kw, 'lifetime_yr': lifetime,
    'discount_rate': discount, 'opex_per_kwhyr': opex_kwh, 'opex_per_kwyr': opex_kw, 'feedin_fraction': feedin,
})

# --- EXECUTION BUTTON ---
if st.button('🚀 Optimierung und Simulation starten'):
    with st.spinner('Berechne Optimierung...'):
        start_time = time.time()
        
        # 1. Daten laden
        load_series, gen_series = load_profile(load_csv_path, gen_csv_path)
        price_profile = generate_price_profile(load_series.index)

        # 2. Run optimization
        best, df_eval = optimize_battery_size(
            load_series, gen_series, price_profile, params_base, 
            cap_range_kwh=(0, params_base['capacity_kwh']), 
            p_cha_range_kw=(0, params_base['p_charge_kw']), 
            p_dis_range_kw=(0, params_base['p_discharge_kw'])
        )

        # 3. Bestwerte extrahieren
        best_params = best['params']
        best_params.update({
            'capacity_kwh': float(best['capacity_kwh']),
            'p_charge_kw': float(best['p_charge_kw']),
            'p_discharge_kw': float(best['p_discharge_kw'])
        })

        # 4. Simulationen
        sim_best = simulate_battery(load_series, gen_series, best_params)
        econ_best = evaluate_annual_costs(sim_best, best_params, price_profile)

        params_nobatt = params_base.copy()
        params_nobatt.update({'capacity_kwh': 0.0, 'p_charge_kw': 0.0, 'p_discharge_kw': 0.0})
        sim_nobatt = simulate_battery(load_series, gen_series, params_nobatt)
        econ_nobatt = evaluate_annual_costs(sim_nobatt, params_nobatt, price_profile)

        # 5. Kennzahlen
        autarky_best = 1.0 - sim_best['total_import_kwh'] / (load_series.sum() * params_base['timestep_hours'])
        autarky_nobatt = 1.0 - sim_nobatt['total_import_kwh'] / (load_series.sum() * params_base['timestep_hours'])
        runtime = time.time() - start_time

        # --- ERGEBNISSE ANZEIGEN ---
        st.success(f"Berechnung abgeschlossen in {runtime:.2f} Sekunden")
        
        st.header("📊 Analyse-Ergebnisse")
        res1, res2, res3 = st.columns(3)
        res1.metric("Optimale Kapazität", f"{best['capacity_kwh']:.1f} kWh")
        res1.metric("Max. Ladeleistung", f"{best['p_charge_kw']:.1f} kW")
        res1.metric("Max. Entladeleistung", f"{best['p_discharge_kw']:.1f} kW")
        

        # Berechnung der prozentualen Ersparnis
        cost_with = econ_best['total_annual_cost']
        cost_no = econ_nobatt['total_cost_no_batt']
        # Vermeidung von Division durch Null
        rel_savings = ((cost_with / cost_no) - 1) * 100 if cost_no != 0 else 0
        
        res2.metric("Jährliche Ersparnis", f"{econ_best['savings_vs_no_batt']:.2f} €")
        
        # Spalte 2: Kosten-Vergleich
        res2.metric(
            label="Kosten mit Batterie", 
            value=f"{cost_with:.2f} €/a", 
            delta=f"{rel_savings:.1f} %",
            delta_color="green" # Zeigt Senkung automatisch in Grün an
        )
        res2.metric("Kosten ohne Batterie", f"{econ_nobatt['total_cost_no_batt']:.2f} €/a")
        
        res3.metric("Autarkiegrad", f"{autarky_best*100:.1f} %", f"{(autarky_best-autarky_nobatt)*100:.1f} %")
        

        # --- PLOTTING (NEU: Nur eine interaktive Gesamtansicht) ---
        st.header("📈 Interaktive Detailanalyse")
        st.info("💡 Nutze die Maus zum Zoomen (Rechteck ziehen) oder Shift+Linksklick zum Verschieben. Doppelklick setzt den Zoom zurück.")

        # Aufruf der neuen Plotly-Funktion mit dem gesamten Zeitraum
        fig = plot_interactive_full_period(sim_best)
        
        # Anzeige in Streamlit
        st.plotly_chart(fig, use_container_width=True)