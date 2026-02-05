import streamlit as st
import os
import time
import pandas as pd
import numpy as np
from read_csv import load_profile, generate_price_profile
from battery_simulation import simulate_battery
from cost_model import evaluate_annual_costs
from optimization import optimize_battery_size
from plot import plot_interactive_full_period 
from parameters import params_base

def format_de(n, decimals=2):
    """Formatiert Zahlen zu 1.234,56"""
    return f"{n:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

# --- INITIALISIERUNG SESSION STATE (NEU) ---
# Wir kopieren die Werte aus params_base einmalig in den Session State.
# So "gehören" die Werte Streamlit und werden nicht bei jedem Klick resettet.
for key, val in params_base.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.set_page_config(page_title="BESS Sizing Tool", layout="wide")


st.title("🔋 BESS-Dimensionierung")

# --- SIDEBAR: DATEI-AUSWAHL & STANDORT ---
st.sidebar.header("📂 Dateien hochladen")

# 1. Lastprofil
uploaded_load = st.sidebar.file_uploader(
    "Lastprofil hochladen (CSV)", 
    type=["csv"], 
    help="Falls keine Datei gewählt wird, wird ein synthetisches Lastprofil berechnet."
)
if uploaded_load is None:
    st.sidebar.info("ℹ️ Last: Synthetisch (Default)")
    load_csv_path = None
else:
    st.sidebar.success(f"✅ Last: {uploaded_load.name}")
    load_csv_path = uploaded_load

# 2. Erzeugungsprofil
uploaded_gen = st.sidebar.file_uploader(
    "Erzeugungsprofil hochladen (CSV)", 
    type=["csv"], 
    help="Falls keine Datei gewählt wird, wird ein synthetisches Erzeugungsprofil berechnet."
)
if uploaded_gen is None:
    st.sidebar.info("ℹ️ Erzeugung: Synthetisch (Default)")
    gen_csv_path = None
else:
    st.sidebar.success(f"✅ Erzeugung: {uploaded_gen.name}")
    gen_csv_path = uploaded_gen


# --- NEU: STANDORT-PARAMETER FÜR SYNTHESE ---
st.sidebar.markdown("---")
st.sidebar.header("📍 Erzeugung & Last")
st.sidebar.caption("Wird genutzt, wenn keine CSV hochgeladen wurde")

lat_input = st.sidebar.number_input("Breitengrad (Lat)", value=50.1319, format="%.4f")
lon_input = st.sidebar.number_input("Längengrad (Lon)", value=8.6838, format="%.4f")
peak_pv_input = st.sidebar.number_input("PV Leistung (kWp)", value=20.0)
peak_load_input = st.sidebar.number_input("Max. Last (kW)", value=20.0)

# --- MAIN: PARAMETER ANPASSEN ---
st.header("⚙️ Parameter")
col1, col2, col3 = st.columns(3)

# ... (Dein restlicher Parameter-Code bleibt gleich bis zum Execution Button) ...

with col1:
    st.subheader("Kapazität & Leistung")
    # Wir nutzen 'key', damit Streamlit den Wert direkt im session_state verwaltet
    cap = st.number_input(
        'Max. Kapazität (kWh)', 
        value=float(st.session_state['capacity_kwh']),
        key='capacity_kwh_input' 
    )
    p_charge = st.number_input(
        'Max. Ladeleistung (kW)', 
        value=float(st.session_state['p_charge_kw']),
        key='p_charge_kw_input'
    )
    p_discharge = st.number_input(
        'Max. Entladeleistung (kW)', 
        value=float(st.session_state['p_discharge_kw']),
        key='p_discharge_kw_input'
    )

with col2:
    st.subheader("Effizienz & Alterung")
    eta_c_pct = st.slider(
        'Lade-Wirkungsgrad (%)',
        0, 100,
        int(params_base['eta_charge'] * 100)
    )
    eta_c = eta_c_pct / 100

    eta_d_pct = st.slider(
        'Entlade-Wirkungsgrad (%)',
        0, 100,
        int(params_base['eta_discharge'] * 100)
    )
    eta_d = eta_d_pct / 100

    soc_min_pct = st.slider(
        'Min. SoC (%)',
        0, 100,
        int(params_base['soc_min_frac'] * 100)
    )
    soc_min = soc_min_pct / 100

    soc_max_pct = st.slider(
        'Max. SoC (%)',
        0, 100,
        int(params_base['soc_max_frac'] * 100)
    )
    soc_max = soc_max_pct / 100

    st.markdown("---")

    c_loss_pct = st.slider(
        'Zyklenverlust (%)',
        0.0, 100.0,
        float(params_base['cyclic_loss'] * 100),
        step=0.5
    )
    c_loss = c_loss_pct / 100

    cal_loss_pct = st.slider(
        'Kalendarischer Verlust / Jahr (%)',
        0.0, 10.0,
        float(params_base['calendar_loss'] * 100),
        step=0.1
    )
    cal_loss = cal_loss_pct / 100

with col3:
    st.subheader("Ökonomie")
    lifetime = st.number_input(
        'Nutzungsdauer (Jahre)',
        value=int(params_base['lifetime_yr'])
    )
    capex_kwh = st.number_input(
        'CAPEX Energie (€/kWh)',
        value=float(params_base['capex_per_kwh'])
    )
    capex_kw = st.number_input(
        'CAPEX Leistung (€/kW)',
        value=float(params_base['capex_per_kw'])
    )
    opex_kwh = st.number_input(
        'OPEX Energie (€/kWh/Jahr)',
        value=float(params_base['opex_per_kwhyr'])
    )
    opex_kw = st.number_input(
        'OPEX Leistung (€/kW/Jahr)',
        value=float(params_base['opex_per_kwyr'])
    )
    feedin = st.number_input(
        'Einspeisefaktor',
        value=float(params_base['feedin_fraction'])
    )
    discount = st.number_input(
        'Discount Rate',
        value=float(params_base['discount_rate'])
    )


# Dictionary aktualisieren
params_base.update({
    'capacity_kwh': cap, 'p_charge_kw': p_charge, 'p_discharge_kw': p_discharge,
    'soc_min_frac': soc_min, 'soc_max_frac': soc_max, 'eta_charge': eta_c, 'eta_discharge': eta_d,
    'capex_per_kwh': capex_kwh, 'capex_per_kw': capex_kw, 'lifetime_yr': lifetime,
    'discount_rate': discount, 'opex_per_kwhyr': opex_kwh, 'opex_per_kwyr': opex_kw, 'feedin_fraction': feedin,
    'cyclic_loss': c_loss, 'calendar_loss': cal_loss
})

# --- NEU: PREIS-PROFIL VORSCHAU ---
st.header("📉 Preisprofil")

# Wir laden die Profile kurz vorab für die Visualisierung
with st.spinner('Lade Profile für Vorschau...'):
    load_preview, gen_preview = load_profile(
        load_csv_path, 
        gen_csv_path, 
        lat=lat_input, 
        lon=lon_input, 
        peak_kw=peak_pv_input, 
        peak_load=peak_load_input
    )
    price_preview = generate_price_profile(load_preview.index, 'SpotPreis_2018_2025.csv')

# Plot des Preisprofils (z.B. die ersten 7 Tage zur Übersicht)
import plotly.express as px
fig_price = px.line(
    x=price_preview.index, 
    y=price_preview.values, 
    labels={'x': 'Zeit', 'y': 'Preis (€/kWh)'},
    title="Börsenstrompreis (Spotmarkt)",
    line_shape="hv"
)
fig_price.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
# Optional: Nur die erste Woche anzeigen für bessere Performance beim Laden
# fig_price.update_xaxes(range=[price_preview.index[0], price_preview.index[min(len(price_preview)-1, 672)]])

st.plotly_chart(fig_price, use_container_width=True)
# --- EXECUTION BUTTON ---
if st.button('🚀 Optimierung und Simulation starten'):
    with st.spinner('Berechne Optimierung...'):
        start_time = time.time()
        
        # --- ANGEPASST: Nutzt nun die Sidebar-Inputs ---
        load_series, gen_series = load_profile(
            load_csv_path, 
            gen_csv_path, 
            lat=lat_input, 
            lon=lon_input, 
            peak_kw=peak_pv_input, 
            peak_load=peak_load_input
        )
        
        price_profile = generate_price_profile(load_series.index, 'SpotPreis_2018_2025.csv')

        best, df_eval = optimize_battery_size(
            load_series, gen_series, price_profile, params_base, 
            cap_range_kwh=(0, params_base['capacity_kwh']), 
            p_cha_range_kw=(0, params_base['p_charge_kw']), 
            p_dis_range_kw=(0, params_base['p_discharge_kw'])
        )

        best_params = best['params']
        best_params.update({
            'capacity_kwh': float(best['capacity_kwh']),
            'p_charge_kw': float(best['p_charge_kw']),
            'p_discharge_kw': float(best['p_discharge_kw'])
        })

        sim_best = simulate_battery(load_series, gen_series, best_params)
        econ_best = evaluate_annual_costs(sim_best, best_params, price_profile)

        params_nobatt = params_base.copy()
        params_nobatt.update({'capacity_kwh': 0.0, 'p_charge_kw': 0.0, 'p_discharge_kw': 0.0})
        sim_nobatt = simulate_battery(load_series, gen_series, params_nobatt)
        econ_nobatt = evaluate_annual_costs(sim_nobatt, params_nobatt, price_profile)

        total_load_kwh = load_series.sum() * params_base.get('timestep_hours', 0.25)
        autarky_best = 1.0 - (sim_best['total_import_kwh'] / total_load_kwh) if total_load_kwh > 0 else 0.0
        autarky_nobatt = 1.0 - (sim_nobatt['total_import_kwh'] / total_load_kwh) if total_load_kwh > 0 else 0.0
        runtime = time.time() - start_time

        # --- ERGEBNISSE ANZEIGEN ---
        st.success(f"Berechnung abgeschlossen in {format_de(runtime, 2)} Sekunden")
        
        st.header("📊 Analyse-Ergebnisse")
        res1, res2, res3, res4 = st.columns(4)
        
        # Spalte 1: Dimensionierung
        res1.metric("Optimale Kapazität", f"{format_de(best['capacity_kwh'], 1)} kWh")
        res1.metric("Max. Ladeleistung", f"{format_de(best['p_charge_kw'], 1)} kW")
        res1.metric("Max. Entladeleistung", f"{format_de(best['p_discharge_kw'], 1)} kW")

        # Spalte 2: Technik
        cap_loss_pct = (1 - best['soh_final'] / best_params['capacity_kwh']) * 100
        duration_days = (load_series.index[-1] - load_series.index[0]).total_seconds() / (24 * 3600)
        scaling_factor = 365.0 / duration_days if duration_days > 0 else 1.0
        yearly_th = (sim_best.get('throughput_kwh', 0.0) * scaling_factor) / 1000
        
        res2.metric("Durchsatz (Jahr)", f"{format_de(yearly_th, 1)} MWh")
        res2.metric("Finaler SoH", f"{format_de(best['soh_final'], 1)} kWh", 
                    delta=f"-{format_de(cap_loss_pct, 2)} %", delta_color="normal")

        # Spalte 3: Ökonomie
        cost_with = econ_best['total_annual_cost']
        cost_no = econ_nobatt['total_cost_no_batt']
        rel_savings = ((cost_with / cost_no) - 1) * 100 if cost_no != 0 else 0
        
        res3.metric("Jährliche Ersparnis", f"{format_de(econ_best['savings_vs_no_batt'], 2)} €")
        res3.metric("Kosten mit BESS", f"{format_de(cost_with, 2)} €/a", 
                    delta=f"{format_de(rel_savings, 1)} %", delta_color="inverse")
        res3.metric("Kosten ohne BESS", f"{format_de(econ_nobatt['total_cost_no_batt'], 2)} €/a")

        # Spalte 4: Autarkie
        res4.metric("Autarkiegrad", f"{format_de(autarky_best*100, 1)} %", 
                    f"+{format_de((autarky_best-autarky_nobatt)*100, 1)} %")
        # --- PLOTTING ---
        st.header("📈 Interaktive Detailanalyse")
        
        fig = plot_interactive_full_period(sim_best)

        # WICHTIG: Hier nutzen wir 'timeline', da deine Funktion so benannt ist
        if 'timeline' in sim_best:
            df_plot = sim_best['timeline']
            
            # Mitte des Zeitraums finden
            mid_index = len(df_plot) // 2
            
            # Bereich festlegen: 1 Woche (bei 15-Min-Intervallen sind 672 Zeilen = 7 Tage)
            # Wir nehmen 336 Schritte (3,5 Tage) vor und nach der Mitte
            start_zoom = df_plot.index[max(0, mid_index - 336)]
            end_zoom = df_plot.index[min(len(df_plot)-1, mid_index + 336)]

            # Den initialen Zoom-Bereich in Plotly setzen
            fig.update_xaxes(range=[start_zoom, end_zoom])

        # Bedienungshinweise für deine Nutzer
        st.info("""
        **🔍 Bedienungsanleitung:**
        * **Zoom:** linke Maustaste
        * **Verschieben:** Shift + linke Maustaste
        * **Zurück zur Übersicht:** Doppelklick
        * **Zur Jahresansicht:** 2 x Doppelklick
                
        """)

        st.plotly_chart(fig, use_container_width=True)

