import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import io, os, math
import plotly.graph_objects as go
from plotly.subplots import make_subplots


plt.ion()   # interaktiver Modus




def plot_example_week(load_series, sim_best, week=1):
    timeline = sim_best['timeline']
    data_start = timeline.index.min().normalize()
    week_start = data_start + timedelta(days=(week - 1) * 7)
    week_end = week_start + timedelta(days=7)

    df_plot = timeline.loc[week_start:week_end].copy()
    
    # Erstellt 3 Subplots (übereinander, wie in deinem Matplotlib-Code)
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        subplot_titles=(f"Last & Erzeugung", "Batterie Füllstand (SoC)", "Energiefluss-Zusammensetzung")
    )

    # --- PLOT 1: Load & Gen ---
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['load_kW'], name="Last", line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['gen_kW'], name="PV Erzeugung", line=dict(color='orange')), row=1, col=1)

    # --- PLOT 2: SoC ---
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['soc_kWh'], name="SoC (kWh)", fill='tozeroy', line=dict(color='blue')), row=2, col=1)

    # --- PLOT 3: Stacked Energy Flows ---
    dt_h = 0.25  # Annahme 15-Min-Takt
    
    # Definition der Flächen (Reihenfolge wie in deinem ax[2].stackplot)
    # Wichtig: 'stackgroup="one"' sorgt für das Stapeln
    flows = [
        ('pv_to_load_kW', 'PV zu Last', "#F6FF00"),
        ('discharge_kW', 'Batterie Entladung', "#0026FF"),
        ('charge_kW', 'Batterie Ladung (PV)', "#001AAE"),
    ]
    
    for col, label, color in flows:
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot[col], name=label,
            stackgroup='one', line=dict(width=0, color=color), fillcolor=color
        ), row=3, col=1)

    # Netz-Bezug und Export (werden oft getrennt betrachtet, hier als Linien oder weitere Stacks)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['import_kWh']/dt_h, name="Netzbezug", stackgroup='one', line=dict(width=0, color="#747474"), fillcolor="#747474"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['export_kWh']/dt_h, name="Einspeisung", stackgroup='one', line=dict(width=0, color="#FFBB00"), fillcolor="#FFBB00"), row=3, col=1)

    # Die schwarze gestrichelte Last-Linie zur Orientierung im Stackplot
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['load_kW'], name="Last Referenz", line=dict(color='black', dash='dash')), row=3, col=1)

    # Layout-Anpassungen
    fig.update_layout(height=900, hovermode="x unified", showlegend=True, title_text=f"Woche {week} Detailanalyse")
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="kWh", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=3, col=1)
    
    return fig


def plot_interactive_full_period(sim_best):
    """
    Kompakte Version ohne Rangeslider.
    Mit intelligenten Zeitachsen (Uhrzeiten erscheinen beim Reinzoomen).
    """
    df_plot = sim_best['timeline']
    dt_h = 0.25
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.07,
        subplot_titles=("Last & Erzeugung", "Batterie SoC", "Lastflüsse")
    )

    # --- PLOT 1: Load & Gen ---
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['load_kW'], name="Last", 
                             line=dict(color='black', width=1.5), legend='legend'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['gen_kW'], name="PV Erzeugung", 
                             line=dict(color='orange', width=1.5), legend='legend'), row=1, col=1)

    # --- PLOT 2: SoC ---
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['soc_kWh'], name="SoC (kWh)", 
                             fill='tozeroy', line=dict(color='blue'), legend='legend2'), row=2, col=1)

    # --- PLOT 3: Lastflüsse ---
    flows = [('pv_to_load_kW', 'PV zu Last', "#F6FF00"),
             ('discharge_kW', 'Batterie Entladung', "#00A6FF"),
             ('charge_kW', 'Batterie Ladung (PV)', "#001AAE")]
    
    for col, label, color in flows:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], name=label, stackgroup='one', 
                                 line=dict(width=0, color=color), fillcolor=color, legend='legend3'), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['import_kWh']/dt_h, name="Netzbezug", 
                             stackgroup='one', line=dict(width=0, color="#747474"), fillcolor="#747474", legend='legend3'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['export_kWh']/dt_h, name="Einspeisung", 
                             stackgroup='one', line=dict(width=0, color="#FFBB00"), fillcolor="#FFBB00", legend='legend3'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['load_kW'], name="Last Referenz", 
                             line=dict(color='black', dash='dash', width=1), legend='legend3'), row=3, col=1)

    # --- LAYOUT OPTIMIERUNG ---
    fig.update_layout(
        height=850,
        template="plotly_white",
        dragmode="zoom",
        hovermode="x unified",
        margin=dict(l=60, r=160, t=60, b=80),
        
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=1.02),
        legend2=dict(yanchor="top", y=0.65, xanchor="left", x=1.02),
        legend3=dict(yanchor="top", y=0.32, xanchor="left", x=1.02)
    )

    # --- INTELLIGENTE X-ACHSE ---
    fig.update_xaxes(
        row=3, col=1,
        type="date",
        showticklabels=True,
        gridcolor="lightgrey",
        # tickformatstops definiert das Format je nach Zoom-Level
        tickformatstops=[
            dict(dtickrange=[None, 1000], value="%H:%M:%S.%L ms"), # Millisekunden
            dict(dtickrange=[1000, 60000], value="%H:%M:%S s"),   # Sekunden
            dict(dtickrange=[60000, 3600000], value="%H:%M"),     # Minuten
            dict(dtickrange=[3600000, 86400000], value="%H:%M\n%d.%b"), # Stunden & Tag
            dict(dtickrange=[86400000, 604800000], value="%d.%m."),     # Tage
            dict(dtickrange=[604800000, "M1"], value="%d.%m."),         # Wochen
            dict(dtickrange=["M1", "M12"], value="%b %Y"),              # Monate
            dict(dtickrange=["M12", None], value="%Y")                  # Jahre
        ],
        rangeselector=dict(
            visible=True,
            bgcolor="rgba(0,0,0,0)",
            buttons=list([])
        )
    )
    
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="kWh", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=3, col=1)
    
    return fig
