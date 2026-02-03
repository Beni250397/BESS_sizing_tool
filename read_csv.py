import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io, os, math

def read_quarter_hour_csv(path):
    try:
        # Einlesen mit automatischer Erkennung (Komma, Semikolon etc.)
        df = pd.read_csv(path, sep=None, engine='python')
        df.columns = [c.strip() for c in df.columns]

        # Spezialbehandlung für den "Monat 00" Fehler in deiner CSV:
        # Wir ersetzen -00- durch -01-, damit pandas das Datum versteht.
        if df['Datum'].dtype == object:
            df['Datum'] = df['Datum'].str.replace('-00-', '-01-')

        # Datum und Zeit kombinieren
        df['Datum_Zeit'] = pd.to_datetime(
            df['Datum'].astype(str) + ' ' + df['Zeit'].astype(str)
        )
        df.set_index('Datum_Zeit', inplace=True)

        target_col = 'Wirkleistung [kW]'
        if target_col not in df.columns:
            # Fallback auf die erste numerische Spalte
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[0]
            else:
                raise ValueError(f"Keine Leistungsdaten in {path} gefunden.")

        series = pd.to_numeric(df[target_col], errors='coerce').dropna()

        # --- Resampling auf 15min ---
        # 1. Erstelle Ziel-Index (15 Min)
        start = series.index.min()
        end = series.index.max()
        quarter_index = pd.date_range(start=start, end=end, freq='15min')
        
        # 2. Reindex & Interpolation
        # Da es Wirkleistung (kW) ist, ist eine lineare Interpolation sinnvoll
        series_15min = series.reindex(quarter_index).interpolate(method='linear')

        return series_15min.ffill().bfill()

    except Exception as e:
        print(f"Fehler beim Einlesen von {path}: {e}")
        return pd.Series(dtype=float)




def load_profile(load_input, gen_input, target_year=2025):
    load_series = pd.Series(dtype=float)
    gen_series = pd.Series(dtype=float)

    # 1. Dateien einlesen
    # Wir prüfen nicht mehr auf os.path.exists, da UploadedFile kein Pfad ist.
    # Wenn load_input nicht None ist, versuchen wir es zu lesen.
    if load_input is not None:
        try:
            load_series = read_quarter_hour_csv(load_input)
            print(f"✅ Lastprofil erfolgreich geladen.")
        except Exception as e:
            print(f"❌ Fehler beim Laden des Lastprofils: {e}")

    if gen_input is not None:
        try:
            gen_series = read_quarter_hour_csv(gen_input)
            print(f"✅ Erzeugungsprofil erfolgreich geladen.")
        except Exception as e:
            print(f"❌ Fehler beim Laden des Erzeugungsprofils: {e}")

    # 2. Zeitrahmen bestimmen (Logik bleibt gleich)
    if not load_series.empty and not gen_series.empty:
        # Falls beide da sind, nehmen wir den Überschneidungsbereich
        start_date = max(load_series.index.min(), gen_series.index.min())
        end_date = min(load_series.index.max(), gen_series.index.max())
        common_idx = pd.date_range(start_date, end_date, freq='15min')
        load_series = load_series.reindex(common_idx).interpolate()
        gen_series = gen_series.reindex(common_idx).interpolate()
        print(f"✅ Kombiniere beide Profile von {start_date} bis {end_date}")

    elif not load_series.empty:
        # Nur Last vorhanden -> PV-Default an Last anpassen
        print(f"ℹ️ Erzeuge PV-Default passend zum Last-Zeitraum ({load_series.index.min().date()} bis {load_series.index.max().date()})")
        gen_series = generate_default_gen(load_series.index)
        
    elif not gen_series.empty:
        # Nur PV vorhanden -> Last-Default an PV anpassen
        print(f"ℹ️ Erzeuge Last-Default passend zum PV-Zeitraum ({gen_series.index.min().date()} bis {gen_series.index.max().date()})")
        load_series = generate_default_load(gen_series.index)

    else:
        # Gar nichts da -> Full Default für das Zieljahr (Wichtig für dein "None"-Kriterium)
        print(f"ℹ️ Keine Daten übergeben. Erzeuge Default für Kalenderjahr {target_year}")
        full_idx = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31 23:45:00", freq='15min')
        load_series = generate_default_load(full_idx)
        gen_series = generate_default_gen(full_idx)

    return load_series, gen_series

def generate_default_load(idx, seed=0):
    rng = np.random.default_rng(seed)
    hours = np.array([t.hour + t.minute/60.0 for t in idx])
    day_of_year = np.array([t.timetuple().tm_yday for t in idx])

    # Höhere Grundlast durch Wärmepumpe/Standby (ca. 2.5 kW Base im Winter)
    seasonal_factor = (1 + np.cos((day_of_year-15)/365*2*np.pi))/2
    base = 1.0 + 1.5 * seasonal_factor 
    
    # Abend-Peak verstärkt (Kochen + E-Auto laden ab 18 Uhr)
    morning_peak = 1.5 * np.exp(-((hours-7)/1.5)**2)
    evening_peak = 4.5 * np.exp(-((hours-19)/3.0)**2) # Erhöht auf 4.5kW

    noise = rng.normal(0, 0.2, size=len(idx))
    load = base + morning_peak + evening_peak + noise
    return pd.Series(np.clip(load, 0.3, None), index=idx, name='load_kW')

def generate_default_gen(idx):
    hours = np.array([t.hour + t.minute/60.0 for t in idx])
    day_of_year = np.array([t.timetuple().tm_yday for t in idx])

    # 20 kWp Anlage (realistisch skaliert)
    pv_capacity = 20.0 
    decl = (0.5 + 0.5 * np.cos((day_of_year-172)/365*2*np.pi))
    
    # Realistischere Glockenkurve
    sun = np.maximum(0, np.sin((hours-6)/12*np.pi))**1.5
    
    gen = pv_capacity * decl * sun
    return pd.Series(gen, index=idx, name='gen_kW')


# ------------------ Price profile generator ------------------


def generate_price_profile(idx):
    """
    Realistisches Preisprofil 2026 für Gewerbe/Privat (20 kWp Bereich).
    Beinhaltet Beschaffung + Netzentgelte + Steuern.
    """
    hours = np.array([t.hour + t.minute/60.0 for t in idx])
    day_of_year = np.array([t.timetuple().tm_yday for t in idx])
    
    # 1. Grundpreis (Alles inklusive: Energie + Netzentgelte + Umlagen)
    # 2026 realistisch für KMU/Privat: ca. 28 - 32 Cent/kWh
    base_price = 0.28  
    
    # 2. Die "Entenkurve" (Solar-Dip am Mittag, Peak am Abend)
    # Mittags sinkt der Preis durch PV-Überangebot im Netz
    solar_dip = -0.06 * np.exp(-((hours-13)/2.5)**2) 
    
    # Abend-Peak (wenn alle kochen und die Sonne weg ist)
    evening_peak = 0.08 * np.exp(-((hours-19)/3.0)**2)
    
    # 3. Saisonalität
    # Im Winter ist Strom tendenziell teurer (weniger PV, mehr Heizlast)
    seasonal = 0.02 * np.cos((day_of_year - 15) / 365 * 2 * np.pi)
    
    # 4. Negative Preise (Extremereignisse im Sommer bei Wind + Sonne)
    rng = np.random.default_rng(42)
    # Ca. 0.5% der Zeit (ca. 45h im Jahr) gibt es Preisstürze
    neg_mask = rng.choice(len(idx), size=int(len(idx)*0.005), replace=False)
    
    price = base_price + solar_dip + evening_peak + seasonal
    
    # Bei negativen Preisen fällt oft nur der Energieteil weg, 
    # Steuern/Netzentgelte bleiben meist (daher selten echt negativ für Endkunden)
    price[neg_mask] = 0.05 
    
    return pd.Series(price, index=idx, name='price_EUR_per_kWh')
# def load_price_profile_from_csv(csv_path, idx):
#     df = pd.read_csv(csv_path, parse_dates=["timestamp"])
#     df = df.set_index("timestamp")

#     # auf deinen Simulationsindex bringen
#     price = df["price_EUR_per_kWh"].reindex(idx)

#     # falls Lücken: auffüllen
#     price = price.interpolate().ffill().bfill()

#     return price
