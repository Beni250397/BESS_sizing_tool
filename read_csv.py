import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io, os, math
from parameters import params_base

def load_profile(load_input, gen_input, lat=50.1319, lon=8.6838, peak_kw=20, peak_load=20, target_year=2025):
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
        gen_series = generate_default_gen(load_series.index, lat, lon, peak_kw)
        
    elif not gen_series.empty:
        # Nur PV vorhanden -> Last-Default an PV anpassen
        print(f"ℹ️ Erzeuge Last-Default passend zum PV-Zeitraum ({gen_series.index.min().date()} bis {gen_series.index.max().date()})")
        load_series = generate_default_load(gen_series.index, peak_load)

    else:
        # Gar nichts da -> Full Default für das Zieljahr (Wichtig für dein "None"-Kriterium)
        print(f"ℹ️ Keine Daten übergeben. Erzeuge Default für Kalenderjahr {target_year}")
        full_idx = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31 23:45:00", freq='15min')
        load_series = generate_default_load(full_idx, peak_load)
        gen_series = generate_default_gen(full_idx, lat, lon, peak_kw)

    return load_series, gen_series



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

def generate_default_load(idx, peak_load=20, seed=0):
    rng = np.random.default_rng(seed)

    hours = np.array([t.hour + t.minute/60.0 for t in idx])
    day_of_year = np.array([t.timetuple().tm_yday for t in idx])
    weekday = np.array([t.weekday() for t in idx])  # 0=Mo, 6=So

    # -----------------------------
    # 1. Saisonale Grundlast
    # -----------------------------
    # Winter höher (WP, Beleuchtung), Sommer niedriger
    seasonal = 0.5 + 0.5 * np.cos((day_of_year - 15) / 365 * 2 * np.pi)
    base = 0.6 + 1.8 * seasonal   # ca. 0.6–2.4 kW

    # -----------------------------
    # 2. Tagesstruktur (Haushalt)
    # -----------------------------
    morning_peak = 1.2 * np.exp(-((hours - 7.5) / 1.6)**2)
    evening_peak = 3.8 * np.exp(-((hours - 19.5) / 2.8)**2)

    # -----------------------------
    # 3. Wochenend-Effekt
    # -----------------------------
    weekend_factor = np.where(weekday >= 5, 0.9, 1.0)
    base *= weekend_factor
    morning_peak *= np.where(weekday >= 5, 0.6, 1.0)
    evening_peak *= np.where(weekday >= 5, 1.1, 1.0)

    # -----------------------------
    # 4. EV-Laden (nicht jeden Tag!)
    # -----------------------------
    ev_load = np.zeros(len(idx))
    days = len(idx) // 96
    ev_days = rng.choice(days, size=int(0.35 * days), replace=False)  # ~2–3 Tage/Woche

    for d in ev_days:
        start = d * 96 + rng.integers(72, 84)  # 18–21 Uhr
        duration = rng.integers(8, 16)        # 2–4 h
        power = rng.uniform(3.5, 7.0)         # Wallbox gedrosselt
        ev_load[start:start+duration] += power

    # -----------------------------
    # 5. Kurzzeitrauschen
    # -----------------------------
    noise = rng.normal(0, 0.15, size=len(idx))

    # -----------------------------
    # 6. Gesamtlast
    # -----------------------------
    load = base + morning_peak + evening_peak + ev_load + noise

    # Sanftes Glätten (realistisch für 15-min-Mittelwerte)
    load = pd.Series(load).rolling(3, center=True, min_periods=1).mean().values

    # -----------------------------
    # 7. Skalierung auf Peak
    # -----------------------------
    load = load / load.max() * peak_load

    return pd.Series(np.clip(load, 0.3, None), index=idx, name="load_kW")


def generate_default_gen(idx, lat=50.1319, lon=8.6838, peak_kw=20.0):
    day_of_year = idx.dayofyear
    local_hour = idx.hour + idx.minute/60.0
    
    # Deklination
    decl = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))
    
    # Stundenwinkel (15° pro Stunde)
    # Wir korrigieren die Lokalzeit auf die wahre Sonnenzeit:
    # 15 Grad * (Stunde - 12) + (Längengrad - Zeitzonen_Meridian)
    # Für Mitteleuropa (CET) ist der Referenzmeridian 15.0
    hour_angle = (local_hour - 12) * 15 + (lon - 15)
    
    lat_rad = np.radians(lat)
    decl_rad = np.radians(decl)
    hour_angle_rad = np.radians(hour_angle)
    
    # Zenitwinkel-Formel
    cos_zenith = (np.sin(lat_rad) * np.sin(decl_rad) + 
                  np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle_rad))
    
    sun_intensity = np.maximum(0, cos_zenith)

    # NEU: Wolken-Simulation
    # Wir erzeugen ein Rauschen, das über den Tag hinweg korreliert ist (nicht nur wildes Flackern)
    rng = np.random.default_rng(seed=42)
    days = len(idx) // 96 # 15-min Schritte pro Tag
    # Ein Faktor pro Tag (gut/schlecht Wetter Perioden)
    daily_cloud_factor = rng.uniform(0.3, 1.0, size=days+1)
    cloud_mask = np.repeat(daily_cloud_factor, 96)[:len(idx)]
    
    # Kurzzeitiges Flackern hinzufügen
    short_term_noise = rng.uniform(0.8, 1.0, size=len(idx))
    
     # Exponent 1.5 für atmosphärische Absorption (Air Mass Effekt)
    gen = peak_kw * (sun_intensity**1.5) * cloud_mask * short_term_noise
    return pd.Series(gen, index=idx, name='gen_kW')
    

    # hours = np.array([t.hour + t.minute/60.0 for t in idx])
    # day_of_year = np.array([t.timetuple().tm_yday for t in idx])

    # # 20 kWp Anlage (realistisch skaliert)
    # pv_capacity = 20.0 
    # decl = (0.5 + 0.5 * np.cos((day_of_year-172)/365*2*np.pi))
    
    # # Realistischere Glockenkurve
    # sun = np.maximum(0, np.sin((hours-6)/12*np.pi))**1.5
    
    # gen = pv_capacity * decl * sun
    # return pd.Series(gen, index=idx, name='gen_kW')


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
    n_neg = max(1, int(len(idx)*0.005))
    neg_mask = rng.choice(len(idx), size=n_neg, replace=False)

    
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
