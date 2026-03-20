"""
Z3 Theory: 3-Year KM3NeT Ultra-High-Energy Neutrino Transparent Windows
Full English Output + Clean Table Format
Author: Optimized for you | March 2026
"""

import numpy as np
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import EarthLocation
import pandas as pd
from datetime import datetime

print("=== Z3 Vacuum Lattice Theory: KM3NeT >100 PeV Neutrino Windows (2026–2029) ===")
print("Core Principle: Events are only allowed in narrow sidereal time windows.")
print("Any detection outside these windows falsifies the Z3 geometric channeling model.\n")

# KM3NeT ARCA location and reference event
location = EarthLocation(lat=36.42*u.deg, lon=15.17*u.deg)
t_ref = Time('2023-02-13T01:16:47', scale='utc')
lst_ref = t_ref.sidereal_time('mean', longitude=location.lon).hour

window_width_h = 1.0          # 1-hour window (±30 min) around the transparent LST
lst_min = (lst_ref - window_width_h/2) % 24
lst_max = (lst_ref + window_width_h/2) % 24

print(f"Reference Transparent LST : {lst_ref:.3f} hours")
print(f"Allowed Window (Z3 Channel) : {lst_min:.2f} – {lst_max:.2f} LST every sidereal day")
print(f"Dead Zone (Z3 Blocked)      : All other LST hours\n")

# Generate table for next 3 years (daily windows)
start_date = Time('2026-03-20T00:00:00', scale='utc')
days = 1096  # 3 years ≈ 1096 days

data = []

for i in range(days):
    t = start_date + i * u.day
    lst_0 = t.sidereal_time('mean', longitude=location.lon).hour
    hours_to_add = (lst_ref - lst_0) % 24
    utc_hours = hours_to_add * 0.997269566  # Convert sidereal to solar time
    
    t_peak = t + utc_hours * u.hour
    t_start = t_peak - (window_width_h/2 * 0.997269) * u.hour
    t_end   = t_peak + (window_width_h/2 * 0.997269) * u.hour
    
    data.append({
        "Date (UTC)": t.strftime("%Y-%m-%d"),
        "Allowed Start (UTC)": t_start.strftime("%H:%M:%S"),
        "Allowed End (UTC)": t_end.strftime("%H:%M:%S"),
        "Center Time (UTC)": t_peak.strftime("%H:%M:%S"),
        "Status": "✅ Z3 Transparent Window (Allowed)",
        "Note": "Detection permitted only in this window"
    })

df = pd.DataFrame(data)

# ==================== 输出漂亮表格 ====================
print("=== Full 3-Year Prediction Table (Sample: First 10 + Last 10 days) ===")
print(df.head(10).to_string(index=False))
print("\n...\n")
print(df.tail(10).to_string(index=False))

# 保存完整表格
df.to_csv("Z3_KM3NeT_3Year_Transparent_Windows.csv", index=False)
print("\n✅ Complete 3-year table saved to: Z3_KM3NeT_3Year_Transparent_Windows.csv")
print("   You can open it in Excel for full 1096 rows.")

print("\n=== Summary for Experimentalists ===")
print("• Allowed detections must occur exclusively within the daily Z3 Transparent Window.")
print("• Any >100 PeV event detected outside this window falsifies the geometric channeling hypothesis.")
print("• The sterile neutrino / Earth-matter resonance models predict no such sidereal restriction.")
print("\nThe ball is now in the experimentalists' court.")