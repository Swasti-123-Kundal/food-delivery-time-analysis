"""
Food Delivery Time Analysis
Dataset: Kaggle "Food Delivery Dataset" (train.csv) - ~45K orders
Run: python delivery_analysis.py train.csv
Outputs: cleaned CSV, 5 charts (PNG) and insights.txt in ./output
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
os.makedirs("output", exist_ok=True)

# ---------- 1. LOAD + CLEAN ----------
df = pd.read_csv(path)
df.columns = [c.strip() for c in df.columns]
print("Raw shape:", df.shape)

# strip whitespace in text columns, turn 'NaN ' strings into real NaN
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip().replace({"NaN": np.nan, "nan": np.nan})

# target: '(min) 24' -> 24
df["time_taken"] = (
    df["Time_taken(min)"].astype(str).str.extract(r"(\d+)")[0].astype(float)
)
df["weather"] = df["Weatherconditions"].astype(str).str.replace("conditions ", "", regex=False)

for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# time features
df["order_date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
df["order_time"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce")
df["hour"] = df["order_time"].dt.hour
df["weekday"] = df["order_date"].dt.day_name()

# distance (haversine, km) between restaurant and delivery location
def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine(
    df["Restaurant_latitude"], df["Restaurant_longitude"],
    df["Delivery_location_latitude"], df["Delivery_location_longitude"],
)
# some rows have (0,0) coordinates -> invalid
df.loc[(df["Restaurant_latitude"] == 0) | (df["distance_km"] > 50), "distance_km"] = np.nan

df = df.dropna(subset=["time_taken"]).drop_duplicates()
print("Clean shape:", df.shape)
df.to_csv("output/cleaned_data.csv", index=False)

# define "late" = slower than 30 min (change threshold if you want)
LATE = 30
df["is_late"] = (df["time_taken"] > LATE).astype(int)

insights = []
insights.append(f"Orders analysed: {len(df):,}")
insights.append(f"Average delivery time: {df['time_taken'].mean():.1f} min")
insights.append(f"Orders taking > {LATE} min: {df['is_late'].mean()*100:.1f}%")

# ---------- 2. GROUP ANALYSIS ----------
def group_report(col, title):
    g = (df.groupby(col)
           .agg(orders=("time_taken", "size"),
                avg_time=("time_taken", "mean"),
                late_pct=("is_late", lambda s: s.mean() * 100))
           .sort_values("avg_time", ascending=False)
           .round(1))
    g.to_csv(f"output/by_{col}.csv")
    top = g.index[0]
    insights.append(
        f"{title}: worst = '{top}' (avg {g.loc[top,'avg_time']} min, "
        f"{g.loc[top,'late_pct']}% late)"
    )
    return g

traffic = group_report("Road_traffic_density", "Traffic")
weather = group_report("weather", "Weather")
city = group_report("City", "City type")
festival = group_report("Festival", "Festival")
multi = group_report("multiple_deliveries", "Multiple deliveries carried")
vehicle = group_report("Type_of_vehicle", "Vehicle")

# distance bands
df["dist_band"] = pd.cut(df["distance_km"], [0, 5, 10, 15, 20, 50],
                         labels=["0-5", "5-10", "10-15", "15-20", "20+"])
dist = group_report("dist_band", "Distance band (km)")

# rating bands
df["rating_band"] = pd.cut(df["Delivery_person_Ratings"], [0, 3.5, 4.0, 4.5, 5.0],
                           labels=["<=3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0"])
rating = group_report("rating_band", "Rider rating band")

# hour of day
hour = (df.groupby("hour")["time_taken"].mean().round(1))
hour.to_csv("output/by_hour.csv")
if hour.notna().any():
    insights.append(f"Peak slow hour: {int(hour.idxmax())}:00 (avg {hour.max()} min)")

# correlations
corr = df[["time_taken", "distance_km", "Delivery_person_Age",
           "Delivery_person_Ratings", "multiple_deliveries"]].corr()["time_taken"].round(2)
insights.append("Correlation with delivery time:\n" + corr.to_string())

# combined effect: traffic x weather (late %)
combo = (df.pivot_table(index="Road_traffic_density", columns="weather",
                        values="is_late", aggfunc="mean") * 100).round(1)
combo.to_csv("output/traffic_x_weather_late_pct.csv")

# ---------- 3. CHARTS ----------
def bar(series, title, ylabel, fname, color="#c0392b"):
    plt.figure(figsize=(7, 4))
    series.plot(kind="bar", color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"output/{fname}", dpi=150)
    plt.close()

bar(traffic["avg_time"], "Avg delivery time by traffic density", "minutes", "1_traffic.png")
bar(weather["avg_time"], "Avg delivery time by weather", "minutes", "2_weather.png")
bar(dist["avg_time"].sort_index(), "Avg delivery time by distance band (km)", "minutes", "3_distance.png")
bar(multi["avg_time"].sort_index(), "Avg delivery time by # of parallel deliveries", "minutes", "4_multi.png")

plt.figure(figsize=(8, 4))
hour.plot(marker="o", color="#2c3e50")
plt.title("Avg delivery time by hour of order")
plt.xlabel("hour of day")
plt.ylabel("minutes")
plt.tight_layout()
plt.savefig("output/5_hour.png", dpi=150)
plt.close()

with open("output/insights.txt", "w") as f:
    f.write("\n\n".join(insights))
print("\n\n".join(insights))
print("\nDone. Check the ./output folder.")
