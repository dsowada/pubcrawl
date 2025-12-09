import os
import math
import requests
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium


# ==============================
# Distanz- & Geocoding-Funktionen
# ==============================

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Luftliniendistanz in Kilometern (Fallback nur für Notfälle).
    """
    R = 6371  # Erdradius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjBjZTg0MmEwMDk0NjRkY2RiNzYzM2Q0NjBiZmJhN2EwIiwiaCI6Im11cm11cjY0In0="

def get_ors_api_key():
    """
    Holt den ORS-API-Key entweder aus der Umgebung oder aus st.secrets.
    Gibt None zurück, wenn kein Key vorhanden ist.
    """
    api_key = os.environ.get("ORS_API_KEY")
    if api_key:
        return api_key

    try:
        api_key = st.secrets["ORS_API_KEY"]
        return api_key
    except Exception:
        return None


def ors_walking_distance(lat1, lon1, lat2, lon2):
    """
    Gehentfernung (in km) zwischen zwei Punkten über OpenRouteService.
    Wir gehen davon aus, dass ein API-Key vorhanden ist, sonst wird die
    Berechnung bereits vorher abgebrochen.
    Bei HTTP-Problemen fallback auf Luftlinie.
    """
    api_key = get_ors_api_key()
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    body = {
        "coordinates": [
            [lon1, lat1],
            [lon2, lat2],
        ],
        "instructions": False
    }

    try:
        resp = requests.post(ORS_BASE_URL, json=body, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        dist_m = data["features"][0]["properties"]["summary"]["distance"]
        return dist_m / 1000.0
    except Exception:
        # Für Einzelfehler: Notfall-Fallback
        return haversine_distance(lat1, lon1, lat2, lon2)


def ors_walking_route_geometry(coords):
    """
    Holt die komplette Fußgängerroute (Liniengeometrie) für eine Liste von
    Koordinaten über ORS.

    coords: Liste von [lon, lat] in Besuchsreihenfolge (Start + Bars).

    Rückgabe:
      Liste von (lat, lon)-Tupeln für die Polyline.
      Fallback: einfache Verbindung der Punkte (Luftlinie), falls ORS scheitert.
    """
    api_key = get_ors_api_key()
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    body = {
        "coordinates": coords,
        "instructions": False
    }

    try:
        resp = requests.post(ORS_BASE_URL, json=body, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        geom = data["features"][0]["geometry"]["coordinates"]  # Liste [lon, lat]
        path = [(lat, lon) for lon, lat in geom]
        return path
    except Exception:
        # Fallback: gerade Linien zwischen Punkten
        return [(lat, lon) for lon, lat in coords]


def geocode_address(address: str):
    """
    Geocodiert eine Adresse via Nominatim (OpenStreetMap) zu (lat, lon).

    Gibt (lat, lon) zurück oder (None, None), wenn nichts gefunden wird.
    """
    if not address or not address.strip():
        return None, None

    if "regensburg" not in address.lower():
        query = f"{address}, Regensburg, Deutschland"
    else:
        query = address

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "pubcrawl-regensburg-app/1.0"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"Fehler beim Geocoding: {e}")
        return None, None

    if not data:
        return None, None

    try:
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
        return lat, lon
    except (KeyError, ValueError):
        return None, None


# ==============================
# Scoring & Routing
# ==============================

def compute_preference_score(row, pref_essen, pref_bier, pref_sport):
    """
    Preference-Score für eine Bar.
    Nutzt:
      - has_food
      - is_beer_spot
      - is_sportsbar
    """
    score = 0.0
    w_food = 2.0
    w_beer = 2.0
    w_sport = 2.0

    def as_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return x != 0
        if isinstance(x, str):
            return x.lower() in ["1", "true", "yes", "ja"]
        return False

    if pref_essen and "has_food" in row and as_bool(row["has_food"]):
        score += w_food
    if pref_bier and "is_beer_spot" in row and as_bool(row["is_beer_spot"]):
        score += w_beer
    if pref_sport and "is_sportsbar" in row and as_bool(row["is_sportsbar"]):
        score += w_sport

    if "distance_km" in row and not pd.isna(row["distance_km"]):
        score -= 0.1 * float(row["distance_km"])

    return score


def build_greedy_route(user_lat, user_lon, bars_df):
    """
    Greedy-Route mit ORS-Gehdistanz:
      - Start bei (user_lat, user_lon)
      - immer zur nächstgelegenen (zu Fuß) noch unbesuchten Bar
      - Startpunkt wandert mit
    """
    remaining = bars_df.copy()
    route_rows = []

    current_lat, current_lon = user_lat, user_lon
    cum = 0.0

    while not remaining.empty:
        remaining["temp_dist"] = remaining.apply(
            lambda r: ors_walking_distance(current_lat, current_lon, r["lat"], r["lon"]),
            axis=1
        )

        next_idx = remaining["temp_dist"].idxmin()
        next_row = remaining.loc[next_idx].copy()

        d = float(remaining.loc[next_idx, "temp_dist"])
        cum += d
        next_row["step_distance_km"] = d
        next_row["cum_distance_km"] = cum

        route_rows.append(next_row)

        current_lat, current_lon = next_row["lat"], next_row["lon"]
        remaining = remaining.drop(next_idx)

    route_df = pd.DataFrame(route_rows).drop(columns=["temp_dist"], errors="ignore")
    total_dist = cum
    return route_df, total_dist


def plot_route_map(user_lat, user_lon, route_df):
    """
    Folium-Karte mit echter Fußgängerroute:
      - Startmarker
      - Barmarker in Reihenfolge
      - Polyline entlang der ORS-Route (nicht Luftlinie)
    """
    if route_df.empty:
        return folium.Map(location=[user_lat, user_lon], zoom_start=15)

    first_bar = route_df.iloc[0]
    center_lat = (user_lat + first_bar["lat"]) / 2
    center_lon = (user_lon + first_bar["lon"]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Startpunkt
    folium.Marker(
        [user_lat, user_lon],
        tooltip="Start (Du)",
        icon=folium.Icon(color="green", icon="home")
    ).add_to(m)

    # Bars
    coords = [[user_lon, user_lat]]  # Start zuerst
    for i, row in enumerate(route_df.itertuples(), start=1):
        folium.Marker(
            [row.lat, row.lon],
            tooltip=f"{i}. {row.name}",
            icon=folium.Icon(color="blue", icon="beer")
        ).add_to(m)
        coords.append([row.lon, row.lat])

    # ORS-Route als Liniengeometrie
    path = ors_walking_route_geometry(coords)
    folium.PolyLine(path, weight=4, opacity=0.7).add_to(m)

    return m


def load_bars_csv(csv_path: str) -> pd.DataFrame:
    """
    Lädt deine Bars aus CSV.
    Erwartet:
      - name
      - lat
      - lon
    """
    df = pd.read_csv(csv_path)

    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError("CSV braucht mindestens Spalten 'lat' und 'lon'.")

    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    for col in ["has_food", "is_beer_spot", "is_sportsbar"]:
        if col not in df.columns:
            df[col] = False

    return df


# ==============================
# Streamlit App
# ==============================

def main():
    st.set_page_config(page_title="Pub Crawl Regensburg", layout="centered")
    st.title("🍻 Pub Crawl Regensburg")

    # --- ORS-Key Pflicht für Fußweg ---
    if get_ors_api_key() is None:
        st.error(
            "Kein OpenRouteService API-Key konfiguriert.\n\n"
            "Bitte setze `ORS_API_KEY` als Umgebungsvariable oder in den Streamlit-Secrets, "
            "damit der Fußweg berechnet und angezeigt werden kann."
        )
        return

    if "route_ready" not in st.session_state:
        st.session_state["route_ready"] = False

    try:
        bars_df = load_bars_csv("bars_regensburg.csv")
    except Exception as e:
        st.error(f"Fehler beim Laden der CSV: {e}")
        return

    # ========== PHASE 1: Eingabe ==========
    if not st.session_state["route_ready"]:
        st.markdown("### Deine Einstellungen")

        st.markdown("**Was ist dir wichtig?**")
        pref_essen = st.checkbox("🍽️ Essen")
        pref_bier = st.checkbox("🍺 Bierpunkt")
        pref_sport = st.checkbox("🏟️ Sport / Sportsbar")

        st.markdown("**Wo startest du?**")
        address = st.text_input(
            "Adresse in Regensburg",
            placeholder="z.B. Domplatz 1 oder Bahnhof Regensburg",
        )

        st.markdown("**Wie viele Bars möchtest du besuchen?**")
        k = st.slider("Anzahl Bars", min_value=3, max_value=20, value=8)

        if st.button("Route berechnen 🚀", use_container_width=True):
            user_lat, user_lon = geocode_address(address)
            if user_lat is None or user_lon is None:
                st.error("Konnte deinen Standort nicht finden. Probier eine genauere Adresse.")
                return

            if bars_df.empty:
                st.error("Der Bar-DataFrame ist leer.")
                return
            if not {"name", "lat", "lon"}.issubset(bars_df.columns):
                st.error("DataFrame braucht mindestens die Spalten: 'name', 'lat', 'lon'.")
                return

            k_effective = min(k, len(bars_df))

            # Schritt 1: Gehentfernung Start → alle Bars
            bars = bars_df.copy()
            bars["distance_km"] = bars.apply(
                lambda r: ors_walking_distance(user_lat, user_lon, r["lat"], r["lon"]),
                axis=1
            )

            # Schritt 2: k-nächste Bars
            bars_nearest = bars.sort_values("distance_km").head(k_effective).copy()

            # Schritt 3: Score
            bars_nearest["score"] = bars_nearest.apply(
                lambda row: compute_preference_score(row, pref_essen, pref_bier, pref_sport),
                axis=1
            )

            # Schritt 4: Ranking
            bars_ranked = bars_nearest.sort_values(
                by=["score", "distance_km"],
                ascending=[False, True]
            ).copy()

            # Schritt 5: Route mit ORS-Distanzen
            route_df, total_dist = build_greedy_route(user_lat, user_lon, bars_ranked)

            st.session_state["route_ready"] = True
            st.session_state["user_lat"] = user_lat
            st.session_state["user_lon"] = user_lon
            st.session_state["route_df"] = route_df
            st.session_state["total_dist"] = total_dist

            st.rerun()

    # ========== PHASE 2: Ergebnis ==========
    if st.session_state["route_ready"]:
        user_lat = st.session_state["user_lat"]
        user_lon = st.session_state["user_lon"]
        route_df = st.session_state["route_df"]
        total_dist = st.session_state["total_dist"]

        st.markdown("### Deine Pub-Crawl-Route 🍺")

        st.write(f"Gesamtdistanz (zu Fuß, ca.): **{total_dist:.2f} km**")

        st.markdown("**Reihenfolge der Bars:**")
        route_view = route_df.reset_index(drop=True)[["name", "step_distance_km", "cum_distance_km"]]
        route_view.index = route_view.index + 1
        route_view.rename(
            columns={
                "name": "Bar",
                "step_distance_km": "Fußweg zur vorherigen (km)",
                "cum_distance_km": "Gesamt-Fußweg (km)",
            },
            inplace=True,
        )
        st.dataframe(route_view, use_container_width=True)

        st.markdown("**Karte (Fußweg über Straßen):**")
        route_map = plot_route_map(user_lat, user_lon, route_df)
        st_folium(
            route_map,
            width=900,
            height=600,
            key="route_map"
        )

        st.markdown("---")
        if st.button("🔁 Neue Route planen", use_container_width=True):
            st.session_state["route_ready"] = False
            st.rerun()


if __name__ == "__main__":
    main()
