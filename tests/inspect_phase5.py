import json
import pickle
import pandas as pd
from pathlib import Path

data_dir = Path("d:/legion/Projects/EcoEconomicAnalyzer/data/processed")

# 1. Silos
silos_path = data_dir / "silos.parquet"
if silos_path.exists():
    df_silos = pd.read_parquet(silos_path)
    print("=== SILOS ===")
    print("Count:", len(df_silos))
    print("Columns:", df_silos.columns.tolist())
    print("Regions:", df_silos["region"].value_counts().to_dict() if "region" in df_silos.columns else "No region col")
    print(df_silos.head(3))
else:
    print("silos.parquet NOT found")

# 2. Logistics routes
routes_path = data_dir / "logistics_routes.json"
if routes_path.exists():
    with open(routes_path, "r", encoding="utf-8") as f:
        routes = json.load(f)
    print("\n=== LOGISTICS ROUTES ===")
    print("Type:", type(routes))
    if isinstance(routes, dict):
        print("Keys count:", len(routes.keys()))
        first_key = next(iter(routes.keys()))
        print(f"Sample key '{first_key}':", routes[first_key])
    elif isinstance(routes, list):
        print("Length:", len(routes))
        print("Sample:", routes[0] if routes else "Empty")
else:
    print("logistics_routes.json NOT found")

# 3. Agri Graph
graph_path = data_dir / "agri_graph.gpickle"
if graph_path.exists():
    print("\n=== AGRI GRAPH ===")
    try:
        with open(graph_path, "rb") as f:
            graph_data = pickle.load(f)
        print("Loaded graph:", type(graph_data))
        if hasattr(graph_data, "nodes"):
            print("Nodes count:", len(graph_data.nodes))
            print("Edges count:", len(graph_data.edges))
    except Exception as e:
        print("Error reading graph:", e)

# 4. Agri Contagion
contagion_path = data_dir / "agricontagion.parquet"
if contagion_path.exists():
    df_contagion = pd.read_parquet(contagion_path)
    print("\n=== AGRI CONTAGION ===")
    print("Count:", len(df_contagion))
    print("Columns:", df_contagion.columns.tolist())
    print(df_contagion.head(3))

