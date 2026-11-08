import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.elevator_logistics import calculate_dynamic_silo_loads

def sync():
    df = calculate_dynamic_silo_loads()
    out_path = PROJECT_ROOT / "data" / "processed" / "silos.parquet"
    df.to_parquet(out_path)
    print(f"Successfully saved {len(df)} elevators to {out_path}")

if __name__ == "__main__":
    sync()

