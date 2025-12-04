import os
import csv
from functools import lru_cache
from typing import Optional, Dict

# Path to CSV file with model metadata
DATA_FILE = os.path.join(os.path.dirname(__file__), "../..", "data", "equipment_models_named.csv")
DATA_FILE = os.path.abspath(DATA_FILE)


@lru_cache(None)
def load_model_metadata() -> Dict[str, Dict[str, str]]:
    """
    Load model metadata from CSV file.

    Expected CSV format:
        id,model_code,equipment_type_id,manufacturer_id,is_active,created_at,updated_at
        uuid,lxp_lb_eu_10k,inverter,LuxPower,1,timestamp,timestamp

    Returns:
        Dict mapping model_code to metadata dict
        Example: {
            "lxp_lb_eu_10k": {"manufacturer": "luxpower", "equipment_type": "inverter"},
            "us5000": {"manufacturer": "pylontech", "equipment_type": "battery"}
        }
    """
    metadata = {}

    if not os.path.exists(DATA_FILE):
        print(f"⚠️  Warning: Model metadata file not found: {DATA_FILE}")
        return metadata

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                model_code = row.get("model_code", "").strip().lower()
                manufacturer = row.get("manufacturer_id", "").strip().lower()
                equipment_type = row.get("equipment_type_id", "").strip().lower()
                is_active = row.get("is_active", "1").strip()

                # Only include active models
                if model_code and is_active == "1":
                    metadata[model_code] = {
                        "manufacturer": manufacturer,
                        "equipment_type": equipment_type
                    }

        print(f"✅ Loaded metadata for {len(metadata)} models")

    except Exception as e:
        print(f"❌ Error loading model metadata: {e}")

    return metadata


def get_model_metadata(canonical_model: str) -> Optional[Dict[str, str]]:
    """
    Get metadata (manufacturer, equipment_type) for a canonical model.

    Args:
        canonical_model: Canonical model name (e.g., "lxp_lb_eu_10k")

    Returns:
        Dict with "manufacturer" and "equipment_type", or None if not found
        Example: {"manufacturer": "luxpower", "equipment_type": "inverter"}
    """
    if not canonical_model:
        return None

    metadata = load_model_metadata()
    key = canonical_model.strip().lower()

    return metadata.get(key)


def get_manufacturer(canonical_model: str) -> Optional[str]:
    """
    Get manufacturer for a canonical model.

    Args:
        canonical_model: Canonical model name

    Returns:
        Manufacturer name or None
    """
    meta = get_model_metadata(canonical_model)
    return meta.get("manufacturer") if meta else None


def get_equipment_type(canonical_model: str) -> Optional[str]:
    """
    Get equipment type for a canonical model.

    Args:
        canonical_model: Canonical model name

    Returns:
        Equipment type or None
    """
    meta = get_model_metadata(canonical_model)
    return meta.get("equipment_type") if meta else None


# For debugging
if __name__ == "__main__":
    print("=" * 60)
    print("MODEL METADATA LOADER")
    print("=" * 60)

    print(f"\nLoading from: {DATA_FILE}\n")

    metadata = load_model_metadata()

    if not metadata:
        print("❌ No metadata loaded!")
        print(f"\n💡 CSV file should be at: {DATA_FILE}")
        print("   Format: id,model_code,equipment_type_id,manufacturer_id,is_active,created_at,updated_at")
        print("   Example:")
        print("   uuid,lxp_lb_eu_10k,inverter,LuxPower,1,timestamp,timestamp")
        print("   uuid,us5000,battery,Pylontech,1,timestamp,timestamp")
    else:
        print(f"📋 First 10 models:")
        for idx, (model, meta) in enumerate(list(metadata.items())[:10], 1):
            print(f"   {idx}. {model}")
            print(f"      Manufacturer: {meta['manufacturer']}")
            print(f"      Equipment Type: {meta['equipment_type']}")

        print("\n🧪 Testing lookups:")
        test_models = [
            "4600tlm_g2",
            "multiplus_ii_48_10000_140_100",
            "b4850",
            "unknown_model"
        ]

        for model in test_models:
            meta = get_model_metadata(model)
            if meta:
                print(f"   ✅ {model}: {meta['manufacturer']} / {meta['equipment_type']}")
            else:
                print(f"   ❌ {model}: Not found")