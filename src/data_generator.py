"""
data_generator.py
Generates a realistic synthetic housing dataset for the NCR region.
"""

import numpy as np
import pandas as pd
import os

def generate_housing_data(n_samples=1200, random_state=42):
    """Generate a realistic synthetic housing dataset."""
    np.random.seed(random_state)

    # ── Location settings ────────────────────────────────────────────────────
    locations = ["Delhi", "Gurgaon", "Noida", "Greater Noida",
                 "Ghaziabad", "Faridabad"]
    location_multiplier = {
        "Delhi": 1.60,
        "Gurgaon": 1.45,
        "Noida": 1.20,
        "Greater Noida": 1.00,
        "Ghaziabad": 0.90,
        "Faridabad": 0.85,
    }

    furnishing_options = ["Furnished", "Semi-Furnished", "Unfurnished"]
    furnishing_mult   = {"Furnished": 1.20, "Semi-Furnished": 1.08, "Unfurnished": 1.00}

    property_types     = ["Apartment", "Builder Floor", "Independent House", "Villa"]
    property_type_mult = {
        "Apartment": 1.00,
        "Builder Floor": 1.10,
        "Independent House": 1.25,
        "Villa": 1.55,
    }

    # ── Raw feature generation ────────────────────────────────────────────────
    location_arr    = np.random.choice(locations, n_samples)
    property_type_arr = np.random.choice(
        property_types, n_samples, p=[0.50, 0.20, 0.20, 0.10]
    )
    furnishing_arr  = np.random.choice(
        furnishing_options, n_samples, p=[0.30, 0.40, 0.30]
    )

    bedrooms_arr  = np.random.choice([1, 2, 3, 4, 5], n_samples,
                                     p=[0.10, 0.30, 0.35, 0.18, 0.07])
    bathrooms_arr = np.clip(
        bedrooms_arr - 1 + np.random.randint(0, 2, n_samples), 1, 5
    )
    balconies_arr = np.random.choice([0, 1, 2, 3], n_samples,
                                     p=[0.10, 0.45, 0.35, 0.10])
    floors_arr    = np.random.randint(1, 25, n_samples)
    parking_arr   = np.random.choice([0, 1, 2], n_samples, p=[0.20, 0.60, 0.20])

    base_area = bedrooms_arr * 400 + np.random.randint(200, 800, n_samples)
    area_sqft_arr = np.clip(base_area, 500, 5000).astype(float)

    property_age_arr           = np.random.randint(0, 30, n_samples)
    distance_from_center_arr   = np.round(
        np.random.uniform(1, 35, n_samples), 1
    )
    nearby_schools_arr   = np.random.randint(0, 10, n_samples)
    nearby_hospitals_arr = np.random.randint(0, 8, n_samples)
    crime_rate_arr       = np.round(np.random.uniform(1, 10, n_samples), 1)

    # ── Price formula ─────────────────────────────────────────────────────────
    base_price_per_sqft = 4000  # INR / sq ft baseline

    prices = []
    for i in range(n_samples):
        loc  = location_arr[i]
        ptype = property_type_arr[i]
        furn  = furnishing_arr[i]
        area  = area_sqft_arr[i]
        dist  = distance_from_center_arr[i]
        age   = property_age_arr[i]
        schools   = nearby_schools_arr[i]
        hospitals = nearby_hospitals_arr[i]
        crime     = crime_rate_arr[i]

        per_sqft = (
            base_price_per_sqft
            * location_multiplier[loc]
            * property_type_mult[ptype]
            * furnishing_mult[furn]
        )
        per_sqft *= max(0.75, 1 - 0.015 * dist)       # distance penalty
        per_sqft *= max(0.80, 1 - 0.008 * age)        # age depreciation
        per_sqft *= 1 + 0.02 * min(schools, 5)        # nearby schools boost
        per_sqft *= 1 + 0.015 * min(hospitals, 4)     # hospital boost
        per_sqft *= max(0.85, 1 - 0.012 * crime)      # crime penalty

        price = per_sqft * area
        # Add realistic noise ±8 %
        noise = np.random.uniform(0.92, 1.08)
        prices.append(round(price * noise, -3))        # round to nearest 1000

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "location":                location_arr,
        "area_sqft":               area_sqft_arr.astype(int),
        "bedrooms":                bedrooms_arr,
        "bathrooms":               bathrooms_arr,
        "balconies":               balconies_arr,
        "floors":                  floors_arr,
        "parking":                 parking_arr,
        "property_age":            property_age_arr,
        "furnishing_status":       furnishing_arr,
        "property_type":           property_type_arr,
        "distance_from_city_center": distance_from_center_arr,
        "nearby_schools":          nearby_schools_arr,
        "nearby_hospitals":        nearby_hospitals_arr,
        "crime_rate":              crime_rate_arr,
        "price":                   prices,
    })

    return df


def save_dataset(filepath="data/housing_data.csv"):
    """Generate and save the dataset to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = generate_housing_data()
    df.to_csv(filepath, index=False)
    print(f"Dataset saved → {filepath}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    save_dataset()
