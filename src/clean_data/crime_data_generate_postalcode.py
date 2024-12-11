import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm  # Import tqdm for progress tracking
import warnings
import time
import random

warnings.filterwarnings('ignore')

# Load the CSV file, the file is in the google cloud, please download it via
# the link provided in README.md and place it in the correct relative path.
df = pd.read_csv('data/raw/[Org]Crime_Data_from_2020_to_Present_20241109.csv')
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])

# Define date range for half a year
start_date = '2024-05-09'
end_date = '2024-11-09'
df = df[(df['DATE OCC'] >= start_date) & (df['DATE OCC'] <= end_date)]

# Remove rows with missing or zero latitude/longitude values
df = df.dropna(subset=['LAT', 'LON'])
df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

# Remove duplicates
df = df.drop_duplicates(subset=['DR_NO'], keep='first')


# Function to get a geolocator with a random user_agent
def get_geolocator():
    user_agent = f'user_{random.randint(10000, 99999)}'
    # print(user_agent)
    return Nominatim(user_agent=user_agent)


def get_postal_code(lat, lon, sleep_sec=2):
    geolocator = get_geolocator()
    # Get a new geolocator instance with a random user_agent
    geocode = RateLimiter(
        geolocator.reverse,
        min_delay_seconds=1,
        error_wait_seconds=10
    )
    try:
        location = geocode((lat, lon), exactly_one=True)
        if location and "postcode" in location.raw["address"]:
            return location.raw["address"]["postcode"]
        return None
    except Exception as e:
        print(f"Error retrieving postal code for lat={lat}, lon={lon}: {e}")
        return None
    finally:
        # Sleep for a random interval between requests
        time.sleep(random.randint(1 * 100, sleep_sec * 100) / 100)


# List to store postal codes
postal_codes = []

# Loop through each row in the DataFrame with tqdm for progress tracking
for index, row in tqdm(
    df.iterrows(),
    total=df.shape[0],
    desc="Geocoding Progress",
    unit="record"
):
    lat = row['LAT']
    lon = row['LON']
    postal_code = get_postal_code(lat, lon)
    postal_codes.append(postal_code)

# Add postal codes to the DataFrame
df["postal_code"] = postal_codes

# Drop rows where postal code is missing
df.dropna(subset=['postal_code'], inplace=True)

# Save the cleaned DataFrame to a new CSV file
output_file_path = \
    'data/processed/Crime_data_with_postalcodes_240509_241109.csv'
df.to_csv(output_file_path, index=False)

print(f"New CSV file with postal codes saved to {output_file_path}")
