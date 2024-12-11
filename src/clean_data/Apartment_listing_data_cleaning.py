import pandas as pd
import numpy as np

# Load the CSV file
file_path = "data/raw/apartments_listings_with_details.csv"
# Replace with your actual file path
df = pd.read_csv(file_path)

# Step 1: Standardize Price Column
# Handle non-numeric values like "Call for Rent" by replacing them with NaN
df['Price'] = df['Price'].replace("Call for Rent", np.nan)

# Split Price range and create min_price and max_price columns
# Remove $, commas, and replace non-numeric ranges with NaN
df[['min_price', 'max_price']] = (
    df['Price']
    .str.replace(r'[\$,]', '', regex=True)
    .str.split('-', expand=True)
)
df['min_price'] = pd.to_numeric(df['min_price'], errors='coerce')
df['max_price'] = pd.to_numeric(df['max_price'], errors='coerce')
df = df.dropna(subset=['min_price'])
# Remove rows with min_price > 7000
df = df[df['min_price'] <= 7000]


# Step 2: Extract ZIP Codes from Address
def extract_zip(address):
    address_parts = str(address).split()
    zip_code = address_parts[-1]
    if zip_code:
        return zip_code
    return np.nan


df['postal_code'] = df['Address'].apply(extract_zip)

# Step 3: Merge with City Names based on ZIP code
ca_cities_file_path = "data/processed/filtered_CA_cities_zip_code_database.csv"
ca_cities = pd.read_csv(ca_cities_file_path)
# Ensure postal_code columns are numeric for merging
df['postal_code'] = pd.to_numeric(df['postal_code'], errors='coerce')
ca_cities['zip'] = pd.to_numeric(ca_cities['zip'], errors='coerce')
# Merge on postal_code (zip code)
df = df.merge(
    ca_cities[['zip', 'primary_city']],
    left_on='postal_code',
    right_on='zip',
    how='left'
)
# Rename column and drop rows without city names
df = df.rename(columns={'primary_city': 'city_name'}).drop(columns=['zip'])
df = df.dropna(subset=['city_name'])

# Step 4: Handle Missing and Inconsistent Data in 'Property Rating'
# Replace "No rating found" and other non-numeric entries with 0,
# and round numeric ratings to 1 decimal place
df['Property Rating'] = pd.to_numeric(df['Property Rating'], errors='coerce')\
    .fillna(0).astype(float).round(1)


# Step 5: Clean Amenities Column
# Define a function to check for amenities
all_amenities = df['Amenities'].dropna().str.split(',').sum()
unique_amenities = pd.Series(all_amenities).str.strip().unique()

security_keywords = [
    "security", "gated", "surveillance", "camera", "alarm",
    "guard", "controlled access", "key fob entry", "night patrol",
    "24-hour surveillance", "access control", "security system",
    "electronic locks", "monitoring", "fenced", "safe",
    "locked", "protected", "secure entry", "patrol"
]

# Create a dictionary to hold security-related amenities by keyword
security_related = {keyword: [] for keyword in security_keywords}

# Categorize amenities based on the first matching keyword
for amenity in unique_amenities:
    for keyword in security_keywords:
        if keyword in amenity.lower():
            security_related[keyword].append(amenity)
            break

# Step 6: Add keyword columns to DataFrame and count matches
for keyword in security_keywords:
    df[keyword] = df['Amenities'].apply(
        lambda row: sum(
            amenity in str(row) for amenity in security_related[keyword]
        )
    )

# Calculate the total number of security-related amenities for each row
# df['total_security_amenities'] = df[security_related_amenities].sum(axis=1)
df['total_security_amenities'] = df[security_keywords].sum(axis=1)


# Step 6: Categorize amenity
for keyword in security_keywords:
    df[keyword] = df['Amenities'].apply(
        lambda row: sum(
            amenity in str(row) for amenity in security_related[keyword]
        )
    )

# Step 7: URL Validation
df['Property Link'] = df['Property Link']\
    .apply(lambda x: x if pd.notnull(x) and x.startswith("http") else np.nan)

# Step 8: Deduplicate Entries
df = df.drop_duplicates(subset=['Property Link'])

# Step 9: Handle 'Review Count' Column
# Replace "No reviews" with 0 and convert to integer
df['Review Count'] = df['Review Count']\
    .replace("No reviews", 0).fillna(0).astype(int)

# Save the cleaned DataFrame to a new CSV file
output_file_path = "data/processed/cleaned_rental_data_with_postalcode.csv"
df.to_csv(output_file_path, index=False)

print(f"Cleaned data saved as {output_file_path}")
