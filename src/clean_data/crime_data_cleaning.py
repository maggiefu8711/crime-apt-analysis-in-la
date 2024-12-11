import pandas as pd

crime_file_path = \
    "data/processed/Crime_data_with_postalcodes_240509_241109.csv"
df = pd.read_csv(crime_file_path)

# Step 1: Remove Duplicates
df = df.drop_duplicates(subset=['DR_NO'], keep='first')

# Step 2: Load ZIP Code-to-City Mapping File and Merge with City Names
ca_cities_file_path = "data/processed/filtered_CA_cities_zip_code_database.csv"

ca_cities = pd.read_csv(ca_cities_file_path)

df['postal_code'] = pd.to_numeric(df['postal_code'], errors='coerce')
ca_cities['zip'] = pd.to_numeric(ca_cities['zip'], errors='coerce')

df = df.merge(
    ca_cities[['zip', 'primary_city']],
    left_on='postal_code',
    right_on='zip',
    how='left'
)
# Rename column and drop rows without city names
df = df.rename(columns={'primary_city': 'city_name'}).drop(columns=['zip'])
df = df.dropna(subset=['city_name'])

# Step 3: Clean Crime Type Column
# Define a mapping dictionary for crime type simplification and categorization
crime_type_mapping = {
    # Theft-related crimes (including vehicle theft)
    'theft from motor vehicle - petty ($950 & under)': 'theft',
    'theft plain - petty ($950 & under)': 'theft',
    'shoplifting - petty theft ($950 & under)': 'theft',
    'theft from motor vehicle - grand ($950.01 and over)': 'theft',
    'bike - stolen': 'theft',
    'theft-grand ($950.01 & over)excpt,guns,fowl,livestk,prod': 'theft',
    'theft of identity': 'theft',
    'robbery': 'theft',
    'theft, person': 'theft',
    'pickpocket': 'theft',
    'defrauding innkeeper/theft of services, $950 & under': 'theft',
    'embezzlement, grand theft ($950.01 & over)': 'theft',
    'embezzlement, petty theft ($950 & under)': 'theft',
    'shoplifting-grand theft ($950.01 & over)': 'theft',
    'bunco, petty theft': 'theft',
    'credit cards, fraud use ($950 & under': 'theft',
    'theft plain - attempt': 'theft',
    'drunk roll': 'theft',
    'bunco, attempt': 'theft',
    'grand theft / insurance fraud': 'theft',
    'theft, coin machine - petty ($950 & under)': 'theft',
    'theft, coin machine - grand ($950.01 & over)': 'theft',
    'theft from motor vehicle - attempt': 'theft',
    'shoplifting - attempt': 'theft',
    'drunk roll - attempt': 'theft',
    'bunco, grand theft': 'theft',
    'burglary': 'theft',
    'burglary from vehicle': 'theft',
    'burglary from vehicle, attempted': 'theft',
    'attempted robbery': 'theft',
    'burglary, attempted': 'theft',
    'vehicle - stolen': 'theft',
    'vehicle, stolen - other (motorized scooters, bikes, etc)': 'theft',
    'vehicle - attempt stolen': 'theft',
    'driving without owner consent (dwoc)': 'theft',
    'boat - stolen': 'theft',

    # Assault-related crimes
    'battery - simple assault': 'assault',
    'intimate partner - simple assault': 'assault',
    'assault with deadly weapon, aggravated assault': 'assault',
    'battery police (simple)': 'assault',
    'other assault': 'assault',
    'intimate partner - aggravated assault': 'assault',
    'child abuse (physical) - aggravated assault': 'assault',
    'child abuse (physical) - simple assault': 'assault',
    'battery on a firefighter': 'assault',
    'assault with deadly weapon on police officer': 'assault',
    'brandish weapon': 'assault',
    'criminal threats - no weapon displayed': 'assault',

    # Sexual Offenses
    'lewd conduct': 'sexual_offense',
    'indecent exposure': 'sexual_offense',
    'oral copulation': 'sexual_offense',
    'sexual penetration w/foreign object': 'sexual_offense',
    'sodomy/sexual contact b/w penis of one pers to anus oth': (
        'sexual_offense'
    ),
    'rape, forcible': 'sexual_offense',
    'rape, attempted': 'sexual_offense',
    'battery with sexual contact': 'sexual_offense',
    'child pornography': 'sexual_offense',
    'child annoying (17yrs & under)': 'sexual_offense',

    # Public Order Crimes
    'violation of court order': 'public_order',
    'violation of temporary restraining order': 'public_order',
    'stalking': 'public_order',
    'child stealing': 'public_order',
    'child neglect (see 300 w.i.c.)': 'public_order',
    'crm agnst chld (13 or under) (14-15 & susp 10 yrs older)': 'public_order',
    'trespassing': 'public_order',
    'failure to yield': 'public_order',
    'reckless driving': 'public_order',
    'disturbing the peace': 'public_order',
    'violation of restraining order': 'public_order',
    'peeping tom': 'public_order',
    'resisting arrest': 'public_order',
    'sex offender registrant out of compliance': 'public_order',
    'bomb scare': 'public_order',
    'vandalism - felony ($400 & over, all church vandalisms)': 'public_order',
    'vandalism - misdeameanor ($399 or under)': 'public_order',

    # Fraud and Financial Crimes
    'document forgery / stolen felony': 'fraud',
    'extortion': 'fraud',

    # Violent Crimes
    'criminal homicide': 'violent_crime',
    'arson': 'violent_crime',
    'discharge firearms/shots fired': 'violent_crime',
    'shots fired at inhabited dwelling': 'violent_crime',
    'shots fired at moving vehicle, train or aircraft': 'violent_crime',
    'kidnapping': 'violent_crime',
    'false imprisonment': 'violent_crime',

    # Human Trafficking
    'human trafficking - commercial sex acts': 'human_trafficking',
    'human trafficking - involuntary servitude': 'human_trafficking',
    'pimping': 'human_trafficking',
    'pandering': 'human_trafficking',

    # Animal Cruelty
    'cruelty to animals': 'animal_cruelty',

    # Miscellaneous or Other Crimes
    'other miscellaneous crime': 'other'
}
# Convert all entries to lowercase and strip whitespace
df['Crm Cd Desc'] = df['Crm Cd Desc'].str.lower().str.strip()
df['crime_category'] = df['Crm Cd Desc']\
    .map(crime_type_mapping).fillna('other')

df.to_csv('data/processed/crime_with_postalcode_cityname.csv', index=False)
print("New CSV file done!!!")
