from selenium import webdriver
from bs4 import BeautifulSoup
import time
import csv
import unicodedata
import re
import random


def format_city_name(city_name):
    # Remove accents and special characters
    city_normalized = unicodedata.normalize('NFD', city_name) \
        .encode('ascii', 'ignore').decode('utf-8')
    # Replace '&' with 'and'
    city_normalized = city_normalized.replace('&', 'and')
    # Replace spaces and slashes with hyphens
    city_formatted = re.sub(r'[ /]+', '-', city_normalized.lower())
    # Remove any remaining non-alphanumeric characters except hyphens
    city_formatted = re.sub(r'[^a-z0-9\-]', '', city_formatted)
    return city_formatted


def fetch_apartment_listings_with_selenium(cities):
    driver = webdriver.Chrome()
    listings_data = []
    state = 'ca'
    total_listings = 0

    for city in cities:
        city_formatted = format_city_name(city)
        print(f"Scraping data for {city_formatted}, {state}...")
        base_url = f"https://www.apartments.com/{city_formatted}-{state}/"

        # Step 1: Determine the total number of pages with explicit waits
        driver.get(base_url)
        time.sleep(5)  # Additional wait for the pagination to load

        # Parse the page to find the maximum number of pages
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # pagination = soup.find('nav', id='paging')
        page_string = soup.find('span', class_='pageRange')
        # page_list = page_string.text.split(" ")
        # max_page = int(page_list[-1])
        if page_string:
            # page_links = pagination.find_all('a', {'data-page': True})
            # max_page = int(page_links[-1]['data-page'])
            page_list = page_string.text.split(" ")
            max_page = int(page_list[-1])
        else:
            max_page = 1  # If no pagination is found, assume one page

        print(f"Total pages for {city}: {max_page}")
        city_listings_count = 0

        for page in range(1, max_page + 1):
            url = base_url if page == 1 else f"{base_url}{page}/"
            driver.get(url)
            time.sleep(random.uniform(5, 8))

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            property_cards = soup.find_all('li', class_='mortar-wrapper')

            if not property_cards:
                print(f"No property cards found for {city} on page {page}.")
                break

            for card in property_cards:
                try:
                    name_tag = card.find('span', class_='js-placardTitle')
                    name = name_tag.text.strip() if name_tag else 'N/A'
                    if name == 'N/A':
                        title_tag = card.find('div', class_='property-title')
                        name = title_tag['title'] if title_tag \
                            and 'title' in title_tag.attrs else 'N/A'

                    address_tag = card.find('div', class_='property-address')
                    address = address_tag.text.strip() \
                        if address_tag else 'N/A'
                    if address == 'N/A':
                        title_tag = card.find('div', class_='property-title')
                        address = title_tag['title'] if title_tag \
                            and 'title' in title_tag.attrs else 'N/A'
                    if address == 'N/A':
                        property_address_tag = \
                            card.find('p', class_='property-address js-url')
                        address = property_address_tag.text.strip() \
                            if property_address_tag else 'N/A'

                    price_tag = card.find('p', class_='property-pricing')
                    price = price_tag.text.strip() if price_tag else 'N/A'
                    if price == 'N/A':
                        price_range_tag = \
                            card.find('div', class_='price-range')
                        price = price_range_tag.text.strip() \
                            if price_range_tag else 'N/A'
                    if price == 'N/A':
                        property_rents_tag = \
                            card.find('p', class_='property-rents')
                        price = property_rents_tag.text.strip() \
                            if property_rents_tag else 'N/A'

                    link_tag = card.find('a', class_='property-link')
                    if link_tag and 'href' in link_tag.attrs:
                        listing_url = link_tag['href']
                    else:
                        listing_url = 'N/A'

                    listings_data.append({
                        'City': city,
                        'Name': name,
                        'Address': address,
                        'Price': price,
                        'Property Link': listing_url
                    })
                    city_listings_count += 1
                    total_listings += 1
                except Exception as e:
                    print(f"Error parsing property card: {e}")
                    continue

            print(
                f"Collected {city_listings_count}"
                f"listings so far for {city}."
                )

        print(f"Total listings collected for {city}: {city_listings_count}")
        time.sleep(random.uniform(10, 15))

    print(f"\nTotal listings collected: {total_listings}")
    driver.quit()

    if listings_data:
        csv_headers = ['City', 'Name', 'Address', 'Price', 'Property Link']

        with open(
            'apartments_listings.csv', 'w', newline='', encoding='utf-8'
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(listings_data)

        print("Data has been saved to 'apartments_listings.csv'.")
    else:
        print("No data collected to save.")


def get_detailed_info_from_url(link):
    # driver = webdriver.Chrome()
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("--window-size=1920,1080")
    # chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
    # chrome_options.add_argument("--disable-http2")
    driver = webdriver.Chrome(options=chrome_options)
    amenities = []
    property_rating = None
    review_count = None
    try:
        driver.get(link)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # print(soup.text)

        amenities_card = soup.find_all('div', class_='amenityCard')
        for card in amenities_card:
            amenities_labels = card.find('p', class_='amenityLabel')
            if amenities_labels:
                amenities.append(amenities_labels.text.strip())

        spec_info_items = soup.find_all('li', class_='specInfo')
        for item in spec_info_items:
            amenity_text = item.find('span')
            if amenity_text:
                amenities.append(amenity_text.text.strip())

        rating_tag = soup.find('div', class_='averageRating')
        if rating_tag:
            property_rating = rating_tag.text.strip()

        review_count_tag = soup.find('p', class_='renterReviewsLabel')
        if review_count_tag:
            review_count_tag_list = review_count_tag.text.split(' ')
            review_count = review_count_tag_list[0]

    except Exception as e:
        print(f"Error fetching amenities from {link}: {e}")

    driver.quit()
    return {
        'Amenities': amenities,
        'Property Rating': property_rating,
        'Review Count': review_count
        }
    # print (amenities)


def append_data_to_csv(input_csv, output_csv):
    rows = []
    row_counter = 0
    with open(input_csv, 'r', newline='', encoding='utf-8') as csv_file:
        data = csv.DictReader(csv_file)
        for row in data:
            row_counter += 1
            link = row['Property Link']
            print(f"Fetching amenities for the {row_counter} row...")

            details = get_detailed_info_from_url(link)
            row['Amenities'] = ', '.join(details['Amenities'])\
                if details['Amenities'] else 'No amenities found'
            row['Property Rating'] = details['Property Rating']\
                if details['Property Rating'] else 'No rating found'
            row['Review Count'] = details['Review Count']\
                if details['Review Count'] else 'No reviews'
            rows.append(row)
            print(f"Amenities for {link}")
            time.sleep(2)

    fieldnames = data.fieldnames +\
        ['Amenities', 'Property Rating', 'Review Count']
    # Add new columns to the header
    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Data with amenities has been saved to '{output_csv}'.")


if __name__ == "__main__":
    cities = [
        "Agoura Hills",
        "Alhambra",
        "Arcadia",
        "Artesia",
        "Avalon",
        "Azusa",
        "Baldwin Park",
        "Bell",
        "Bell Gardens",
        "Bellflower",
        "Beverly Hills",
        "Bradbury",
        "Burbank",
        "Calabasas",
        "Carson",
        "Cerritos",
        "Claremont",
        "Commerce",
        "Compton",
        "Covina",
        "Cudahy",
        "Culver City",
        "Diamond Bar",
        "Downey",
        "Duarte",
        "El Monte",
        "El Segundo",
        "Gardena",
        "Glendale",
        "Glendora",
        "Hawaiian Gardens",
        "Hawthorne",
        "Hermosa Beach",
        "Hidden Hills",
        "Huntington Park",
        "Industry",
        "Inglewood",
        "Irwindale",
        "La Cañada Flintridge",
        "La Habra Heights",
        "La Mirada",
        "La Puente",
        "La Verne",
        "Lakewood",
        "Lancaster",
        "Lawndale",
        "Lomita",
        "Long Beach",
        "Los Angeles",
        "Lynwood",
        "Malibu",
        "Manhattan Beach",
        "Maywood",
        "Monrovia",
        "Montebello",
        "Monterey Park",
        "Norwalk",
        "Palmdale",
        "Palos Verdes Estates",
        "Paramount",
        "Pasadena",
        "Pico Rivera",
        "Pomona",
        "Rancho Palos Verdes",
        "Redondo Beach",
        "Rolling Hills",
        "Rolling Hills Estates",
        "Rosemead",
        "San Dimas",
        "San Fernando",
        "San Gabriel",
        "San Marino",
        "Santa Clarita",
        "Santa Fe Springs",
        "Santa Monica",
        "Sierra Madre",
        "Signal Hill",
        "South El Monte",
        "South Gate",
        "South Pasadena",
        "Temple City",
        "Torrance",
        "Vernon",
        "Walnut",
        "West Covina",
        "West Hollywood",
        "Westlake Village",
        "Whittier"
    ]

    input_csv = fetch_apartment_listings_with_selenium(cities)
    output_csv = 'data/raw/apartments_listings_with_details.csv'
    append_data_to_csv(input_csv, output_csv)
