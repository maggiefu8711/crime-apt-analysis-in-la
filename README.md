[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/q4BQ8R99)
# DSCI 510 Final Project

## Impact of Crime and Property Quality on Rental Prices and Tenant Satisfaction in Los Angeles

## Team Members: Fan-Chen Fu

## Instructions to Create a Conda Environment

### Setting up the Conda Environment for the Project

Follow these instructions to create and set up a Conda environment for this project.

---

### Prerequisites
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).
- Ensure that `conda` is added to your system PATH (this is typically done during installation).

---

### Steps to Create and Activate the Environment

#### 1. Create a Conda Environment
Run the following command in your terminal or command prompt to create a new Conda environment named `dsci510_project`:
```bash
conda create --name dsci510_project python=3.12
```
#### 2. Activate the Environment
Once the environment is created, activate it using:
```bash
conda activate dsci510_project
```
- You should now see (dsci510_project) at the beginning of your command prompt, indicating that the environment is active.


### Instructions to Install the required libraries

#### 1. Install Dependencies
Install all required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
#### 2. Verify the Installation
Ensure all packages are installed correctly by running:
```bash
pip list
```
## Instructions on how to download some of the data
### Apartment.com data crawling
#### Prerequisites

1. **Python Environment**
   - Ensure you have Python 3.12 or higher installed.
   - Install the required libraries by running:
     ```bash
     pip install -r requirements.txt
     ```

2. **Selenium WebDriver**
   - Install the Chrome browser.
   - Download the ChromeDriver version matching your Chrome browser from [ChromeDriver](https://chromedriver.chromium.org/downloads).
   - Place the ChromeDriver executable in your system PATH or the script directory.

3. **BeautifulSoup**
   - Ensure the `beautifulsoup4` library is installed (included in `requirements.txt`).

---

#### Steps to Crawl Apartment.com Data

##### 1. Prepare City List
The script includes a predefined list of cities in Los Angeles County. You can modify the `cities` list in the code to include or exclude specific cities.

##### 2. Scrape Apartment Listings
Run the following command to fetch apartment listings from Apartments.com:
```bash
python src/get_data/Apartment_listing_data_scrape.py
```
And we will get a generated output file as a raw data in `data/raw/apartments_listings_with_details.csv`

### Crime Data Download
The large file is hosted on Google Drive. You can download it from the following link:
[Download File](https://drive.google.com/file/d/1d787NinYFIndo5gkx4pFai6Sgxy-6HOP/view?usp=sharing)
### County of Los Angeles Download
https://lacounty.gov/government/about-la-county/maps-and-geography/

### California ZIP Codes List Download
https://www.unitedstateszipcodes.org/ca/

## Instructions on how to clean the data
### 1. Filter California Cities with ZIP Codes
The `filtered_CA_cities_zip_code_database.py` script filters California ZIP code data to create a structured database for analysis.

Run the following command:
```bash
python src/clean_data/filtered_CA_cities_zip_code_database.py
```
### 2. Clean Apartment Listing Data
The `Apartment_listing_data_cleaning.py` script processes raw apartment listing data and saves cleaned results to the data/processed/ directory.

Run the following command:
```bash
python src/clean_data/Apartment_listing_data_cleaning.py
```
### 3. Generate Postal Codes for Crime Data
The `crime_data_generate_postalcode.py` script maps postal codes to crime data records.

Run the following command:
```bash
python src/clean_data/crime_data_generate_postalcode.py
```
### 4. Clean Crime Data
The `crime_data_cleaning.py` script processes raw crime data, removing inconsistencies and preparing it for analysis.

Run the following command:
```bash
python src/clean_data/crime_data_cleaning.py
```
### Outputs

After running these scripts, you will find the cleaned and processed data in the data/processed/ directory. The files generated include:
- Cleaned apartment listing data
- Cleaned and postal-code-mapped crime data
- Filtered California ZIP codes

## Instrucions on how to run analysis code
Before entering this step, please make sure that the following dataset are present in the data/processed/ directory
- cleaned_rental_data_with_postalcode.csv
- crime_with_postalcode_cityname.csv
### 1. RQ1
The script uses argument parsing to execute different analyses. Use the following command structure to run the script:
```bash
python src/run_analysis/RQ1_Crime_and_Price_Correlation_Analysis.py -f <function_name> -m <model_name>
```
Arguments:
- `-f`: Specifies the function to run. Options are:
    - `result_group_by_cities`: Analyze average rental prices grouped by cities and their correlation with crime counts.
    - `result_group_by_la_postal`: Analyze rental prices and crime data for Los Angeles postal codes.
    - `remove_unimportant_features`: Perform feature selection and model evaluation to remove less significant predictors.
- `-m`: Specifies the machine learning model to use (default is "lr"). Options are:
    - `"lr"`: Linear Regression
    - `"rf"`: Random Forest Regression
    - `"xg"`: XGBoost Regression
    - `"cat"`: CatBoost Regression

Examples:

1. Analyze Relationship Between Crime Counts and Average Rental Prices by City:
```bash
python src/run_analysis/RQ1_Crime_and_Price_Correlation_Analysis.py -f result_group_by_cities
```

2.Analyze Data for Los Angeles Postal Codes with Random Forest Regression:
```bash
python src/run_analysis/RQ1_Crime_and_Price_Correlation_Analysis.py -f result_group_by_la_postal -m rf
``` 

3. Remove Unimportant Features and Use Random Forest Regression:
```bash
python src/run_analysis/RQ1_Crime_and_Price_Correlation_Analysis.py -f remove_unimportant_features -m rf
```

4. Remove Unimportant Features and Use XGBoost Regression:
```bash
python src/run_analysis/RQ1_Crime_and_Price_Correlation_Analysis.py -f remove_unimportant_features -m xg
```

5. Remove Unimportant Features and Use CatBoost Regression:
```bash
python src/run_analysis/RQ1_Crime_and_Price_Correlation_Analysis.py -f remove_unimportant_features -m cat
```
### 2. RQ2
The script uses argument parsing to execute different analyses. Use the following command structure to run the script:
```bash
python src/run_analysis/RQ2_Amenity_and_Crime_Pattern_Analysis.py -f <function_name> -m <model_name>
```
Arguments:
- `-f`: Specifies the function to run. Options are:
    - `high_crime_area`: Analyze high-crime areas based on weighted crime scores.
    - `heat_map`: Generate a heatmap to visualize the relationship between amenities and crime types.
- `-m`: Specifies the machine learning model to use (default is "lr"). Options are:
    - `"lr"`: Logistic Regression
    - `"rf"`: Random Forest Regression

Examples:

1. **Analyze High-Crime Areas Using Logistic Regression**:
```bash
python src/run_analysis/RQ2_Amenity_and_Crime_Pattern_Analysis.py -f high_crime_area -m lr
```

2. **Analyze High-Crime Areas Using Random Forest**:
```bash
python src/run_analysis/RQ2_Amenity_and_Crime_Pattern_Analysis.py -f high_crime_area -m rf
``` 

3. **Generate Heatmap of Crime Categories vs. Amenities**:
```bash
python src/run_analysis/RQ2_Amenity_and_Crime_Pattern_Analysis.py -f heat_map
```
### 3. RQ3
The script uses argument parsing to execute different analyses. Use the following command structure to run the script:
```bash
python src/run_analysis/RQ3_Safety_Score_and_Satisfaction_Prediction.py -f <function_name> -m <model_name>
```
Arguments:
- `-f`: Specifies the function to run. Options are:
    - `overall_postal_code`: Analyze overall postal codes and their relationships between property ratings and amenities.
    - `high_crime_area`: Analyze high-crime areas and their safety-related features using regression models.
    - `amenity_among_crime_level`: Evaluate the relationship between crime levels and satisfaction differences based on the presence of amenities.
- `-m`: Specifies the machine learning model to use (default is "lr"). Options are:
    - `"lr"`: Linear Regression
    - `"rf"`: Random Forest Regression

Examples:
1. **Analyze Overall Postal Code Data Using Linear Regression**:
```bash
python src/run_analysis/RQ3_Safety_Score_and_Satisfaction_Prediction.py -f overall_postal_code
```
2.	**Analyze High-Crime Areas Using Random Forest Regression**:
```bash
python src/run_analysis/RQ3_Safety_Score_and_Satisfaction_Prediction.py -f high_crime_area -m rf
```
3.	**Analyze Satisfaction Differences Across Crime Levels and Amenity Presence**:
```bash
python src/run_analysis/RQ3_Safety_Score_and_Satisfaction_Prediction.py -f amenity_among_crime_level
```
## Visualization Results

The visualization results for the analyses performed in this project are available as Jupyter Notebook files. Each notebook corresponds to a specific research question and contains detailed visualizations generated from the analysis results.

---

### Notebooks

1. **RQ1_visualization.ipynb**:
   - Contains visualizations related to the relationship between crime and rental prices.
   - Includes scatter plots, regression lines, bar chart, and table representing models' performances.

2. **RQ2_visualization.ipynb**:
   - Presents visualizations showing patterns between amenities and crime types.
   - Contains heatmaps, feature importance plots, stacked bar chart, and pie chart visualizations.

3. **RQ3_visualization.ipynb**:
   - Visualizations are related to safety scores, satisfaction ratings, and crime levels.
   - Utilizes scatter plots, regression lines, feature importance bar plots, and stacked bar plot.
