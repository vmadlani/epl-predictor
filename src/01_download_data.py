# =============================================================================
# SCRIPT 01 -- Download Historical EPL Data
# =============================================================================
# WHAT THIS SCRIPT DOES:
#   Downloads 4 seasons of Premier League match results from football-data.co.uk
#   and saves them as CSV files in your data/raw folder.
#
# HOW TO RUN IT:
#   python3 src/01_download_data.py
#
# WHAT YOU'LL GET:
#   One CSV file per season in data/raw/, e.g. epl_2023-24.csv
#   A combined file of all seasons:  data/processed/matches.csv
# =============================================================================


# --- IMPORTS -----------------------------------------------------------------
# "import" means: go get this toolkit and make it available in this script.

import requests   # lets us download files from the internet
import pandas     # lets us work with tables of data (like Excel in Python)
import os         # lets us check if folders/files exist and create them
import time       # lets us pause between downloads (polite to servers)
import io         # lets us treat downloaded text like a file


# --- CONFIGURATION -----------------------------------------------------------
# All settings in one place. Change seasons or paths here only.

# The seasons we want to download.
# football-data.co.uk uses a shorthand: "2425" means the 2024/25 season.
SEASONS = [
    ("2021-22", "2122"),
    ("2022-23", "2223"),
    ("2023-24", "2324"),
    ("2024-25", "2425"),
    ("2025-26", "2526"),   # <-- 2025-26 season added
]

# The base URL pattern. {code} gets replaced with e.g. "2324"
URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# Where to save files
RAW_DATA_FOLDER = "data/raw"
PROCESSED_DATA_FOLDER = "data/processed"

# The columns we need from the raw files (they have 50+ columns, we keep these)
COLUMNS_TO_KEEP = [
    "Date",    # match date
    "HomeTeam",# home team name
    "AwayTeam",# away team name
    "FTHG",    # Full Time Home Goals
    "FTAG",    # Full Time Away Goals
    "FTR",     # Full Time Result: H=Home win, D=Draw, A=Away win
    "HS",      # Home Shots
    "AS",      # Away Shots
    "HST",     # Home Shots on Target
    "AST",     # Away Shots on Target
    "B365H",   # Bet365 odds: Home win  (for benchmarking our model later)
    "B365D",   # Bet365 odds: Draw
    "B365A",   # Bet365 odds: Away win
]


# --- HELPER FUNCTIONS --------------------------------------------------------

def ensure_folder_exists(folder_path):
    """
    Creates a folder if it does not already exist.
    exist_ok=True means: do not throw an error if it already exists.
    """
    os.makedirs(folder_path, exist_ok=True)
    print("  [OK] Folder ready: " + folder_path)


def download_season(season_label, season_code):
    """
    Downloads one season's data from football-data.co.uk.
    Returns a pandas DataFrame (a table) or None if the download failed.
    """

    url = URL_TEMPLATE.format(code=season_code)
    print("")
    print("  Downloading " + season_label + "...")
    print("  URL: " + url)

    # try/except: attempt the code in 'try', and if anything goes wrong
    # catch the error and handle it gracefully instead of crashing.
    try:
        # requests.get() downloads the content at the URL.
        # timeout=30 means: give up if no response within 30 seconds.
        response = requests.get(url, timeout=30)

        # status_code 200 = OK. raise_for_status() errors on anything else.
        response.raise_for_status()

        # Read the CSV content directly into a pandas DataFrame (a table object)
        df = pandas.read_csv(io.StringIO(response.text))

        # Drop any completely empty rows (sometimes appear at end of file)
        df = df.dropna(how="all")

        print("  [OK] Downloaded " + str(len(df)) + " matches")
        return df

    except requests.exceptions.HTTPError as e:
        print("  [ERROR] HTTP error: " + str(e))
        return None
    except requests.exceptions.ConnectionError:
        print("  [ERROR] Connection failed -- are you connected to the internet?")
        return None
    except Exception as e:
        print("  [ERROR] Unexpected error: " + str(e))
        return None


def clean_season_data(df, season_label):
    """
    Cleans a single season's DataFrame:
    - Keeps only the columns we need
    - Converts Date to a proper date type
    - Adds a Season column
    - Removes rows with missing scores (unplayed fixtures)
    - Converts goal columns from float to integer
    """

    # Keep only columns that exist in this file AND are in our wanted list.
    # Some older seasons may not have every column, so we check first.
    available_cols = [col for col in COLUMNS_TO_KEEP if col in df.columns]
    df = df[available_cols].copy()

    # Add a season label so after merging all seasons we know which is which
    df["Season"] = season_label

    # Convert the Date column to a proper datetime type.
    # dayfirst=True because the files use DD/MM/YY format.
    # errors='coerce' turns unparseable dates into blanks rather than crashing.
    if "Date" in df.columns:
        df["Date"] = pandas.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Remove rows where the score is missing (unplayed future fixtures)
    if "FTHG" in df.columns and "FTAG" in df.columns:
        before = len(df)
        df = df.dropna(subset=["FTHG", "FTAG"])
        removed = before - len(df)
        if removed > 0:
            print("  [INFO] Removed " + str(removed) + " rows with missing scores")

    # Convert goal columns from float to integer (they download as 1.0, 2.0 etc.)
    for col in ["FTHG", "FTAG"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


# --- MAIN FUNCTION -----------------------------------------------------------

def main():
    print("=" * 60)
    print("  EPL Score Predictor -- Script 01: Download Data")
    print("=" * 60)

    # Step 1: Make sure our folders exist
    print("")
    print("[Step 1] Setting up folders...")
    ensure_folder_exists(RAW_DATA_FOLDER)
    ensure_folder_exists(PROCESSED_DATA_FOLDER)

    # Step 2: Download each season
    print("")
    print("[Step 2] Downloading seasons from football-data.co.uk...")

    # Collect each season's cleaned data here, then combine at the end
    all_seasons = []

    for season_label, season_code in SEASONS:

        # Download raw data for this season
        df_raw = download_season(season_label, season_code)

        # If download failed, skip to the next season
        if df_raw is None:
            print("  Skipping " + season_label)
            continue

        # Save the raw file exactly as downloaded -- never modify raw files
        raw_filename = RAW_DATA_FOLDER + "/epl_" + season_label + ".csv"
        df_raw.to_csv(raw_filename, index=False)
        print("  [OK] Raw file saved: " + raw_filename)

        # Clean the data for this season
        df_clean = clean_season_data(df_raw, season_label)
        print("  [OK] Cleaned: " + str(len(df_clean)) + " valid matches retained")

        # Add to our collection
        all_seasons.append(df_clean)

        # Pause 1 second between downloads -- polite to the server
        time.sleep(1)

    # Step 3: Combine all seasons into one file
    print("")
    print("[Step 3] Combining all seasons...")

    if not all_seasons:
        print("  [ERROR] No data downloaded. Check your internet connection.")
        return

    # pandas.concat() stacks multiple DataFrames on top of each other
    df_all = pandas.concat(all_seasons, ignore_index=True)

    # Sort by date so matches are in chronological order
    df_all = df_all.sort_values("Date").reset_index(drop=True)

    # Save the combined file
    combined_filename = PROCESSED_DATA_FOLDER + "/matches.csv"
    df_all.to_csv(combined_filename, index=False)

    # Step 4: Print a summary
    print("")
    print("=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
    print("")
    print("  Combined file: " + combined_filename)
    print("  Total matches: " + str(len(df_all)))
    print("")
    print("  Breakdown by season:")

    season_counts = df_all.groupby("Season").size().reset_index(name="Matches")
    for _, row in season_counts.iterrows():
        print("    " + str(row["Season"]) + ": " + str(row["Matches"]) + " matches")

    print("")
    print("  Date range: " + str(df_all["Date"].min().date()) + " to " + str(df_all["Date"].max().date()))
    print("")
    print("  Columns saved:")
    for col in df_all.columns:
        print("    - " + col)

    print("")
    print("  Ready for Script 02!")
    print("")


# --- RUN ---------------------------------------------------------------------
# This block only runs when you execute this file directly.
# It does NOT run if another script imports this one.

if __name__ == "__main__":
    main()
