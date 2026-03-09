# =============================================================================
# SCRIPT 02 -- Clean, Merge and Enrich with xG Data
# =============================================================================
# WHAT THIS SCRIPT DOES:
#   1. Loads the 4 raw season CSV files downloaded by Script 01
#   2. Cleans and merges them into one combined table
#   3. Fetches xG (expected goals) data from understat.com for each season
#   4. Merges xG into the combined table
#   5. Saves the final enriched file to data/processed/matches.csv
#
# HOW TO RUN IT:
#   python3 src/02_clean_and_enrich.py
#
# WHAT YOU'LL GET:
#   data/processed/matches.csv  -- one row per match, all seasons, with xG
#   data/processed/summary.txt  -- a readable summary of what was produced
# =============================================================================

import pandas as pd    # working with tables
import requests        # downloading from the web
import json            # parsing JSON data (a common web data format)
import re              # regular expressions (for finding patterns in text)
import os              # file and folder operations
import time            # pausing between web requests


# --- CONFIGURATION -----------------------------------------------------------

RAW_DATA_FOLDER  = "data/raw"
PROCESSED_FOLDER = "data/processed"

# Seasons to process: (our label, year for understat URL)
SEASONS = [
    ("2021-22", 2021),
    ("2022-23", 2022),
    ("2023-24", 2023),
    ("2024-25", 2024),
    ("2025-26", 2025),   # <-- 2025-26 season added
]

# Columns to keep from the raw files
COLUMNS_TO_KEEP = [
    "Date",      # match date
    "HomeTeam",  # home team
    "AwayTeam",  # away team
    "FTHG",      # Full Time Home Goals
    "FTAG",      # Full Time Away Goals
    "FTR",       # Full Time Result (H/D/A)
    "HS",        # Home Shots
    "AS",        # Away Shots
    "HST",       # Home Shots on Target
    "AST",       # Away Shots on Target
    "HY",        # Home Yellow Cards
    "AY",        # Away Yellow Cards
    "HR",        # Home Red Cards
    "AR",        # Away Red Cards
    "B365H",     # Bet365 Home Win odds  (for model validation later)
    "B365D",     # Bet365 Draw odds
    "B365A",     # Bet365 Away Win odds
]

# Understat uses slightly different team names to football-data.co.uk
# This dictionary maps football-data names -> understat names
# "Key": "Value" means: when we see Key in our data, look for Value in understat
TEAM_NAME_MAP = {
    "Man City":        "Manchester City",
    "Man United":      "Manchester United",
    "Newcastle":       "Newcastle United",
    "Tottenham":       "Tottenham",
    "Nott'm Forest":   "Nottingham Forest",
    "Sheffield United":"Sheffield United",
    "West Ham":        "West Ham",
    "Wolves":          "Wolverhampton Wanderers",
    "Brighton":        "Brighton",
    "Aston Villa":     "Aston Villa",
    "Brentford":       "Brentford",
    "Crystal Palace":  "Crystal Palace",
    "Fulham":          "Fulham",
    "Bournemouth":     "Bournemouth",
    "Everton":         "Everton",
    "Arsenal":         "Arsenal",
    "Chelsea":         "Chelsea",
    "Liverpool":       "Liverpool",
    "Burnley":         "Burnley",
    "Luton":           "Luton",
    "Ipswich":         "Ipswich",
    "Leicester":       "Leicester",
    "Southampton":     "Southampton",
}


# --- STEP 1: LOAD AND CLEAN RAW FILES ----------------------------------------

def load_and_clean_all_seasons():
    """
    Loads all 4 raw CSV files, cleans each one, and returns a single
    combined DataFrame with all 1520 matches.
    """
    print("[Step 1] Loading and cleaning raw season files...")

    all_seasons = []

    for season_label, _ in SEASONS:
        filepath = RAW_DATA_FOLDER + "/epl_" + season_label + ".csv"

        if not os.path.exists(filepath):
            print("  [WARN] File not found, skipping: " + filepath)
            continue

        # encoding='utf-8-sig' automatically strips the 'BOM' marker that
        # caused the garbled 'Div' column name we saw earlier
        df = pd.read_csv(filepath, encoding="utf-8-sig")

        # Drop completely empty rows
        df = df.dropna(how="all")

        # Keep only the columns we want (that also exist in this file)
        cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
        df = df[cols].copy()

        # Add season label
        df["Season"] = season_label

        # Parse dates. dayfirst=True because format is DD/MM/YYYY
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

        # Remove rows without a score (unplayed future fixtures)
        before = len(df)
        df = df.dropna(subset=["FTHG", "FTAG"])
        removed = before - len(df)
        if removed > 0:
            print("  [INFO] " + season_label + ": removed " + str(removed) +
                  " rows with no score")

        # Convert goals from float to int (they come in as 1.0, 2.0 etc.)
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)

        all_seasons.append(df)
        print("  [OK] " + season_label + ": " + str(len(df)) + " matches loaded")

    if not all_seasons:
        raise Exception("No season files found. Have you run Script 01 first?")

    # Stack all seasons into one table, sorted by date
    combined = pd.concat(all_seasons, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)

    print("  [OK] Combined total: " + str(len(combined)) + " matches")
    return combined


# --- STEP 2: FETCH xG FROM UNDERSTAT -----------------------------------------

def fetch_understat_xg(season_year):
    """
    Fetches xG data for one season from understat.com.

    Understat embeds all match data as JSON inside the HTML page.
    We download the page, extract the JSON, and parse it.

    season_year: the year the season started, e.g. 2023 for 2023-24

    Returns a DataFrame with columns:
    HomeTeam_us, AwayTeam_us, xG_home, xG_away, Date
    Or None if the fetch failed.
    """
    url = "https://understat.com/league/EPL/" + str(season_year)

    # We set a User-Agent header to identify ourselves as a normal browser.
    # Without this some websites block automated requests.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # The match data is embedded in the HTML like this:
        #   var datesData = JSON.parse('[ {...}, {...} ]')
        # We use a regular expression to find and extract it.
        # r"..." is a "raw string" -- backslashes are treated literally
        pattern = r"datesData\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, response.text)

        if not match:
            print("  [WARN] Could not find match data in understat page")
            return None

        # The JSON is unicode-escaped, so we decode it first
        raw_json = match.group(1).encode().decode("unicode_escape")
        matches = json.loads(raw_json)

        # Flatten the nested JSON into a simple list of rows
        rows = []
        for m in matches:
            # Skip matches with no xG yet (future or very early games)
            if not m.get("xG") or not m["xG"].get("h"):
                continue
            try:
                rows.append({
                    "HomeTeam_us": m["h"]["title"],
                    "AwayTeam_us": m["a"]["title"],
                    "xG_home":     round(float(m["xG"]["h"]), 3),
                    "xG_away":     round(float(m["xG"]["a"]), 3),
                    # Parse date -- understat format: "2023-08-12 12:30:00"
                    "Date_us": pd.to_datetime(m["datetime"]).date(),
                })
            except (KeyError, ValueError):
                continue

        df_xg = pd.DataFrame(rows)
        print("  [OK] Understat " + str(season_year) + ": " +
              str(len(df_xg)) + " matches with xG")
        return df_xg

    except requests.exceptions.ConnectionError:
        print("  [WARN] Could not connect to understat.com")
        return None
    except Exception as e:
        print("  [WARN] Understat fetch failed: " + str(e))
        return None


def fetch_all_xg():
    """
    Fetches xG for all seasons and combines into one DataFrame.
    """
    print("")
    print("[Step 2] Fetching xG data from understat.com...")

    all_xg = []

    for season_label, season_year in SEASONS:
        print("  Season " + season_label + "...")
        df_xg = fetch_understat_xg(season_year)
        if df_xg is not None and len(df_xg) > 0:
            df_xg["Season"] = season_label
            all_xg.append(df_xg)
        time.sleep(2)  # polite pause between requests

    if not all_xg:
        return None

    return pd.concat(all_xg, ignore_index=True)


# --- STEP 3: MERGE xG INTO MATCH DATA ----------------------------------------

def merge_xg(df_matches, df_xg):
    """
    Merges xG data into the main matches DataFrame.

    The challenge: football-data.co.uk and understat use different team names
    (e.g. "Man City" vs "Manchester City"). We use TEAM_NAME_MAP to translate.

    We match on: translated home team name + translated away team name + date.
    """
    print("")
    print("[Step 3] Merging xG into match data...")

    # Add translated team name columns to the matches DataFrame
    # The .get(x, x) pattern means: look up x in the map;
    # if not found, use x unchanged (already correct name)
    df_matches["HomeTeam_us"] = df_matches["HomeTeam"].apply(
        lambda x: TEAM_NAME_MAP.get(x, x)
    )
    df_matches["AwayTeam_us"] = df_matches["AwayTeam"].apply(
        lambda x: TEAM_NAME_MAP.get(x, x)
    )

    # Add a date-only column to match on (understat has time, we just want date)
    df_matches["Date_us"] = df_matches["Date"].dt.date

    # Merge: find matching rows between the two DataFrames
    # on= specifies which columns must match
    # how="left" means: keep ALL rows from df_matches, add xG where found
    # (if no xG match found, the xG columns will be blank/NaN)
    df_merged = df_matches.merge(
        df_xg[["HomeTeam_us", "AwayTeam_us", "Date_us", "xG_home", "xG_away", "Season"]],
        on=["HomeTeam_us", "AwayTeam_us", "Date_us", "Season"],
        how="left"
    )

    # Count how many matches got xG data
    matched = df_merged["xG_home"].notna().sum()
    total   = len(df_merged)
    pct     = round(100 * matched / total, 1)
    print("  [OK] xG matched for " + str(matched) + " of " + str(total) +
          " matches (" + str(pct) + "%)")

    # Clean up the temporary merge columns
    df_merged = df_merged.drop(columns=["HomeTeam_us", "AwayTeam_us", "Date_us"])

    return df_merged


# --- STEP 4: ADD USEFUL DERIVED COLUMNS --------------------------------------

def add_derived_columns(df):
    """
    Adds extra columns that will be useful for the ratings model:
    - TotalGoals: total goals in the match
    - GoalDiff: home goals minus away goals (positive = home win)
    - xG_total: total xG in the match
    - xG_diff: home xG minus away xG
    """
    print("")
    print("[Step 4] Adding derived columns...")

    df["TotalGoals"] = df["FTHG"] + df["FTAG"]
    df["GoalDiff"]   = df["FTHG"] - df["FTAG"]

    # Only calculate xG derived columns where we have xG data
    if "xG_home" in df.columns:
        df["xG_total"] = df["xG_home"] + df["xG_away"]
        df["xG_diff"]  = df["xG_home"] - df["xG_away"]

    print("  [OK] Derived columns added")
    return df


# --- STEP 5: PRINT SUMMARY AND SAVE ------------------------------------------

def print_summary(df):
    """
    Prints a readable summary of the final dataset and saves it to a text file.
    """
    print("")
    print("[Step 5] Summary")
    print("-" * 50)

    lines = []
    lines.append("EPL Score Predictor -- Dataset Summary")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Total matches: " + str(len(df)))
    lines.append("Date range:    " + str(df["Date"].min().date()) +
                 " to " + str(df["Date"].max().date()))
    lines.append("")

    lines.append("Matches per season:")
    for season, count in df.groupby("Season").size().items():
        lines.append("  " + str(season) + ": " + str(count))
    lines.append("")

    lines.append("Results breakdown:")
    lines.append("  Home wins: " + str((df["FTR"] == "H").sum()) +
                 " (" + str(round(100*(df["FTR"]=="H").mean(), 1)) + "%)")
    lines.append("  Draws:     " + str((df["FTR"] == "D").sum()) +
                 " (" + str(round(100*(df["FTR"]=="D").mean(), 1)) + "%)")
    lines.append("  Away wins: " + str((df["FTR"] == "A").sum()) +
                 " (" + str(round(100*(df["FTR"]=="A").mean(), 1)) + "%)")
    lines.append("")

    lines.append("Average goals per game: " +
                 str(round(df["TotalGoals"].mean(), 2)))
    lines.append("Average home goals:     " +
                 str(round(df["FTHG"].mean(), 2)))
    lines.append("Average away goals:     " +
                 str(round(df["FTAG"].mean(), 2)))
    lines.append("")

    if "xG_home" in df.columns:
        xg_coverage = df["xG_home"].notna().sum()
        lines.append("xG data available for " + str(xg_coverage) +
                     " matches (" +
                     str(round(100 * xg_coverage / len(df), 1)) + "%)")
        if xg_coverage > 0:
            lines.append("Average home xG: " +
                         str(round(df["xG_home"].mean(), 2)))
            lines.append("Average away xG: " +
                         str(round(df["xG_away"].mean(), 2)))
        lines.append("")

    lines.append("Top 5 most attacking teams (avg goals/game home):")
    top5 = df.groupby("HomeTeam")["FTHG"].mean().sort_values(
        ascending=False).head(5)
    for team, avg in top5.items():
        lines.append("  " + str(team) + ": " + str(round(avg, 2)))
    lines.append("")

    lines.append("Columns in final dataset:")
    for col in df.columns:
        lines.append("  - " + col)

    # Print to terminal
    for line in lines:
        print(line)

    # Save to file
    summary_path = PROCESSED_FOLDER + "/summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print("")
    print("  [OK] Summary saved to " + summary_path)


# --- MAIN --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  EPL Score Predictor -- Script 02: Clean and Enrich")
    print("=" * 60)

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    # Step 1: Load and clean raw files
    df_matches = load_and_clean_all_seasons()

    # Step 2: Fetch xG from understat
    df_xg = fetch_all_xg()

    # Step 3: Merge xG (if we got it)
    if df_xg is not None and len(df_xg) > 0:
        df_final = merge_xg(df_matches, df_xg)
    else:
        print("")
        print("[Step 3] Skipping xG merge (understat unavailable)")
        print("  The model will still work -- we will use shots on target")
        print("  as a proxy for xG until understat data is available.")
        df_final = df_matches.copy()

    # Step 4: Add derived columns
    df_final = add_derived_columns(df_final)

    # Save the final file
    output_path = PROCESSED_FOLDER + "/matches.csv"
    df_final.to_csv(output_path, index=False)
    print("")
    print("  [OK] Final dataset saved: " + output_path)

    # Step 5: Print and save summary
    print_summary(df_final)

    print("")
    print("=" * 60)
    print("  ALL DONE -- Ready for Script 03 (Team Ratings)!")
    print("=" * 60)
    print("")


if __name__ == "__main__":
    main()
