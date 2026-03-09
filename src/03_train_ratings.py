# =============================================================================
# SCRIPT 03 -- Train Team Ratings (Dixon-Coles Model)
# =============================================================================
# WHAT THIS SCRIPT DOES:
#   Reads all historical match data and fits a Dixon-Coles model to learn
#   an attack rating and defence rating for every Premier League team.
#
# HOW TO RUN IT:
#   python3 src/03_train_ratings.py
#
# WHAT YOU'LL GET:
#   outputs/team_ratings.csv   -- ratings for every team (current + historical)
#   outputs/model_params.json  -- home advantage and rho values
# =============================================================================

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import os
import json


# --- CONFIGURATION -----------------------------------------------------------

PROCESSED_FOLDER = "data/processed"
OUTPUTS_FOLDER   = "outputs"

DECAY_RATE       = 0.003   # 0.003/day = ~half weight after 8 months
MAX_DAYS_HISTORY = 365 * 3 # ignore matches older than 3 seasons

# The 20 teams currently in the Premier League for 2025-26.
# UPDATE THIS LIST each summer when promotion/relegation is confirmed.
#
# CHANGES FROM 2024-25:
#   Relegated: Leicester, Ipswich, Southampton
#   Promoted:  Sunderland, Burnley, Leeds
CURRENT_TEAMS = [
    "Arsenal",       "Aston Villa",   "Bournemouth",    "Brentford",
    "Brighton",      "Burnley",       "Chelsea",        "Crystal Palace",
    "Everton",       "Fulham",        "Leeds",           "Liverpool",
    "Man City",      "Man United",    "Newcastle",       "Nott'm Forest",
    "Sunderland",    "Tottenham",     "West Ham",        "Wolves",
]


# --- LOAD DATA ---------------------------------------------------------------

def load_matches():
    """
    Loads cleaned match data from Script 02.
    Applies time decay weights and removes very old matches.
    """
    path = PROCESSED_FOLDER + "/matches.csv"
    if not os.path.exists(path):
        raise Exception("matches.csv not found. Run Script 02 first.")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    ref_date = df["Date"].max()
    df["days_ago"] = (ref_date - df["Date"]).dt.days

    before = len(df)
    df = df[df["days_ago"] <= MAX_DAYS_HISTORY].copy()
    removed = before - len(df)
    if removed > 0:
        print("  [INFO] Removed " + str(removed) + " matches older than " +
              str(MAX_DAYS_HISTORY) + " days")

    # Time decay: recent matches get weight close to 1.0
    # Old matches get weight close to 0.0
    df["weight"] = np.exp(-DECAY_RATE * df["days_ago"])

    print("  [OK] " + str(len(df)) + " matches loaded")
    print("  Date range: " + str(df["Date"].min().date()) +
          " to " + str(df["Date"].max().date()))
    print("  Weight range: " + str(round(df["weight"].min(), 3)) +
          " to " + str(round(df["weight"].max(), 3)))
    return df


# --- DIXON-COLES MODEL -------------------------------------------------------

def build_team_index(df):
    """
    Creates a sorted list of all teams that appear in the data and a
    dictionary mapping team name -> index number.
    Note: this includes relegated teams (they have historical data),
    but we flag current teams separately in the output.
    """
    teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
    idx   = {team: i for i, team in enumerate(teams)}
    return teams, idx


def dixon_coles_neg_log_likelihood(params, df, teams, idx):
    """
    The Dixon-Coles negative log likelihood.

    The optimiser minimises this function. The minimum corresponds to the
    attack/defence values that best explain all historical match results,
    weighted by recency (time decay).

    Parameters layout:
      params[0 .. n-1]   = attack rating per team (log scale)
      params[n .. 2n-1]  = defence rating per team (log scale)
      params[2n]         = home advantage (log scale)
      params[2n+1]       = rho (Dixon-Coles low-score correction)

    Why log scale? Expected goals must always be positive.
    exp(any number) is always > 0, so working in log scale is safe.
    """
    n        = len(teams)
    attack   = params[:n]
    defence  = params[n:2*n]
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    h_idx = df["HomeTeam"].map(idx).values
    a_idx = df["AwayTeam"].map(idx).values
    hg    = df["FTHG"].values
    ag    = df["FTAG"].values
    w     = df["weight"].values

    # Expected goals for each team in each match
    mu_h = np.exp(attack[h_idx] + defence[a_idx] + home_adv)
    mu_a = np.exp(attack[a_idx] + defence[h_idx])

    # Poisson log probabilities (vectorised)
    ll = poisson.logpmf(hg, mu_h) + poisson.logpmf(ag, mu_a)

    # Dixon-Coles correction: adjusts probabilities for 0-0, 1-0, 0-1, 1-1
    # This fixes Poisson's tendency to underpredict draws
    dc = np.zeros(len(df))
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)
    dc[m00] = np.log(np.maximum(1 - mu_h[m00] * mu_a[m00] * rho, 1e-10))
    dc[m10] = np.log(np.maximum(1 + mu_a[m10] * rho, 1e-10))
    dc[m01] = np.log(np.maximum(1 + mu_h[m01] * rho, 1e-10))
    dc[m11] = np.log(np.maximum(1 - rho, 1e-10))

    return -np.sum(w * (ll + dc))


def train_model(df, teams, idx):
    """
    Runs the Dixon-Coles optimisation using scipy's SLSQP solver.
    Returns the fitted parameter array.
    """
    n  = len(teams)
    x0 = np.zeros(2 * n + 2)
    x0[2*n]     = 0.1  # initial home advantage guess
    x0[2*n + 1] = 0.1  # initial rho guess

    # Constraint: average attack = 0
    # Without this the model is under-identified (attack + defence can both
    # shift together without changing predictions). Anchoring the mean fixes it.
    constraints = [{"type": "eq", "fun": lambda p: np.mean(p[:n])}]

    print("  Running optimiser (" + str(2*n+2) + " parameters, " +
          str(len(df)) + " matches)...")
    print("  This typically takes 15-30 seconds...")

    result = minimize(
        dixon_coles_neg_log_likelihood,
        x0,
        args=(df, teams, idx),
        method="SLSQP",
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-7, "disp": False}
    )

    if not result.success:
        print("  [WARN] Optimiser did not fully converge: " + result.message)
        print("  Results are still usable.")
    else:
        print("  [OK] Optimisation converged successfully")

    return result.x


def build_ratings_table(params, teams):
    """
    Converts the raw parameter array into a readable DataFrame.
    Adds a 'Current' column flagging which teams are in the 2024-25 season.
    """
    n        = len(teams)
    attack   = params[:n]
    defence  = params[n:2*n]
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    ratings = pd.DataFrame({
        "Team":          teams,
        "Attack":        attack.round(4),
        "Defence":       defence.round(4),
        # Exp_Scored:   expected goals scored vs average team (neutral venue)
        # Exp_Conceded: expected goals conceded vs average team (neutral venue)
        "Exp_Scored":    np.exp(attack).round(3),
        "Exp_Conceded":  np.exp(defence).round(3),
        # Flag which teams are currently in the Premier League
        "Current":       [t in CURRENT_TEAMS for t in teams],
    })

    # Sort by attack rating, best first
    ratings = ratings.sort_values("Attack", ascending=False).reset_index(drop=True)
    ratings.index = ratings.index + 1  # rank from 1

    return ratings, home_adv, rho


# --- PRINT AND SAVE ----------------------------------------------------------

def print_ratings(ratings, home_adv, rho):
    """
    Prints the ratings table. Current teams shown normally.
    Historical-only teams (relegated etc.) shown with a note.
    """
    current  = ratings[ratings["Current"] == True]
    historic = ratings[ratings["Current"] == False]

    print()
    print("  Home advantage: x" + str(round(np.exp(home_adv), 3)) +
          " goals  (log: " + str(round(home_adv, 4)) + ")")
    print("  Rho (draw correction): " + str(round(rho, 4)))
    print()

    header = "  {:3s}  {:22s}  {:8s}  {:10s}  {:8s}  {:10s}".format(
        "Rnk", "Team", "Attack", "Exp Scored", "Defence", "Exp Concd")
    divider = "  " + "-" * 70

    print("  2024-25 PREMIER LEAGUE TEAMS (20)")
    print(divider)
    print(header)
    print(divider)
    for rank, row in current.iterrows():
        print("  {:3d}  {:22s}  {:8.4f}  {:10.3f}  {:8.4f}  {:10.3f}".format(
            rank, str(row["Team"]), row["Attack"], row["Exp_Scored"],
            row["Defence"], row["Exp_Conceded"]))
    print(divider)

    if len(historic) > 0:
        print()
        print("  RELEGATED / HISTORICAL TEAMS (have data but not in 2024-25)")
        print(divider)
        print(header)
        print(divider)
        for rank, row in historic.iterrows():
            print("  {:3d}  {:22s}  {:8.4f}  {:10.3f}  {:8.4f}  {:10.3f}".format(
                rank, str(row["Team"]), row["Attack"], row["Exp_Scored"],
                row["Defence"], row["Exp_Conceded"]))
        print(divider)


def save_outputs(ratings, home_adv, rho):
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

    # Save full ratings (all teams including historical)
    ratings_path = OUTPUTS_FOLDER + "/team_ratings.csv"
    ratings.to_csv(ratings_path, index=True, index_label="Rank")
    print("  [OK] Ratings saved: " + ratings_path)

    # Save model parameters needed by Script 04
    params_path = OUTPUTS_FOLDER + "/model_params.json"
    model_params = {
        "home_advantage":    round(float(home_adv), 6),
        "rho":               round(float(rho), 6),
        "decay_rate":        DECAY_RATE,
        "max_days_history":  MAX_DAYS_HISTORY,
        "current_teams":     CURRENT_TEAMS,
    }
    with open(params_path, "w") as f:
        json.dump(model_params, f, indent=2)
    print("  [OK] Model params saved: " + params_path)


# --- MAIN --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  EPL Score Predictor -- Script 03: Train Ratings")
    print("=" * 60)

    print("")
    print("[Step 1] Loading match data...")
    df = load_matches()

    print("")
    print("[Step 2] Building team index...")
    teams, idx = build_team_index(df)
    print("  [OK] " + str(len(teams)) + " teams in historical data")

    # Check all current teams are represented
    missing = [t for t in CURRENT_TEAMS if t not in teams]
    if missing:
        print("  [WARN] These current teams have no historical data: " +
              str(missing))
        print("  They are newly promoted with no prior EPL matches in our data.")
        print("  They will receive average ratings (Attack=0, Defence=0).")

    print("")
    print("[Step 3] Training Dixon-Coles model...")
    params = train_model(df, teams, idx)

    print("")
    print("[Step 4] Building ratings table...")
    ratings, home_adv, rho = build_ratings_table(params, teams)

    # Handle any current teams with no historical data
    # (e.g. a newly promoted team with no EPL history)
    for team in missing:
        new_row = pd.DataFrame([{
            "Team": team, "Attack": 0.0, "Defence": 0.0,
            "Exp_Scored": 1.0, "Exp_Conceded": 1.0, "Current": True
        }])
        ratings = pd.concat([ratings, new_row], ignore_index=True)
        ratings.index = ratings.index + 1

    print("")
    print("[Step 5] Team Ratings")
    print_ratings(ratings, home_adv, rho)

    print("")
    print("[Step 6] Saving outputs...")
    save_outputs(ratings, home_adv, rho)

    print("")
    print("=" * 60)
    print("  ALL DONE -- Ready for Script 04 (Match Predictions)!")
    print("=" * 60)
    print()
    print("  Interpretation guide:")
    print("  Attack  > 0  = scores more than average")
    print("  Attack  < 0  = scores less than average")
    print("  Defence < 0  = concedes less than average (GOOD)")
    print("  Defence > 0  = concedes more than average (BAD)")
    print()


if __name__ == "__main__":
    main()
