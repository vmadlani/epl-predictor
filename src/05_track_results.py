# =============================================================================
# SCRIPT 05 -- Track Prediction Accuracy
# =============================================================================
# WHAT THIS SCRIPT DOES:
#   Compares our model's predictions against actual results to measure
#   how well we're doing. Run it after each gameweek once results are in.
#
# HOW TO RUN IT:
#   python3 src/05_track_results.py
#
# WHAT YOU'LL GET:
#   - Printed accuracy report in the terminal
#   - outputs/tracking.csv  -- full history of every prediction vs result
#   - outputs/accuracy.json -- running accuracy stats
#
# HOW IT WORKS:
#   For each past match, it regenerates our model's prediction and compares
#   it to the actual result. It scores us on three things:
#
#   1. RESULT accuracy  -- did we predict the right outcome? (W/D/L)
#   2. SCORE accuracy   -- did we predict the exact scoreline?
#   3. RANKED PROBABILITY SCORE (RPS) -- a proper statistical measure
#      of how good our probability estimates are. Lower is better.
#      We compare our RPS against Bet365 odds as a benchmark.
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import poisson
import json
import os


# --- CONFIGURATION -----------------------------------------------------------

PROCESSED_FOLDER = "data/processed"
OUTPUTS_FOLDER   = "outputs"

# How many recent gameweeks to show in the terminal report
RECENT_GAMEWEEKS = 5

MAX_GOALS = 9


# --- LOAD DATA ---------------------------------------------------------------

def load_data():
    matches_path = PROCESSED_FOLDER + "/matches.csv"
    ratings_path = OUTPUTS_FOLDER + "/team_ratings.csv"
    params_path  = OUTPUTS_FOLDER + "/model_params.json"

    for p in [matches_path, ratings_path, params_path]:
        if not os.path.exists(p):
            raise Exception("Missing file: " + p + ". Run earlier scripts first.")

    matches = pd.read_csv(matches_path)
    matches["Date"] = pd.to_datetime(matches["Date"])

    ratings = pd.read_csv(ratings_path)
    lookup = {row["Team"]: {"Attack": row["Attack"], "Defence": row["Defence"]}
              for _, row in ratings.iterrows()}

    with open(params_path) as f:
        params = json.load(f)

    return matches, lookup, params


# --- PREDICTION ENGINE (same as Script 04) -----------------------------------

def predict_match(home, away, lookup, home_adv, rho):
    """Returns (home_win_prob, draw_prob, away_win_prob, predicted_home, predicted_away)"""
    if home not in lookup or away not in lookup:
        return None

    mu_h = np.exp(lookup[home]["Attack"] + lookup[away]["Defence"] + home_adv)
    mu_a = np.exp(lookup[away]["Attack"] + lookup[home]["Defence"])

    matrix = np.zeros((MAX_GOALS, MAX_GOALS))
    for hg in range(MAX_GOALS):
        for ag in range(MAX_GOALS):
            p = poisson.pmf(hg, mu_h) * poisson.pmf(ag, mu_a)
            if   hg == 0 and ag == 0: p *= max(1 - mu_h * mu_a * rho, 0)
            elif hg == 1 and ag == 0: p *= max(1 + mu_a * rho, 0)
            elif hg == 0 and ag == 1: p *= max(1 + mu_h * rho, 0)
            elif hg == 1 and ag == 1: p *= max(1 - rho, 0)
            matrix[hg, ag] = p

    home_win = float(np.sum(np.tril(matrix, -1)))
    draw     = float(np.sum(np.diag(matrix)))
    away_win = float(np.sum(np.triu(matrix, 1)))

    scores = [(matrix[h, a], h, a)
              for h in range(MAX_GOALS) for a in range(MAX_GOALS)]
    scores.sort(reverse=True)
    _, pred_h, pred_a = scores[0]

    return home_win, draw, away_win, pred_h, pred_a


# --- RANKED PROBABILITY SCORE ------------------------------------------------

def rps(probs, outcome):
    """
    Ranked Probability Score -- measures how good probability estimates are.
    Lower is better. Perfect certainty on the right outcome = 0.
    Works for 3-outcome events (home win, draw, away win).

    probs:   [p_home_win, p_draw, p_away_win]
    outcome: 0 = home win, 1 = draw, 2 = away win
    """
    actuals = [0.0, 0.0, 0.0]
    actuals[outcome] = 1.0

    cumprob  = [sum(probs[:i+1])   for i in range(3)]
    cumactual = [sum(actuals[:i+1]) for i in range(3)]

    return sum((cumprob[i] - cumactual[i])**2 for i in range(2)) / 2


def bet365_rps(row, actual_outcome):
    """
    Calculates RPS from Bet365 odds for benchmarking.
    Converts decimal odds to implied probabilities (with overround removed).
    Returns None if odds not available.
    """
    try:
        oh = float(row["B365H"])
        od = float(row["B365D"])
        oa = float(row["B365A"])
        if pd.isna(oh) or pd.isna(od) or pd.isna(oa):
            return None
        # Convert odds to raw implied probabilities
        ph = 1 / oh
        pd_ = 1 / od
        pa = 1 / oa
        total = ph + pd_ + pa
        # Normalise to remove bookmaker overround
        probs = [ph/total, pd_/total, pa/total]
        return rps(probs, actual_outcome)
    except:
        return None


# --- BUILD TRACKING TABLE ----------------------------------------------------

def build_tracking(matches, lookup, params):
    """
    Runs predictions for every match in the dataset and compares to actual.
    Only includes matches where both teams are in our ratings.
    """
    home_adv = params["home_advantage"]
    rho      = params["rho"]

    rows = []
    skipped = 0

    for _, match in matches.iterrows():
        home = match["HomeTeam"]
        away = match["AwayTeam"]

        result = predict_match(home, away, lookup, home_adv, rho)
        if result is None:
            skipped += 1
            continue

        hw_prob, d_prob, aw_prob, pred_h, pred_a = result

        actual_h = int(match["FTHG"])
        actual_a = int(match["FTAG"])
        actual_r = match["FTR"]  # H, D, or A

        # Determine actual outcome index for RPS
        outcome_idx = {"H": 0, "D": 1, "A": 2}.get(actual_r, None)
        if outcome_idx is None:
            skipped += 1
            continue

        # What did we predict as the result?
        if hw_prob > d_prob and hw_prob > aw_prob:
            pred_result = "H"
        elif d_prob > hw_prob and d_prob > aw_prob:
            pred_result = "D"
        else:
            pred_result = "A"

        model_rps  = rps([hw_prob, d_prob, aw_prob], outcome_idx)
        b365_rps_v = bet365_rps(match, outcome_idx)

        rows.append({
            "Date":        match["Date"].date(),
            "Season":      match.get("Season", ""),
            "HomeTeam":    home,
            "AwayTeam":    away,
            "Actual_H":    actual_h,
            "Actual_A":    actual_a,
            "Actual_R":    actual_r,
            "Pred_H":      pred_h,
            "Pred_A":      pred_a,
            "Pred_R":      pred_result,
            "P_HomeWin":   round(hw_prob, 4),
            "P_Draw":      round(d_prob, 4),
            "P_AwayWin":   round(aw_prob, 4),
            "Result_Correct": actual_r == pred_result,
            "Score_Correct":  (actual_h == pred_h) and (actual_a == pred_a),
            "Model_RPS":   round(model_rps, 4),
            "B365_RPS":    round(b365_rps_v, 4) if b365_rps_v else None,
        })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if skipped > 0:
        print("  [INFO] Skipped " + str(skipped) +
              " matches (teams not in ratings or missing result)")

    return df


# --- REPORTING ---------------------------------------------------------------

def season_summary(df, season_label):
    """Returns a dict of accuracy stats for a given season (or 'All')."""
    if season_label != "All":
        d = df[df["Season"] == season_label]
    else:
        d = df

    if len(d) == 0:
        return None

    result_acc = d["Result_Correct"].mean() * 100
    score_acc  = d["Score_Correct"].mean() * 100
    model_rps  = d["Model_RPS"].mean()
    b365       = d["B365_RPS"].dropna()
    b365_rps   = b365.mean() if len(b365) > 0 else None

    return {
        "season":      season_label,
        "matches":     len(d),
        "result_acc":  round(result_acc, 1),
        "score_acc":   round(score_acc, 1),
        "model_rps":   round(model_rps, 4),
        "b365_rps":    round(b365_rps, 4) if b365_rps else None,
    }


def print_report(df):
    print()
    print("  " + "=" * 66)
    print("  PREDICTION ACCURACY REPORT")
    print("  " + "=" * 66)

    # --- Per season summary ---
    print()
    print("  BY SEASON")
    print("  " + "-" * 66)
    print("  {:10s}  {:7s}  {:10s}  {:10s}  {:10s}  {:10s}".format(
        "Season", "Matches", "Result%", "Score%", "Model RPS", "Bet365 RPS"))
    print("  " + "-" * 66)

    for season in sorted(df["Season"].unique()):
        s = season_summary(df, season)
        if s:
            b365_str = str(s["b365_rps"]) if s["b365_rps"] else "  n/a"
            vs = ""
            if s["b365_rps"]:
                diff = s["model_rps"] - s["b365_rps"]
                vs = " (+" + str(round(diff,4)) + ")" if diff > 0 else " (" + str(round(diff,4)) + ")"
            print("  {:10s}  {:7d}  {:9.1f}%  {:9.1f}%  {:10.4f}  {:>10s}".format(
                s["season"], s["matches"], s["result_acc"],
                s["score_acc"], s["model_rps"], b365_str + vs))

    # --- Overall ---
    overall = season_summary(df, "All")
    print("  " + "-" * 66)
    b365_str = str(overall["b365_rps"]) if overall["b365_rps"] else "n/a"
    vs = ""
    if overall["b365_rps"]:
        diff = overall["model_rps"] - overall["b365_rps"]
        vs = " (+" + str(round(diff,4)) + ")" if diff > 0 else " (" + str(round(diff,4)) + ")"
    print("  {:10s}  {:7d}  {:9.1f}%  {:9.1f}%  {:10.4f}  {:>10s}".format(
        "OVERALL", overall["matches"], overall["result_acc"],
        overall["score_acc"], overall["model_rps"], b365_str + vs))

    # --- RPS explanation ---
    print()
    print("  RPS (Ranked Probability Score): lower = better.")
    print("  A score close to Bet365 means our model is competitive.")
    print("  Negative vs Bet365 = we're beating the bookmaker on this metric.")

    # --- Recent matches ---
    recent = df.tail(RECENT_GAMEWEEKS * 10)
    print()
    print("  LAST " + str(len(recent)) + " MATCHES")
    print("  " + "-" * 66)
    print("  {:12s}  {:22s}  {:22s}  {:6s}  {:6s}  {:6s}".format(
        "Date", "Home", "Away", "Actual", "Pred", "RPS"))
    print("  " + "-" * 66)

    for _, row in recent.iterrows():
        actual = str(int(row["Actual_H"])) + "-" + str(int(row["Actual_A"]))
        pred   = str(int(row["Pred_H"]))   + "-" + str(int(row["Pred_A"]))
        tick   = "[OK]" if row["Result_Correct"] else "    "
        score_tick = "[SC]" if row["Score_Correct"] else "    "
        print("  {:12s}  {:22s}  {:22s}  {:6s}  {:5s} {:4s}{:4s}  {:.4f}".format(
            str(row["Date"].date()),
            str(row["HomeTeam"])[:22],
            str(row["AwayTeam"])[:22],
            actual, pred, tick, score_tick,
            row["Model_RPS"]))

    print()
    print("  [OK] = result correct   [SC] = exact score correct")
    print("  " + "=" * 66)


def save_outputs(df, overall):
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

    # Full tracking history
    csv_path = OUTPUTS_FOLDER + "/tracking.csv"
    df.to_csv(csv_path, index=False)
    print("  [OK] Tracking data saved: " + csv_path)

    # Summary stats as JSON for easy reading
    seasons = {}
    for season in sorted(df["Season"].unique()):
        s = season_summary(df, season)
        if s:
            seasons[season] = s

    accuracy = {
        "overall": season_summary(df, "All"),
        "by_season": seasons,
    }
    json_path = OUTPUTS_FOLDER + "/accuracy.json"
    with open(json_path, "w") as f:
        json.dump(accuracy, f, indent=2)
    print("  [OK] Accuracy stats saved: " + json_path)


# --- MAIN --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  EPL Score Predictor -- Script 05: Track Results")
    print("=" * 60)

    print()
    print("[Step 1] Loading data...")
    matches, lookup, params = load_data()
    print("  [OK] " + str(len(matches)) + " matches loaded")
    print("  [OK] " + str(len(lookup)) + " teams in ratings")

    print()
    print("[Step 2] Generating predictions for all past matches...")
    tracking = build_tracking(matches, lookup, params)
    print("  [OK] " + str(len(tracking)) + " matches tracked")

    print()
    print("[Step 3] Accuracy report")
    print_report(tracking)

    print()
    print("[Step 4] Saving outputs...")
    overall = season_summary(tracking, "All")
    save_outputs(tracking, overall)

    print()
    print("=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
