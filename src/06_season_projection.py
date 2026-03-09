# =============================================================================
# SCRIPT 06 -- Season Points Projection
# =============================================================================
# WHAT THIS SCRIPT DOES:
#   Simulates the rest of the 2025-26 season thousands of times using our
#   Dixon-Coles match predictions, and projects where each team will finish.
#
# HOW TO RUN IT:
#   python3 src/06_season_projection.py
#
# WHAT YOU'LL GET:
#   - Projected final points table with ranges (10th-90th percentile)
#   - Title / top 4 / relegation probabilities for each team
#   - outputs/season_projection.csv
# =============================================================================

import pandas as pd
import numpy as np
from scipy.stats import poisson
import json
import os


# --- CONFIGURATION -----------------------------------------------------------

RAW_DATA_FOLDER  = "data/raw"
OUTPUTS_FOLDER   = "outputs"

CURRENT_SEASON   = "2025-26"
RAW_FILE         = RAW_DATA_FOLDER + "/epl_2025-26.csv"
N_SIMULATIONS    = 10000   # more = more accurate but slower. 10k takes ~30s


# --- LOAD DATA ---------------------------------------------------------------

def load_current_standings():
    """Calculates actual current points table from played matches."""
    df = pd.read_csv(RAW_FILE, encoding='utf-8-sig')
    df = df.dropna(subset=['FTHG', 'FTAG'])
    df['FTHG'] = df['FTHG'].astype(int)
    df['FTAG'] = df['FTAG'].astype(int)

    teams = sorted(df['HomeTeam'].dropna().unique())
    standings = {}

    for team in teams:
        home = df[df['HomeTeam'] == team]
        away = df[df['AwayTeam'] == team]
        pts = 0; gf = 0; ga = 0
        pts += (home['FTR'] == 'H').sum() * 3
        pts += (home['FTR'] == 'D').sum() * 1
        gf  += home['FTHG'].sum(); ga += home['FTAG'].sum()
        pts += (away['FTR'] == 'A').sum() * 3
        pts += (away['FTR'] == 'D').sum() * 1
        gf  += away['FTAG'].sum(); ga += away['FTHG'].sum()
        played = len(home) + len(away)
        standings[team] = {
            'pts': pts, 'played': played,
            'gf': gf, 'ga': ga, 'gd': gf - ga
        }

    return standings, df


def get_remaining_fixtures(df, teams):
    """Returns list of (home, away) tuples not yet played."""
    played = set(zip(df['HomeTeam'], df['AwayTeam']))
    remaining = []
    for home in teams:
        for away in teams:
            if home != away and (home, away) not in played:
                remaining.append((home, away))
    return remaining


def load_ratings_and_params():
    ratings = pd.read_csv(OUTPUTS_FOLDER + "/team_ratings.csv")
    with open(OUTPUTS_FOLDER + "/model_params.json") as f:
        params = json.load(f)
    lookup = {row['Team']: {'Attack': row['Attack'], 'Defence': row['Defence']}
              for _, row in ratings.iterrows()}
    return lookup, params


# --- MATCH SIMULATION --------------------------------------------------------

def get_match_probs(home, away, lookup, home_adv, rho):
    """
    Returns (p_home_win, p_draw, p_away_win) for a single match.
    Uses same Dixon-Coles model as Scripts 04/05.
    """
    mu_h = np.exp(lookup[home]['Attack'] + lookup[away]['Defence'] + home_adv)
    mu_a = np.exp(lookup[away]['Attack'] + lookup[home]['Defence'])

    MAX_G = 9
    matrix = np.zeros((MAX_G, MAX_G))
    for hg in range(MAX_G):
        for ag in range(MAX_G):
            p = poisson.pmf(hg, mu_h) * poisson.pmf(ag, mu_a)
            if   hg == 0 and ag == 0: p *= max(1 - mu_h * mu_a * rho, 0)
            elif hg == 1 and ag == 0: p *= max(1 + mu_a * rho, 0)
            elif hg == 0 and ag == 1: p *= max(1 + mu_h * rho, 0)
            elif hg == 1 and ag == 1: p *= max(1 - rho, 0)
            matrix[hg, ag] = p

    return (float(np.sum(np.tril(matrix, -1))),  # home win
            float(np.sum(np.diag(matrix))),        # draw
            float(np.sum(np.triu(matrix, 1))))     # away win


def precompute_fixture_probs(remaining, lookup, home_adv, rho):
    """
    Pre-calculates win/draw/loss probs for every remaining fixture.
    Doing this once upfront makes the Monte Carlo loop much faster.
    """
    fixture_probs = {}
    for home, away in remaining:
        hw, d, aw = get_match_probs(home, away, lookup, home_adv, rho)
        fixture_probs[(home, away)] = (hw, d, aw)
    return fixture_probs


# --- MONTE CARLO SIMULATION --------------------------------------------------

def run_simulations(standings, remaining, fixture_probs, n_sims):
    """
    Simulates the rest of the season N times.
    Each simulation randomly assigns results based on match probabilities,
    then calculates final points for each team.

    Returns a dict: team -> array of final points across all simulations.
    """
    teams = list(standings.keys())
    current_pts = np.array([standings[t]['pts'] for t in teams], dtype=float)
    team_idx    = {t: i for i, t in enumerate(teams)}

    # Pre-build fixture arrays for fast vectorised sampling
    home_teams  = [home for home, away in remaining]
    away_teams  = [away for home, away in remaining]
    probs_array = np.array([fixture_probs[(h, a)] for h, a in remaining])

    # probs_array shape: (n_fixtures, 3) -- [p_home, p_draw, p_away]
    n_fixtures = len(remaining)

    # All simulations at once using vectorised random sampling
    # np.random.choice equivalent for categorical: use cumulative probs
    print("  Running " + str(n_sims) + " simulations x " +
          str(n_fixtures) + " remaining fixtures...")

    # Shape: (n_sims, n_fixtures) -- random [0,1) for each sim x fixture
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    rand = rng.random((n_sims, n_fixtures))

    # Cumulative probabilities for each fixture
    cum_home = probs_array[:, 0]                          # shape (n_fixtures,)
    cum_draw = probs_array[:, 0] + probs_array[:, 1]      # shape (n_fixtures,)

    # Result: 0=home win, 1=draw, 2=away win  shape: (n_sims, n_fixtures)
    results = np.where(rand < cum_home, 0,
              np.where(rand < cum_draw, 1, 2))

    # Points earned: home=3pts if result=0, away=3pts if result=2, 1pt each if draw
    home_pts = np.where(results == 0, 3, np.where(results == 1, 1, 0))
    away_pts = np.where(results == 2, 3, np.where(results == 1, 1, 0))

    # Start from current real points, then add simulated future points
    # Shape: (n_sims, n_teams)
    sim_pts = np.tile(current_pts, (n_sims, 1)).copy()

    home_idx = np.array([team_idx[t] for t in home_teams])
    away_idx = np.array([team_idx[t] for t in away_teams])

    # Add points fixture by fixture (sim_pts ends up as TOTAL final points)
    for f_idx in range(n_fixtures):
        sim_pts[:, home_idx[f_idx]] += home_pts[:, f_idx]
        sim_pts[:, away_idx[f_idx]] += away_pts[:, f_idx]

    return teams, sim_pts


# --- RESULTS ANALYSIS --------------------------------------------------------

def analyse_simulations(teams, sim_pts, standings):
    """
    From the matrix of simulated final points, calculate:
    - Mean, median, 10th/90th percentile projected points
    - Probability of finishing in each position band
    """
    n_sims = sim_pts.shape[0]
    n_teams = len(teams)

    # For each simulation, rank teams by points (ties broken randomly)
    # Shape: (n_sims, n_teams) -- rank 1=best
    noise = np.random.default_rng(99).random(sim_pts.shape) * 0.001
    ranked = np.argsort(np.argsort(-(sim_pts + noise), axis=1), axis=1) + 1

    results = []
    for i, team in enumerate(teams):
        pts_dist    = sim_pts[:, i]   # total final points across all sims
        rank_dist   = ranked[:, i]

        results.append({
            'Team':           team,
            'Current_Pts':    int(standings[team]['pts']),
            'Played':         int(standings[team]['played']),
            'Remaining':      38 - int(standings[team]['played']),
            # These are TOTAL projected final points (current + future)
            'Proj_Min':       int(np.percentile(pts_dist, 10)),
            'Proj_Med':       int(np.median(pts_dist)),
            'Proj_Mean':      round(float(np.mean(pts_dist)), 1),
            'Proj_Max':       int(np.percentile(pts_dist, 90)),
            'Title_Pct':      round(float((rank_dist == 1).mean() * 100), 1),
            'Top4_Pct':       round(float((rank_dist <= 4).mean() * 100), 1),
            'Top6_Pct':       round(float((rank_dist <= 6).mean() * 100), 1),
            'Relegation_Pct': round(float((rank_dist >= 18).mean() * 100), 1),
        })

    df = pd.DataFrame(results)
    # Sort by total projected final points (median), highest first
    df = df.sort_values('Proj_Med', ascending=False).reset_index(drop=True)
    df.index += 1
    return df


# --- PRINT AND SAVE ----------------------------------------------------------

def print_projection(df):
    print()
    print("  " + "=" * 88)
    print("  2025-26 PREMIER LEAGUE SEASON PROJECTION  (" +
          str(N_SIMULATIONS) + " simulations)")
    print("  " + "=" * 88)
    print()
    print("  {:3s}  {:18s}  {:5s}  {:4s}  {:28s}  {:7s}  {:6s}  {:6s}  {:8s}".format(
        "Pos", "Team", "Now", "Left",
        "Final Points (P10 -- Med -- P90)",
        "Title%", "Top4%", "Top6%", "Relegate"))
    print("  " + "-" * 90)

    for pos, row in df.iterrows():
        bar_str = str(row['Proj_Min']) + " -- " + str(row['Proj_Med']) + \
                  " -- " + str(row['Proj_Max'])
        print("  {:3d}  {:18s}  {:5d}  {:4d}  {:28s}  {:6.1f}%  {:5.1f}%  {:5.1f}%  {:7.1f}%".format(
            pos,
            str(row['Team'])[:18],
            row['Current_Pts'],
            row['Remaining'],
            bar_str,
            row['Title_Pct'],
            row['Top4_Pct'],
            row['Top6_Pct'],
            row['Relegation_Pct'],
        ))

    print("  " + "-" * 90)
    print()
    print("  Final Points = P10 (pessimistic) -- Median -- P90 (optimistic) across 10k simulations")
    print("  Top 4 = Champions League  |  Top 6 = Europe  |  Bottom 3 = Relegated")
    print("  " + "=" * 90)


def save_outputs(df):
    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
    path = OUTPUTS_FOLDER + "/season_projection.csv"
    df.to_csv(path)
    print("  [OK] Saved: " + path)


# --- MAIN --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  EPL Score Predictor -- Script 06: Season Projection")
    print("=" * 60)

    print()
    print("[Step 1] Loading current standings...")
    standings, df_played = load_current_standings()
    teams = list(standings.keys())
    total_pts = sum(s['pts'] for s in standings.values())
    print("  [OK] " + str(len(teams)) + " teams, " +
          str(sum(s['played'] for s in standings.values()) // 2) +
          " matches played")
    for team, s in sorted(standings.items(),
                          key=lambda x: -x[1]['pts'])[:5]:
        print("    " + team + ": " + str(s['pts']) + " pts")
    print("    ...")

    print()
    print("[Step 2] Loading ratings...")
    lookup, params = load_ratings_and_params()
    home_adv = params['home_advantage']
    rho      = params['rho']
    print("  [OK] Ratings loaded for " + str(len(lookup)) + " teams")

    print()
    print("[Step 3] Finding remaining fixtures...")
    remaining = get_remaining_fixtures(df_played, teams)
    print("  [OK] " + str(len(remaining)) + " fixtures remaining")

    print()
    print("[Step 4] Pre-computing match probabilities...")
    fixture_probs = precompute_fixture_probs(remaining, lookup, home_adv, rho)
    print("  [OK] Probabilities computed for all " +
          str(len(fixture_probs)) + " fixtures")

    print()
    print("[Step 5] Running Monte Carlo simulations...")
    teams_list, sim_pts = run_simulations(
        standings, remaining, fixture_probs, N_SIMULATIONS)
    print("  [OK] Done")

    print()
    print("[Step 6] Analysing results...")
    projection = analyse_simulations(teams_list, sim_pts, standings)

    print_projection(projection)

    print("[Step 7] Saving...")
    save_outputs(projection)

    print()
    print("=" * 60)
    print("  ALL DONE!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
