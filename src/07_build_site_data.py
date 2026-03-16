#!/usr/bin/env python3
# =============================================================================
# SCRIPT 07 -- Build Site Data
# =============================================================================
# WHAT THIS SCRIPT DOES:
#   Reads all model outputs (from scripts 01-06) and rebuilds
#   website/data.js — the single file that powers all pages of the site.
#
# HOW TO RUN IT (every gameweek, after running scripts 01-06):
#   python3 src/07_build_site_data.py
#
# WHAT YOU NEED TO UPDATE MANUALLY EACH WEEK:
#   Add the new gameweek's fixtures to GW_SCHEDULE below, then run.
#   Once a GW has been played, remove it (or leave it — it won't appear
#   in the site because those matches will already be in the results CSV).
#   Everything else (standings, projections, form, ratings) is automatic.
#
# WHAT YOU'LL GET:
#   website/data.js  -- rebuilt with all fresh data
# =============================================================================

import pandas as pd
import numpy as np
import json
import os
from scipy.stats import poisson


# =============================================================================
# >>> UPDATE THIS SECTION EACH GAMEWEEK <<<
# =============================================================================
#
# List ALL remaining fixtures grouped by gameweek.
# Format: ("Home Team", "Away Team")
# Team names must match exactly what's in team_ratings.csv.
# Find fixtures at: bbc.co.uk/sport/football/premier-league/fixtures
#
# HOW TO USE:
#   - Add the new GW's fixtures before running script 07
#   - The LOWEST GW number in this dict becomes the "next GW" on the site
#   - Fixtures are shown on team pages in GW order (next match first)
#   - You don't need to remove old GWs — played matches are filtered out

GW_SCHEDULE = {
    32: [   # 20-22 Mar (Arsenal & Wolves blank — FA Cup; Man City vs Crystal Palace PP)
        ("Bournemouth",    "Man United"),
        ("Brighton",       "Liverpool"),
        ("Fulham",         "Burnley"),
        ("Everton",        "Chelsea"),
        ("Leeds",          "Brentford"),
        ("Newcastle",      "Sunderland"),
        ("Aston Villa",    "West Ham"),
        ("Tottenham",      "Nott'm Forest"),
    ],
    33: [   # 10-13 Apr
        ("West Ham",       "Wolves"),
        ("Arsenal",        "Bournemouth"),
        ("Brentford",      "Everton"),
        ("Burnley",        "Brighton"),
        ("Crystal Palace", "Newcastle"),
        ("Nott'm Forest",  "Aston Villa"),
        ("Liverpool",      "Fulham"),
        ("Sunderland",     "Tottenham"),
        ("Chelsea",        "Man City"),
        ("Man United",     "Leeds"),
    ],
    34: [   # 18-20 Apr
        ("Brentford",      "Fulham"),
        ("Aston Villa",    "Sunderland"),
        ("Leeds",          "Wolves"),
        ("Newcastle",      "Bournemouth"),
        ("Nott'm Forest",  "Burnley"),
        ("Tottenham",      "Brighton"),
        ("Chelsea",        "Man United"),
        ("Everton",        "Liverpool"),
        ("Man City",       "Arsenal"),
        ("Crystal Palace", "West Ham"),
    ],
    35: [   # 24-27 Apr
        ("Sunderland",     "Nott'm Forest"),
        ("Fulham",         "Aston Villa"),
        ("Bournemouth",    "Leeds"),
        ("Liverpool",      "Crystal Palace"),
        ("West Ham",       "Everton"),
        ("Wolves",         "Tottenham"),
        ("Arsenal",        "Newcastle"),
        ("Burnley",        "Man City"),
        ("Brighton",       "Chelsea"),
        ("Man United",     "Brentford"),
    ],
    36: [   # 2 May
        ("Bournemouth",    "Crystal Palace"),
        ("Arsenal",        "Fulham"),
        ("Aston Villa",    "Tottenham"),
        ("Brentford",      "West Ham"),
        ("Chelsea",        "Nott'm Forest"),
        ("Everton",        "Man City"),
        ("Leeds",          "Burnley"),
        ("Man United",     "Liverpool"),
        ("Newcastle",      "Brighton"),
        ("Wolves",         "Sunderland"),
    ],
    37: [   # 9 May
        ("Brighton",       "Wolves"),
        ("Burnley",        "Aston Villa"),
        ("Crystal Palace", "Everton"),
        ("Fulham",         "Bournemouth"),
        ("Liverpool",      "Chelsea"),
        ("Man City",       "Brentford"),
        ("Nott'm Forest",  "Newcastle"),
        ("Sunderland",     "Man United"),
        ("Tottenham",      "Leeds"),
        ("West Ham",       "Arsenal"),
    ],
    38: [   # 17 May
        ("Bournemouth",    "Man City"),
        ("Arsenal",        "Burnley"),
        ("Aston Villa",    "Liverpool"),
        ("Brentford",      "Crystal Palace"),
        ("Chelsea",        "Tottenham"),
        ("Everton",        "Sunderland"),
        ("Leeds",          "Brighton"),
        ("Man United",     "Nott'm Forest"),
        ("Newcastle",      "West Ham"),
        ("Wolves",         "Fulham"),
    ],
    39: [   # 24 May — Final Day
        ("Brighton",       "Man United"),
        ("Burnley",        "Wolves"),
        ("Crystal Palace", "Arsenal"),
        ("Fulham",         "Newcastle"),
        ("Liverpool",      "Brentford"),
        ("Man City",       "Aston Villa"),
        ("Nott'm Forest",  "Bournemouth"),
        ("Sunderland",     "Chelsea"),
        ("Tottenham",      "Everton"),
        ("West Ham",       "Leeds"),
    ],
}

# =============================================================================
# OPTIONAL CONFIG — only change these if something structural changes
# =============================================================================

FEATURE_TEAM = "Arsenal"    # team shown on arsenalHist (backward compat)
OUTPUTS_DIR  = "outputs"
RAW_FILE     = "data/raw/epl_2025-26.csv"
SITE_JS      = "website/data.js"
N_SIM        = 10000        # simulations for all-team histograms


# =============================================================================
# HELPERS — STANDINGS + FORM
# =============================================================================

def load_results():
    """Load played matches from the raw season CSV."""
    df = pd.read_csv(RAW_FILE, encoding='utf-8-sig')
    df = df.dropna(subset=['FTHG', 'FTAG'])
    df['FTHG'] = df['FTHG'].astype(int)
    df['FTAG']  = df['FTAG'].astype(int)
    return df


def current_gw(df):
    """Infer the last completed GW from max games played by any team."""
    home = df.groupby('HomeTeam').size()
    away = df.groupby('AwayTeam').size()
    played = home.add(away, fill_value=0)
    return int(played.max())


def get_form(df, team, n=5):
    """Last N results for a team, most recent first, as W/D/L strings."""
    home = df[df['HomeTeam'] == team][['Date', 'FTR']].copy()
    home['result'] = home['FTR'].map({'H': 'W', 'D': 'D', 'A': 'L'})

    away = df[df['AwayTeam'] == team][['Date', 'FTR']].copy()
    away['result'] = away['FTR'].map({'A': 'W', 'D': 'D', 'H': 'L'})

    combined = pd.concat([home[['Date', 'result']], away[['Date', 'result']]])
    combined['Date'] = pd.to_datetime(combined['Date'], dayfirst=True)
    combined = combined.sort_values('Date', ascending=False)

    return combined['result'].head(n).tolist()


# =============================================================================
# HELPERS — SIMULATION (for feature-team points histogram)
# =============================================================================

def load_ratings_and_params():
    ratings_df = pd.read_csv(f"{OUTPUTS_DIR}/team_ratings.csv")
    with open(f"{OUTPUTS_DIR}/model_params.json") as f:
        params = json.load(f)
    lookup = {row['Team']: {'Attack': row['Attack'], 'Defence': row['Defence']}
              for _, row in ratings_df.iterrows()}
    return lookup, params['home_advantage'], params['rho']


def get_match_probs(home, away, lookup, home_adv, rho):
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
    hw = float(np.sum(np.tril(matrix, -1)))
    d  = float(np.sum(np.diag(matrix)))
    aw = float(np.sum(np.triu(matrix, 1)))
    return hw, d, aw


def run_all_teams_simulation(df_played, lookup, home_adv, rho, n_sim):
    """
    Runs n_sim simulations of the remaining season in one vectorised pass.
    Returns a dict of {team_name: np.array of final points (length n_sim)}.
    This is maximally efficient — the full sim is only executed once regardless
    of how many teams we want histogram data for.
    """
    print(f"  Running {n_sim:,} simulations for all teams...")

    # Current standings
    teams = sorted(df_played['HomeTeam'].dropna().unique().tolist())
    played_set = set(zip(df_played['HomeTeam'], df_played['AwayTeam']))

    standings = {}
    for t in teams:
        h = df_played[df_played['HomeTeam'] == t]
        a = df_played[df_played['AwayTeam'] == t]
        pts = (h['FTR'] == 'H').sum() * 3 + (h['FTR'] == 'D').sum()
        pts += (a['FTR'] == 'A').sum() * 3 + (a['FTR'] == 'D').sum()
        standings[t] = int(pts)

    remaining = [(h, a) for h in teams for a in teams
                 if h != a and (h, a) not in played_set]

    # Pre-compute probabilities
    probs = {}
    for h, a in remaining:
        if h in lookup and a in lookup:
            probs[(h, a)] = get_match_probs(h, a, lookup, home_adv, rho)

    remaining = [(h, a) for h, a in remaining if (h, a) in probs]

    team_idx   = {t: i for i, t in enumerate(teams)}
    curr_pts   = np.array([standings[t] for t in teams], dtype=float)
    home_idx   = np.array([team_idx[h] for h, a in remaining])
    away_idx   = np.array([team_idx[a] for h, a in remaining])
    probs_arr  = np.array([probs[(h, a)] for h, a in remaining])

    rng  = np.random.default_rng(42)
    rand = rng.random((n_sim, len(remaining)))

    cum_h = probs_arr[:, 0]
    cum_d = probs_arr[:, 0] + probs_arr[:, 1]
    res   = np.where(rand < cum_h, 0, np.where(rand < cum_d, 1, 2))

    h_pts = np.where(res == 0, 3, np.where(res == 1, 1, 0))
    a_pts = np.where(res == 2, 3, np.where(res == 1, 1, 0))

    sim_pts = np.tile(curr_pts, (n_sim, 1)).copy()
    for fi in range(len(remaining)):
        sim_pts[:, home_idx[fi]] += h_pts[:, fi]
        sim_pts[:, away_idx[fi]] += a_pts[:, fi]

    # Return dict of team → pts array
    return {t: sim_pts[:, i] for t, i in team_idx.items()}


def build_histogram(pts_array, bins=20):
    """Bin simulated points into histogram entries for the chart."""
    counts = {}
    for p in pts_array:
        counts[int(p)] = counts.get(int(p), 0) + 1
    lo = min(counts)
    hi = max(counts)
    return [{"pts": p, "count": counts.get(p, 0)} for p in range(lo, hi + 1)]


# =============================================================================
# HELPERS — NEXT GW LOOKUP
# =============================================================================

def build_gw_section(fixtures_config, all_remaining):
    """
    For each (home, away) in fixtures_config, find the matching prediction
    in the remaining_fixtures list and build the gw fixture object.
    """
    # Index remaining by (HomeTeam, AwayTeam)
    lookup = {(f['HomeTeam'], f['AwayTeam']): f for f in all_remaining}

    gw_fixtures = []
    for home, away in fixtures_config:
        f = lookup.get((home, away))
        if f is None:
            print(f"  [WARN] Fixture not found in remaining: {home} vs {away} — skipping")
            continue

        # Parse top 5 scores from string like "2-0(12.1%)  1-0(10.3%) ..."
        top5 = []
        for part in f['Top_5_Scores'].split():
            part = part.strip()
            if not part:
                continue
            m_score = part.split('(')
            if len(m_score) == 2:
                score = m_score[0]
                prob  = float(m_score[1].rstrip('%)'))
                top5.append({"score": score, "prob": prob})

        # Build 5x5 score matrix (0-0 through 4-4)
        sm_raw = f['Score_Matrix']
        sm5 = {}
        for row in range(5):
            for col in range(5):
                key = f"{row}-{col}"
                sm5[key] = round(sm_raw.get(key, 0.0), 2)

        gw_fixtures.append({
            "home":     home,
            "away":     away,
            "xgHome":   round(f['xG_Home'], 2),
            "xgAway":   round(f['xG_Away'], 2),
            "predHome": int(f['Pred_Home']),
            "predAway": int(f['Pred_Away']),
            "predProb": round(f['Pred_Prob'], 1),
            "homeWin":  round(f['Home_Win_Pct'], 1),
            "draw":     round(f['Draw_Pct'], 1),
            "awayWin":  round(f['Away_Win_Pct'], 1),
            "top5":     top5,
            "scoreMatrix": sm5,
        })

    return gw_fixtures


# =============================================================================
# HELPERS — TEAM FIXTURES
# =============================================================================

def build_fixture_gw_lookup(gw_schedule):
    """
    Build a lookup from (home, away) -> gw_number using the schedule.
    Also indexes (away, home) so we can look up from either team's perspective.
    Fixtures not in the schedule get a high GW number so they sort last.
    """
    lookup = {}
    for gw_num, fixtures in gw_schedule.items():
        for home, away in fixtures:
            lookup[(home, away)] = gw_num
            lookup[(away, home)] = gw_num  # needed for away team's perspective
    return lookup


def build_team_fixtures(detail_df, gw_schedule):
    """
    Build the teamFixtures dict: team → list of remaining fixture objects,
    sorted by gameweek (next fixture first).
    """
    gw_lookup = build_fixture_gw_lookup(gw_schedule)
    UNKNOWN_GW = 99  # fixtures not yet scheduled sort to the end

    team_fixtures = {}
    for _, row in detail_df.iterrows():
        team = row['Team']
        opponent = row['Opponent']
        if team not in team_fixtures:
            team_fixtures[team] = []

        sm_raw = json.loads(row['Score_Matrix']) if isinstance(row['Score_Matrix'], str) else row['Score_Matrix']

        # Score matrix 5x5 only
        sm5 = {}
        for r in range(5):
            for c in range(5):
                key = f"{r}-{c}"
                sm5[key] = round(sm_raw.get(key, 0.0), 2)

        # Determine GW number for sorting (try both orientations)
        gw_num = gw_lookup.get((team, opponent),
                  gw_lookup.get((opponent, team), UNKNOWN_GW))

        team_fixtures[team].append({
            "gw":          gw_num,
            "ha":          row['HA'],
            "opponent":    opponent,
            "xgFor":       round(float(row['xG_For']), 2),
            "xgAgainst":   round(float(row['xG_Against']), 2),
            "predFor":     int(row['Pred_For']),
            "predAgainst": int(row['Pred_Against']),
            "winPct":      round(float(row['Win_Pct']), 1),
            "drawPct":     round(float(row['Draw_Pct']), 1),
            "lossPct":     round(float(row['Loss_Pct']), 1),
            "expPts":      round(float(row['Exp_Pts']), 2),
            "scoreMatrix": sm5,
        })

    # Sort each team's fixtures by GW number
    for team in team_fixtures:
        team_fixtures[team].sort(key=lambda x: x['gw'])

    return team_fixtures


# =============================================================================
# HELPERS — PROJECTION TABLE
# =============================================================================

def build_projection(proj_df, df_played):
    """
    Merge season projection with form from raw results data.
    Returns list of dicts sorted by projected median (already done in CSV).
    """
    teams = proj_df['Team'].tolist()
    projection = []
    for pos, (_, row) in enumerate(proj_df.iterrows(), start=1):
        team = row['Team']
        projection.append({
            "team":       team,
            "pos":        pos,
            "currentPts": int(row['Current_Pts']),
            "played":     int(row['Played']),
            "remaining":  int(row['Remaining']),
            "projMin":    int(row['Proj_Min']),
            "projMed":    int(row['Proj_Med']),
            "projMax":    int(row['Proj_Max']),
            "titlePct":   float(row['Title_Pct']),
            "top4Pct":    float(row['Top4_Pct']),
            "top6Pct":    float(row['Top6_Pct']),
            "relegPct":   float(row['Relegation_Pct']),
            "form":       get_form(df_played, team),
        })
    return projection


# =============================================================================
# HELPERS — RATINGS
# =============================================================================

def build_ratings(ratings_df):
    ratings = []
    for _, row in ratings_df.iterrows():
        ratings.append({
            "team":       row['Team'],
            "attack":     round(float(row['Attack']), 4),
            "defence":    round(float(row['Defence']), 4),
            "expScored":  round(float(row['Exp_Scored']), 3),
            "expConceded":round(float(row['Exp_Conceded']), 3),
        })
    return ratings


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  EPL Score Predictor -- Script 07: Build Site Data")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Load raw results + infer current GW
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading raw results...")
    df_played = load_results()
    gw = current_gw(df_played)
    next_gw = gw + 1
    print(f"  [OK] {len(df_played)} played matches. Last completed GW: {gw}. Building data for GW{next_gw}.")

    # ------------------------------------------------------------------
    # Step 2 — Load season projection
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading season projection...")
    proj_df = pd.read_csv(f"{OUTPUTS_DIR}/season_projection.csv", index_col=0)
    projection = build_projection(proj_df, df_played)
    print(f"  [OK] {len(projection)} teams loaded.")

    # ------------------------------------------------------------------
    # Step 3 — Load remaining fixtures (filter out already-played matches)
    # ------------------------------------------------------------------
    print("\n[Step 3] Loading remaining fixtures...")
    with open(f"{OUTPUTS_DIR}/remaining_fixtures.json") as f:
        all_remaining = json.load(f)
    played_pairs = set(zip(df_played['HomeTeam'], df_played['AwayTeam']))
    before = len(all_remaining)
    all_remaining = [f for f in all_remaining
                     if (f['HomeTeam'], f['AwayTeam']) not in played_pairs]
    print(f"  [OK] {len(all_remaining)} remaining fixtures ({before - len(all_remaining)} already played, filtered out).")

    # ------------------------------------------------------------------
    # Step 4 — Build next GW section from schedule
    # ------------------------------------------------------------------
    # Derive the next GW: lowest key in GW_SCHEDULE that has fixtures
    scheduled_gws = sorted([gw for gw, fx in GW_SCHEDULE.items() if fx])
    if scheduled_gws:
        next_gw = scheduled_gws[0]
        next_gw_fixtures_list = GW_SCHEDULE[next_gw]
    else:
        print("  [WARN] GW_SCHEDULE is empty — no fixtures page will be built.")
        next_gw_fixtures_list = []

    print(f"\n[Step 4] Building GW{next_gw} fixture section ({len(next_gw_fixtures_list)} fixtures)...")
    gw_fixtures = build_gw_section(next_gw_fixtures_list, all_remaining)
    print(f"  [OK] {len(gw_fixtures)} fixtures matched.")

    # ------------------------------------------------------------------
    # Step 5 — Build team fixtures (sorted by GW schedule, played filtered out)
    # ------------------------------------------------------------------
    print("\n[Step 5] Building team fixtures...")
    detail_df = pd.read_csv(f"{OUTPUTS_DIR}/team_fixture_detail.csv")
    # Filter out fixtures already played (Team=home, Opponent=away or vice versa via HA)
    def is_played(row):
        if row['HA'] == 'H':
            return (row['Team'], row['Opponent']) in played_pairs
        else:
            return (row['Opponent'], row['Team']) in played_pairs
    before_detail = len(detail_df)
    detail_df = detail_df[~detail_df.apply(is_played, axis=1)]
    print(f"  [OK] {len(team_fixtures := build_team_fixtures(detail_df, GW_SCHEDULE))} teams ({before_detail - len(detail_df)} played rows filtered out, sorted by schedule).")

    # ------------------------------------------------------------------
    # Step 6 — Build ratings
    # ------------------------------------------------------------------
    print("\n[Step 6] Loading ratings...")
    ratings_df = pd.read_csv(f"{OUTPUTS_DIR}/team_ratings.csv")
    ratings = build_ratings(ratings_df)
    print(f"  [OK] {len(ratings)} teams.")

    # ------------------------------------------------------------------
    # Step 7 — Simulate all-team points distributions (single pass)
    # ------------------------------------------------------------------
    print(f"\n[Step 7] Simulating all-team points distributions ({N_SIM:,} sims)...")
    lookup, home_adv, rho = load_ratings_and_params()
    all_pts = run_all_teams_simulation(df_played, lookup, home_adv, rho, N_SIM)

    team_hist       = {}
    team_hist_stats = {}
    for t, pts_arr in all_pts.items():
        team_hist[t]       = build_histogram(pts_arr)
        team_hist_stats[t] = {
            "median": int(np.median(pts_arr)),
            "p10":    int(np.percentile(pts_arr, 10)),
            "p90":    int(np.percentile(pts_arr, 90)),
        }

    # Backward-compat keys for existing team-arsenal.html
    hist       = team_hist.get(FEATURE_TEAM, [])
    hist_stats = team_hist_stats.get(FEATURE_TEAM, {"median": 0, "p10": 0, "p90": 0})
    print(f"  [OK] {len(team_hist)} teams. "
          f"{FEATURE_TEAM} median: {hist_stats['median']} pts "
          f"(P10: {hist_stats['p10']} — P90: {hist_stats['p90']})")

    # ------------------------------------------------------------------
    # Step 8 — Assemble and write data.js
    # ------------------------------------------------------------------
    print("\n[Step 8] Writing data.js...")

    site_data = {
        "lastUpdated":      f"GW{gw}",
        "season":           "2025/26",
        "projection":       projection,
        f"gw{next_gw}":     gw_fixtures,
        "teamFixtures":     team_fixtures,
        "ratings":          ratings,
        "teamHist":         team_hist,
        "teamHistStats":    team_hist_stats,
        # backward-compat for existing team-arsenal.html
        "arsenalHist":      hist,
        "arsenalHistStats": hist_stats,
    }

    js_content = (
        "// EPL Predictions — site data (auto-generated)\n"
        "const SITE_DATA = " +
        json.dumps(site_data, indent=2) +
        ";\n"
    )

    os.makedirs(os.path.dirname(SITE_JS), exist_ok=True)
    with open(SITE_JS, 'w') as f:
        f.write(js_content)

    size_kb = os.path.getsize(SITE_JS) / 1024
    print(f"  [OK] Written: {SITE_JS}  ({size_kb:.0f} KB)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  ALL DONE! Website data updated for GW" + str(next_gw))
    print("=" * 60)
    print()
    print("  NOTE: The Fixtures page reads SITE_DATA.gw" + str(next_gw))
    print("  If your fixtures.html still references a different GW key,")
    print("  update the JS variable at the top of fixtures.html:")
    print(f"    const GW_KEY = 'gw{next_gw}';")
    print()


if __name__ == "__main__":
    main()
