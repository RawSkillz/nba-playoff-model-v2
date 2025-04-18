
import streamlit as st
import pandas as pd
import re
import difflib
import requests

@st.cache_data
def load_data():
    return pd.read_csv("final_playoff_projection_model_with_positions.csv")

@st.cache_data
def load_dvp():
    return pd.read_csv("hybrid_positional_dvp_ranked_fixed.csv")

@st.cache_data
def load_h2h():
    try:
        return pd.read_csv("player_vs_opponent_summary_h2h.csv")
    except Exception:
        return pd.DataFrame()

df = load_data()
dvp_ranked = load_dvp()
h2h_df = load_h2h()

df["StarScore"] = (df["PTS_adj"] * 1.5) + df["MPG"]
star_players = df.loc[df.groupby("Team")["StarScore"].idxmax()][["Team", "Player"]]
star_player_by_team = dict(zip(star_players["Team"], star_players["Player"]))

def get_defender_penalty(pos, opp_team):
    opp_team = team_abbr_reverse.get(opp_team, opp_team)
    if not opp_team or opp_team not in dvp_ranked["Team"].values:
        return 0.0
    position_relevance = {
        "G": ["PG", "SG"], "F": ["SF", "PF"], "C": ["C"],
        "F-C": ["PF", "C"], "C-F": ["C", "PF"], "G-F": ["SG", "SF"], "F-G": ["SF", "PF"],
        "PG": ["PG"], "SG": ["SG"], "SF": ["SF"], "PF": ["PF"]
    }
    relevant_positions = position_relevance.get(pos, [])
    if not relevant_positions:
        return 0.0
    best_rank = 99
    for rpos in relevant_positions:
        for stat in ["PTS", "REB", "AST"]:
            col = f"{rpos}_{stat}"
            if col in dvp_ranked.columns:
                try:
                    rank = int(dvp_ranked.loc[dvp_ranked["Team"] == opp_team, col].values[0])
                    best_rank = min(best_rank, rank)
                except:
                    continue
    if best_rank <= 5:
        return -0.05
    elif best_rank <= 10:
        return -0.03
    elif best_rank <= 15:
        return -0.01
    return 0.0

def get_h2h_bonus(player_name, opp_code, stat_type):
    try:
        if h2h_df.empty or not opp_code:
            return 0.0
        filtered = h2h_df[
            (h2h_df["Player"] == player_name) &
            (h2h_df["Opponent"] == opp_code) &
            (h2h_df["Stat"] == stat_type)
        ]
        if not filtered.empty:
            return float(filtered.iloc[0]["Bonus"])
    except:
        pass
    return 0.0

team_abbr = {
    "Atlanta Hawks": "ATL", "Orlando Magic": "ORL", "Memphis Grizzlies": "MEM", "Golden State Warriors": "GSW",
    "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "New York Knicks": "NYK", "Philadelphia 76ers": "PHI",
    "Toronto Raptors": "TOR", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Detroit Pistons": "DET",
    "Indiana Pacers": "IND", "Milwaukee Bucks": "MIL", "Charlotte Hornets": "CHA", "Miami Heat": "MIA",
    "Washington Wizards": "WAS", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Houston Rockets": "HOU",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Phoenix Suns": "PHX", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "New Orleans Pelicans": "NOP", "Minnesota Timberwolves": "MIN", "Oklahoma City Thunder": "OKC",
    "Portland Trail Blazers": "POR", "Utah Jazz": "UTA"
}
team_abbr_reverse = {v: k for k, v in team_abbr.items()}

position_map = {
    "G": ["PG", "SG"], "F": ["SF", "PF"], "C": ["C"], "F-C": ["PF", "C"], "C-F": ["C", "PF"],
    "G-F": ["SG", "SF"], "F-G": ["SF", "PF"], "PG": ["PG"], "SG": ["SG"], "SF": ["SF"], "PF": ["PF"]
}

slate_df = pd.DataFrame()
try:
    url = "https://raw.githubusercontent.com/rawskillz/nba-playoff-model-v2/main/slate.txt"
    response = requests.get(url)
    response.raise_for_status()
    lines = [line.strip() for line in response.text.splitlines() if line.strip()]
    games, i = [], 0
    while i < len(lines) - 10:
        try:
            t1, t2 = lines[i], lines[i+1]
            if t1 not in team_abbr or t2 not in team_abbr:
                i += 1
                continue
            spread1 = float(lines[i+2].replace("+", ""))
            spread2 = float(lines[i+7].replace("+", ""))
            total = None
            for j in range(i, i+14):
                if 'O ' in lines[j] or 'U ' in lines[j]:
                    total = float(re.findall(r"\d+\.?\d*", lines[j])[0])
                    break
            games.append({
                "Team1": team_abbr[t1], "Team2": team_abbr[t2],
                "Spread1": spread1, "Spread2": spread2, "Total": total
            })
            i += 14
        except:
            i += 1
    slate_df = pd.DataFrame(games)
    st.sidebar.success("✅ Slate loaded from GitHub!")
    st.sidebar.dataframe(slate_df)
except:
    st.sidebar.warning("⚠️ Could not load slate.txt from GitHub.")


st.markdown("## 📊 Top 10 Stat Movers (PTS, REB, AST)")


def compute_full_projection(row):
    team = row["Team"]
    pos = row["Position"]
    opp, opp_full, spread, total = None, None, 0, 224
    if not slate_df.empty:
        matchup = slate_df[(slate_df["Team1"] == team) | (slate_df["Team2"] == team)]
        if not matchup.empty:
            m = matchup.iloc[0]
            opp = m["Team2"] if m["Team1"] == team else m["Team1"]
            opp_full = team_abbr_reverse.get(opp, "")
            spread = m["Spread1"] if m["Team1"] == team else m["Spread2"]
            total = m["Total"]

    ts = row["TS%"]
    mpg = row["MPG"]
    pace_factor = total / 224
    blowout_penalty = 0.90 if abs(spread) >= 12 and mpg < 28 else 0.95 if abs(spread) >= 10 else 1.0

    pts, reb, ast = row["PTS_adj"], row["REB_adj"], row["AST_adj"]
    usage_proxy = pts / mpg if mpg > 0 else 0

    bonuses = {"PTS": 0, "REB": 0, "AST": 0}
    for stat in ["PTS", "REB", "AST"]:
        ranks = []
        for mapped_pos in position_map.get(pos, [pos]):
            col = f"{mapped_pos}_{stat}"
            if col in dvp_ranked.columns and opp_full in dvp_ranked["Team"].values:
                rank = int(dvp_ranked.loc[dvp_ranked["Team"] == opp_full, col].values[0])
                ranks.append(rank)
        if ranks:
            avg_rank = sum(ranks) / len(ranks)
            if avg_rank <= 5:
                bonuses[stat] = 0.06
            elif avg_rank <= 10:
                bonuses[stat] = 0.03
            elif avg_rank <= 20:
                bonuses[stat] = 0.0
            elif avg_rank <= 25:
                bonuses[stat] = -0.03
            else:
                bonuses[stat] = -0.06

    def apply(stat, base, is_pts, stat_label):
        usage = 1.025 if mpg >= 30 else 0.985 if mpg >= 20 else 0.93
        dvp_bonus = bonuses[stat]
        val = base * (1 + dvp_bonus)
        if is_pts:
            val *= min(ts / 0.57, 1.07)

        h2h_bonus = get_h2h_bonus(row["Player"], opp, stat_label)
        defender_bonus = get_defender_penalty(pos, opp) if row["Player"] == star_player_by_team.get(team, "") else 0.0

        total_penalty = dvp_bonus + defender_bonus
        if total_penalty < -0.07:
            total_penalty = -0.07

        adjusted_defender_bonus = total_penalty - dvp_bonus
        val *= (1 + h2h_bonus)
        val *= (1 + adjusted_defender_bonus)

        return val * pace_factor * blowout_penalty * usage

    return pd.Series({
        "PTS_base": pts, "PTS_proj": apply("PTS", pts, True, "Points"),
        "REB_base": reb, "REB_proj": apply("REB", reb, False, "Rebounds"),
        "AST_base": ast, "AST_proj": apply("AST", ast, False, "Assists")
    })

    team = row["Team"]
    pos = row["Position"]
    opp, opp_full, spread, total = None, None, 0, 224
    if not slate_df.empty:
        matchup = slate_df[(slate_df["Team1"] == team) | (slate_df["Team2"] == team)]
        if not matchup.empty:
            m = matchup.iloc[0]
            opp = m["Team2"] if m["Team1"] == team else m["Team1"]
            opp_full = team_abbr_reverse.get(opp, "")
            spread = m["Spread1"] if m["Team1"] == team else m["Spread2"]
            total = m["Total"]

    ts = row["TS%"]
    mpg = row["MPG"]
    pace_factor = total / 224
    blowout_penalty = 0.90 if abs(spread) >= 12 and mpg < 28 else 0.95 if abs(spread) >= 10 else 1.0
    usage_proxy = row["PTS_adj"] / mpg if mpg > 0 else 0

    bonuses = {"PTS": 0, "REB": 0, "AST": 0}
    for stat in ["PTS", "REB", "AST"]:
        ranks = []
        for mapped_pos in position_map.get(pos, [pos]):
            col = f"{mapped_pos}_{stat}"
            if col in dvp_ranked.columns and opp_full in dvp_ranked["Team"].values:
                rank = int(dvp_ranked.loc[dvp_ranked["Team"] == opp_full, col].values[0])
                ranks.append(rank)
        if ranks:
            avg_rank = sum(ranks) / len(ranks)
            if avg_rank <= 5:
                bonuses[stat] = 0.06
            elif avg_rank <= 10:
                bonuses[stat] = 0.03
            elif avg_rank <= 20:
                bonuses[stat] = 0.0
            elif avg_rank <= 25:
                bonuses[stat] = -0.03
            else:
                bonuses[stat] = -0.06

    def final(stat, base, is_pts, label):
        usage = 1.025 if mpg >= 30 else 0.985 if mpg >= 20 else 0.93
        val = base * (1 + bonuses[stat])
        if is_pts:
            val *= min(ts / 0.57, 1.07)
        h2h_bonus = get_h2h_bonus(row["Player"], opp, label)
        defender_bonus = get_defender_penalty(pos, opp) if row["Player"] == star_player_by_team.get(team, "") else 0.0
        total_penalty = min(dvp_bonus := bonuses[stat] + defender_bonus, -0.07)
        val *= (1 + h2h_bonus)
        val *= (1 + (total_penalty - bonuses[stat]))
        return val * pace_factor * blowout_penalty * usage

    pts_proj = final("PTS", row["PTS_adj"], True, "Points")
    reb_proj = final("REB", row["REB_adj"], False, "Rebounds")
    ast_proj = final("AST", row["AST_adj"], False, "Assists")
    return pd.Series({
        "PTS_base": row["PTS_adj"], "PTS_proj": pts_proj,
        "REB_base": row["REB_adj"], "REB_proj": reb_proj,
        "AST_base": row["AST_adj"], "AST_proj": ast_proj
    })

# Filter only players in games on the slate
active_teams = set(slate_df["Team1"]).union(set(slate_df["Team2"]))
active_df = df[df["Team"].isin(active_teams)].copy()
projections = active_df.apply(compute_full_projection, axis=1)
active_df = pd.concat([active_df, projections], axis=1)

movers = []
for stat in ["PTS", "REB", "AST"]:
    active_df[f"{stat}_diff"] = (active_df[f"{stat}_proj"] - active_df[f"{stat}_base"]) / active_df[f"{stat}_base"]
    movers.append(active_df[["Player", "Team", f"{stat}_base", f"{stat}_proj", f"{stat}_diff"]]
        .rename(columns={f"{stat}_base": "Base", f"{stat}_proj": "Final", f"{stat}_diff": "Diff"})
        .assign(Stat=stat)
    )

top_movers_all = pd.concat(movers)
top_movers_all = top_movers_all.merge(df[["Player", "MPG"]], on="Player", how="left")
top_movers_all = top_movers_all[top_movers_all["Base"] >= 5]
top_movers_all = top_movers_all[top_movers_all["MPG"] >= 13]
top_movers_all["Diff"] = (top_movers_all["Final"] - top_movers_all["Base"]) / top_movers_all["Base"]

top_10_over = top_movers_all.sort_values("Diff", ascending=False).head(10).copy()
top_10_under = top_movers_all[
    (top_movers_all["Diff"] >= -0.13) & (top_movers_all["Diff"] < 0)
].sort_values("Diff", ascending=True).head(10).copy()


top_10_over["Diff"] = top_10_over["Diff"].map(lambda x: f"+{x:.1%}")
top_10_under["Diff"] = top_10_under["Diff"].map(lambda x: f"{x:.1%}")
for df_ in [top_10_over, top_10_under]:
    df_["Base"] = df_["Base"].round(2)
    df_["Final"] = df_["Final"].round(2)





with st.expander("📊 Top 10 Stat Movers (PTS, REB, AST)", expanded=False):
    def df_to_html(df):
        rows = ""
        for _, row in df.iterrows():
            color = "limegreen" if row["Diff"].startswith("+") else "crimson"
            rows += f"<tr style='color:{color}'><td>{row['Player']}</td><td>{row['Team']}</td><td>{row['Stat']}</td><td>{row['Base']}</td><td>{row['Final']}</td><td>{row['Diff']}</td></tr>"
        return f"<table><thead><tr><th>Player</th><th>Team</th><th>Stat</th><th>Base</th><th>Final</th><th>Diff</th></tr></thead><tbody>{rows}</tbody></table>"

    st.markdown("### 🔺 Positive Movers", unsafe_allow_html=True)
    st.markdown(df_to_html(top_10_over), unsafe_allow_html=True)

    st.markdown("### 🔻 Negative Movers", unsafe_allow_html=True)
    st.markdown(df_to_html(top_10_under), unsafe_allow_html=True)




query_params = st.query_params
default_player = st.query_params.get("player", "")
default_stat = st.query_params.get("stat", "Points")

st.title("🔎 NBA Player Projection Lookup")
player_query = st.text_input("Search Player", placeholder="e.g. Trae Young", value=default_player)

stat_options = ["Points", "Rebounds", "Assists", "PR", "PA", "RA", "PRA"]
stat_index = stat_options.index(default_stat) if default_stat in stat_options else 0
stat_type = st.radio("Select Stat", stat_options, horizontal=True, index=stat_index)

if player_query:
    player_names = df["Player"].tolist()
    best_match = difflib.get_close_matches(player_query, player_names, n=1, cutoff=0.6)
    matches = df[df["Player"] == best_match[0]] if best_match else pd.DataFrame()
    if matches.empty:
        st.warning("No player found.")
    else:
        row = matches.iloc[0]
        team = row["Team"]
        pos = row["Position"]
        opp, opp_full, spread, total = None, None, 0, 224
        if not slate_df.empty:
            matchup = slate_df[(slate_df["Team1"] == team) | (slate_df["Team2"] == team)]
            if not matchup.empty:
                m = matchup.iloc[0]
                opp = m["Team2"] if m["Team1"] == team else m["Team1"]
                opp_full = team_abbr_reverse.get(opp, "")
                spread = m["Spread1"] if m["Team1"] == team else m["Spread2"]
                total = m["Total"]

        ts = row["TS%"]
        mpg = row["MPG"]
        pace_factor = total / 224
        blowout_penalty = 0.90 if abs(spread) >= 12 and mpg < 28 else 0.95 if abs(spread) >= 10 else 1.0

        pts, reb, ast = row["PTS_adj"], row["REB_adj"], row["AST_adj"]
        usage_proxy = pts / mpg if mpg > 0 else 0
        bonuses = {"PTS": 0, "REB": 0, "AST": 0}
        dvp_info = {}

        for stat in ["PTS", "REB", "AST"]:
            ranks = []
            for mapped_pos in position_map.get(pos, [pos]):
                col = f"{mapped_pos}_{stat}"
                if col in dvp_ranked.columns and opp_full in dvp_ranked["Team"].values:
                    rank = int(dvp_ranked.loc[dvp_ranked["Team"] == opp_full, col].values[0])
                    ranks.append(rank)
            if ranks:
                avg_rank = sum(ranks) / len(ranks)
                dvp_info[stat] = (f"avg_{stat}", round(avg_rank))
                if avg_rank <= 5:
                    bonuses[stat] = 0.06
                elif avg_rank <= 10:
                    bonuses[stat] = 0.03
                elif avg_rank <= 20:
                    bonuses[stat] = 0.0
                elif avg_rank <= 25:
                    bonuses[stat] = -0.03
                else:
                    bonuses[stat] = -0.06

        def apply(stat, base, is_pts, stat_label):
            usage = 1.0
            if mpg >= 30:
                usage = 1.025
            elif mpg >= 20:
                usage = 0.985
            else:
                usage = 0.93

            dvp_bonus = bonuses[stat]
            val = base * (1 + dvp_bonus)
            if is_pts:
                val *= min(ts / 0.57, 1.07)

            h2h_bonus = get_h2h_bonus(row["Player"], opp, stat_label)

            defender_bonus = get_defender_penalty(pos, opp) if row["Player"] == star_player_by_team.get(team, "") else 0.0

            total_penalty = dvp_bonus + defender_bonus
            if total_penalty < -0.07:
                total_penalty = -0.07

            adjusted_defender_bonus = total_penalty - dvp_bonus
            val *= (1 + h2h_bonus)
            val *= (1 + adjusted_defender_bonus)

            return val * pace_factor * blowout_penalty * usage, h2h_bonus, defender_bonus

        projections = {}
        h2h_applied = {}
        defender_applied = {}

        projections["Points"], h2h_applied["Points"], defender_applied["Points"] = apply("PTS", pts, True, "Points")
        projections["Rebounds"], h2h_applied["Rebounds"], defender_applied["Rebounds"] = apply("REB", reb, False, "Rebounds")
        projections["Assists"], h2h_applied["Assists"], defender_applied["Assists"] = apply("AST", ast, False, "Assists")
        projections["PR"] = projections["Points"] + projections["Rebounds"]
        projections["PA"] = projections["Points"] + projections["Assists"]
        projections["RA"] = projections["Rebounds"] + projections["Assists"]
        projections["PRA"] = projections["Points"] + projections["Rebounds"] + projections["Assists"]

        st.metric(label=f"{stat_type} Projection", value=round(projections[stat_type], 2))
        st.caption(f"{row['Player']} ({team} - {pos})")
        if opp:
            st.caption(f"🆚 Opponent: {opp} | Spread: {spread} | Total: {total}")

        st.markdown("### 🧠 Why this projection?")
        if stat_type in ["Points", "Rebounds", "Assists"]:
            key = {"Points": "PTS", "Rebounds": "REB", "Assists": "AST"}[stat_type]
            base = pts if key == "PTS" else reb if key == "REB" else ast
            ts_mult = min(ts / 0.57, 1.07) if key == "PTS" else 1.0
            dvp_bonus = bonuses[key]
            pace = pace_factor
            blowout = blowout_penalty
            h2h_perc = h2h_applied[stat_type]
            defender_bonus = defender_applied[stat_type]
            final = round(projections[stat_type], 2)

            explanation = f"Base: {round(base,2)} → TS: x{round(ts_mult,2)} → DvP: +{round(dvp_bonus*100,1)}% → Pace: x{round(pace,2)}"
            if h2h_perc != 0:
                explanation += f" → H2H: {('+' if h2h_perc>0 else '')}{round(h2h_perc*100,1)}%"
            if defender_bonus != 0:
                explanation += f" → Star Defender Penalty: {round(defender_bonus*100,1)}%"
            if blowout < 1.0:
                explanation += f" → Blowout Penalty x{blowout}"
            explanation += f" → **{final} final**"
            st.write(explanation)
            st.caption(f"⚡ Usage Proxy (PTS/MPG): {round(usage_proxy, 2)}")
            if key in dvp_info:
                col, rank = dvp_info[key]
                st.caption(f"📊 DvP Rank vs {col}: {rank} of 30")
