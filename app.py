import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go

st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache_data
def load_data():
    return pd.read_csv('player_advanced_stats.csv')


df = load_data()

# Filter by position
position_list = df['Pos'].unique()
selected_position = st.sidebar.selectbox('Select a position:', position_list)
filtered_df = df[df['Pos'] == selected_position]
player_list = filtered_df['Player'].unique()
selected_player = st.sidebar.selectbox('Select a player:', player_list)

# user chooses metric
metric_list = ['PER', 'TS%', 'FTr', 'ORB%', 'DRB%', 'AST%', 'STL%', 'BLK%', 'TOV%',
               'USG%', 'OWS', 'DWS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']
selected_metric = st.sidebar.selectbox('Select a metric:', metric_list)

# show data accordingly
player_data = filtered_df[filtered_df['Player'] == selected_player]

# explain metrics
st.sidebar.write("""
**Metrics:**
- **PER**: Player Efficiency Rating
- **TS%**: True Shooting Percentage
- **FTr**: Free Throw Rate
- **ORB%**: Offensive Rebound Percentage
- **DRB%**: Defensive Rebound Percentage
- **AST%**: Assist Percentage
- **STL%**: Steal Percentage
- **BLK%**: Block Percentage
- **TOV%**: Turnover Percentage
- **USG%**: Usage Percentage
- **OWS**: Offensive Win Shares
- **DWS**: Defensive Win Shares
- **WS/48**: Win Shares per 48 Minutes
- **OBPM**: Offensive Box Plus-Minus
- **DBPM**: Defensive Box Plus-Minus
- **BPM**: Box Plus-Minus
- **VORP**: Value Over Replacement Player
""")

# title
st.title(f"{selected_player}: {selected_metric}")
st.divider()

# seasons as earlier column
if 'Season' in player_data.columns:
    cols = player_data.columns.tolist()
    cols.remove('Season')
    cols.insert(3, 'Season')
    player_data = player_data[cols]

# Display data
st.header(f"Player: {selected_player} ({selected_position})")
st.write(player_data)

# ===========================================================================
# NEW FEATURE: Value/Age — headline callout stat
# ===========================================================================
#
# Design:
#   - value_metric summarizes overall production (defaults to VORP, an
#     all-in-one wins-above-replacement style stat).
#   - Baseline = league-average value at that specific age. Value/Age of 1.0
#     means "exactly as productive as the typical player at that age."
#   - Youth bonus: the same production is worth MORE the younger the player
#     is, since a young player producing at an average (or above-average)
#     level has more years of development/team-control runway ahead than an
#     older player putting up the same numbers — the same logic real
#     trade-value charts use to price production + youth together.
#
#   value_age = (player_value / league_avg_value_at_age) * youth_multiplier
#   youth_multiplier = 1 + max(0, peak_age - age) * youth_bonus_per_year
def compute_value_age(df, value_metric='VORP', peak_age=27, youth_bonus_per_year=0.03):
    data = df.dropna(subset=[value_metric, 'Age']).copy()
    league_avg_by_age = data.groupby('Age')[value_metric].mean()

    def _row_value_age(row):
        avg_at_age = league_avg_by_age.get(row['Age'], np.nan)
        if pd.isna(avg_at_age) or avg_at_age == 0:
            return np.nan
        ratio = row[value_metric] / avg_at_age
        youth_multiplier = 1 + max(0, peak_age - row['Age']) * youth_bonus_per_year
        return round(ratio * youth_multiplier, 3)

    data['Value/Age'] = data.apply(_row_value_age, axis=1)
    return data


def render_value_age(df, selected_player, selected_season=None, value_metric='VORP',
                      peak_age=27, youth_bonus_per_year=0.03):
    """
    Highlights Value/Age as a standout callout metric (not a chart) - a big
    number plus a one-line interpretation.
    """
    data = compute_value_age(df, value_metric, peak_age, youth_bonus_per_year)
    p_data = data[data['Player'] == selected_player].sort_values('Age')

    if p_data.empty:
        st.write("No Value/Age data available for this player/metric combination.")
        return

    if selected_season is not None and selected_season in p_data['Season'].values:
        row = p_data[p_data['Season'] == selected_season].iloc[0]
    else:
        row = p_data.iloc[-1]

    va = row['Value/Age']
    delta = round(va - 1.0, 3)

    st.markdown("### 🌟 Value/Age")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            label=f"{selected_player} · Age {int(row['Age'])} · {row['Season']}",
            value=f"{va:.2f}",
            delta=f"{delta:+.2f} vs. league avg for age"
        )
    with col2:
        if va > 1.0:
            verdict = "above what's expected for a player this age"
        elif va < 1.0:
            verdict = "below what's expected for a player this age"
        else:
            verdict = "exactly average for a player this age"
        st.write(
            f"A Value/Age of **{va:.2f}** means this season's production "
            f"(measured by **{value_metric}**) is **{verdict}**, after factoring in a "
            f"youth bonus for players younger than the assumed peak age ({peak_age})."
        )
    st.caption(
        "1.0 = average league player at that exact age. Younger players earn a bonus for "
        "matching that bar, since the same production carries more development/team-control "
        "runway ahead of it."
    )


render_value_age(df, selected_player, value_metric='VORP')
st.divider()

# ===========================================================================
# NEW FEATURE: Percentile radar chart
# ===========================================================================
def compute_percentile_radar(df, player_row, position, metric_list):
    position_df = df[df['Pos'] == position]
    percentiles = []
    for metric in metric_list:
        lower_is_better = {'TOV%'}
        values = position_df[metric].dropna()
        player_value = player_row[metric].values[0]
        pct = stats.percentileofscore(values, player_value)
        if metric in lower_is_better:
            pct = 100 - pct
        percentiles.append(round(pct, 1))
    return percentiles


def render_percentile_radar(df, player_row, selected_player, selected_position, metric_list):
    percentiles = compute_percentile_radar(df, player_row, selected_position, metric_list)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=percentiles,
        theta=metric_list,
        fill='toself',
        name=selected_player,
        line=dict(color='royalblue')
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title=f"{selected_player} — Percentile Profile vs. {selected_position}s"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each axis shows where the player ranks (0-100th percentile) among all "
               f"{selected_position}s in the dataset for that metric.")


render_percentile_radar(df, player_data, selected_player, selected_position, metric_list)
st.divider()

# ===========================================================================
# NEW FEATURE: Most similar players finder
# ===========================================================================
def find_similar_players(df, selected_player, metric_list, n=5):
    data = df.dropna(subset=metric_list).copy()
    z = data[metric_list].apply(lambda col: (col - col.mean()) / col.std())

    if selected_player not in data['Player'].values:
        return pd.DataFrame()

    player_idx = data.index[data['Player'] == selected_player][0]
    player_vec = z.loc[[player_idx]]

    nn = NearestNeighbors(n_neighbors=min(n + 1, len(z)))
    nn.fit(z.values)
    distances, indices = nn.kneighbors(player_vec.values)

    result_idx = data.index[indices[0]]
    results = data.loc[result_idx, ['Player', 'Season', 'Pos', 'Age'] + metric_list].copy()
    results['similarity_distance'] = distances[0]
    results = results[results['Player'] != selected_player]
    return results.head(n)


def render_similarity_finder(df, selected_player, metric_list):
    similar = find_similar_players(df, selected_player, metric_list, n=5)
    st.subheader(f"Players Most Similar to {selected_player}")
    if similar.empty:
        st.write("Not enough data to compute similarity.")
    else:
        st.dataframe(similar.reset_index(drop=True))
        st.caption("Similarity is based on standardized distance across all listed metrics — "
                   "smaller distance means a closer statistical match.")


render_similarity_finder(df, selected_player, metric_list)
st.divider()

# ===========================================================================
# ORIGINAL GRAPHS
# ===========================================================================
# Calculate average
average_metric = filtered_df.groupby('Age')[selected_metric].mean().reset_index()

# Calculate median
median_metric = filtered_df.groupby('Age')[selected_metric].median().reset_index()

# player's vs Age and average vs Age and median vs Age
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(player_data['Age'], player_data[selected_metric], marker='o', linestyle='-',
        color='blue', label=f'{selected_player}')
ax.plot(average_metric['Age'], average_metric[selected_metric], marker='o', linestyle='--',
        color='orange', label='Average')
ax.plot(median_metric['Age'], median_metric[selected_metric], marker='o', linestyle='--',
        color='green', label='Median')
ax.set_title(f'{selected_player} vs Average and Median - {selected_metric} vs Age')
ax.set_xlabel('Age')
ax.set_ylabel(selected_metric)
ax.legend()
ax.grid(True)

st.pyplot(fig)

# note about survivorship bias
st.info("Note: the player metrics sometimes spike in the later years because typically, "
        "only star players continue to play at an older age. The sample size is therefore "
        "smaller, including only very good players, and this leads to a higher rating than "
        "what otherwise may be expected with an increase in age.")
