import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objects as go


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

# Display data (hide_index drops the leading row-number column)
st.header(f"Player: {selected_player} ({selected_position})")
st.dataframe(player_data, hide_index=True)

# ===========================================================================
# NEW FEATURE: PlayerValue — headline callout stat
# ===========================================================================
#
# Design:
#   - value_metric summarizes overall production (defaults to VORP, an
#     all-in-one wins-above-replacement style stat).
#   - Baseline = league-average value at that specific age. PlayerValue of 1.0
#     means "exactly as productive as the typical player at that age."
#   - Youth bonus: the same production is worth MORE the younger the player
#     is, since a young player producing at an average (or above-average)
#     level has more years of development/team-control runway ahead than an
#     older player putting up the same numbers — the same logic real
#     trade-value charts use to price production + youth together.
#
#   player_value = (player_value / league_avg_value_at_age) * youth_multiplier
#   youth_multiplier = 1 + max(0, peak_age - age) * youth_bonus_per_year
def compute_player_value(df, value_metric='VORP', peak_age=27, youth_bonus_per_year=0.03):
    data = df.dropna(subset=[value_metric, 'Age']).copy()
    league_avg_by_age = data.groupby('Age')[value_metric].mean()

    def _row_player_value(row):
        avg_at_age = league_avg_by_age.get(row['Age'], np.nan)
        if pd.isna(avg_at_age) or avg_at_age == 0:
            return np.nan
        ratio = row[value_metric] / avg_at_age
        youth_multiplier = 1 + max(0, peak_age - row['Age']) * youth_bonus_per_year
        return round(ratio * youth_multiplier, 3)

    data['PlayerValue'] = data.apply(_row_player_value, axis=1)
    return data


def render_player_value(df, selected_player, selected_season=None, value_metric='VORP',
                      peak_age=27, youth_bonus_per_year=0.03):
    """
    Highlights PlayerValue as a standout callout metric (not a chart) - a big
    number plus a one-line interpretation.
    """
    data = compute_player_value(df, value_metric, peak_age, youth_bonus_per_year)
    p_data = data[data['Player'] == selected_player].sort_values('Age')

    if p_data.empty:
        st.write("No PlayerValue data available for this player/metric combination.")
        return

    if selected_season is not None and selected_season in p_data['Season'].values:
        row = p_data[p_data['Season'] == selected_season].iloc[0]
    else:
        row = p_data.iloc[-1]

    va = row['PlayerValue']
    delta = round(va - 1.0, 3)

    st.markdown("### 🌟 PlayerValue")
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
            f"A PlayerValue of **{va:.2f}** means this season's production "
            f"(measured by **{value_metric}**) is **{verdict}**, after factoring in a "
            f"youth bonus for players younger than the assumed peak age ({peak_age})."
        )
    st.caption(
        "1.0 = average league player at that exact age. Younger players earn a bonus for "
        "matching that bar, since the same production carries more development/team-control "
        "runway ahead of it."
    )


render_player_value(df, selected_player, value_metric='VORP')
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
        metric_value = player_row[metric].values[0]
        pct = stats.percentileofscore(values, metric_value)
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
# NEW FEATURE: Interactive shot-zone hot map
# ===========================================================================
#
# IMPORTANT DATA NOTE:
#   The underlying CSV only has season-level *advanced* box-score stats
#   (PER, TS%, VORP, etc.) — it does not contain shot-by-shot location or
#   defender-distance data (that lives in NBA tracking data, which isn't
#   available here). So the FG% shown per zone/contest-level below is a
#   MODEL, not a lookup of real shot logs:
#     1. Each zone/contest cell starts from a realistic league-average FG%
#        (roughly matching publicly known closest-defender splits, e.g.
#        shots at the rim are made far more often than contested threes).
#     2. That baseline is then scaled up or down by the selected player's
#        True Shooting % that season relative to the league/position
#        average TS% that same season — a real, computed number from the
#        dataset — so a more efficient scorer shows a higher probability
#        across the board and vice versa.
#   Treat the numbers as an illustrative estimate of shooting difficulty
#   and player efficiency, not as verified play-by-play shooting splits.

ZONE_LEAGUE_AVG_FG = {
    'Restricted Area (Layup)':   {'Wide Open': 0.68, 'Slight Contest': 0.62, 'Heavy Contest': 0.52},
    'Free Throw / Short Range':  {'Wide Open': 0.46, 'Slight Contest': 0.41, 'Heavy Contest': 0.33},
    'Mid-Range':                 {'Wide Open': 0.44, 'Slight Contest': 0.39, 'Heavy Contest': 0.31},
    'Left Corner 3':             {'Wide Open': 0.40, 'Slight Contest': 0.35, 'Heavy Contest': 0.28},
    'Right Corner 3':            {'Wide Open': 0.40, 'Slight Contest': 0.35, 'Heavy Contest': 0.28},
    'Above the Break 3':         {'Wide Open': 0.37, 'Slight Contest': 0.33, 'Heavy Contest': 0.26},
}

# Center point (court coordinates, hoop at x=0, y=5.25) used both to place
# the clickable marker and to anchor the shaded zone patch.
ZONE_MARKER_XY = {
    'Restricted Area (Layup)':  (0, 4),
    'Free Throw / Short Range': (0, 14),
    'Mid-Range':                (14, 16),
    'Left Corner 3':            (-23, 6),
    'Right Corner 3':           (23, 6),
    'Above the Break 3':        (0, 32),
}


def player_efficiency_factor(df, player_row):
    """Ratio of the player's TS% to the league/position average TS% for that
    same season, clipped so the model stays in a sane range."""
    ts = player_row['TS%'].values[0]
    season = player_row['Season'].values[0]
    pos = player_row['Pos'].values[0]
    if pd.isna(ts):
        return 1.0
    baseline = df[(df['Season'] == season) & (df['Pos'] == pos)]['TS%'].mean()
    if pd.isna(baseline) or baseline == 0:
        return 1.0
    factor = ts / baseline
    return float(np.clip(factor, 0.6, 1.6))


def build_court_figure(zone_fg_for_color):
    """Draws a half-court and places one clickable marker per zone, colored
    by that zone's 'Slight Contest' make probability for the selected player."""
    fig = go.Figure()

    court_shapes = [
        # Court boundary (half court)
        dict(type='rect', x0=-25, y0=0, x1=25, y1=47, line=dict(color='white', width=2)),
        # Paint / lane
        dict(type='rect', x0=-8, y0=0, x1=8, y1=19, line=dict(color='white', width=2)),
        # Free throw circle (top half solid via full circle, good enough visually)
        dict(type='circle', x0=-6, y0=13, x1=6, y1=25, line=dict(color='white', width=2)),
        # Restricted area arc
        dict(type='circle', x0=-4, y0=1.25, x1=4, y1=9.25, line=dict(color='white', width=2)),
        # Backboard
        dict(type='line', x0=-3, y0=4, x1=3, y1=4, line=dict(color='white', width=3)),
        # Half-court line + center circle
        dict(type='line', x0=-25, y0=47, x1=25, y1=47, line=dict(color='white', width=2)),
        dict(type='circle', x0=-6, y0=41, x1=6, y1=53, line=dict(color='white', width=2)),
        # Three point corners (straight sections)
        dict(type='line', x0=-22, y0=0, x1=-22, y1=14.2, line=dict(color='white', width=2)),
        dict(type='line', x0=22, y0=0, x1=22, y1=14.2, line=dict(color='white', width=2)),
    ]

    # Three point arc (approximate with a path)
    import math
    arc_x, arc_y = [], []
    for deg in range(0, 181):
        rad = math.radians(deg)
        x = 23.75 * math.cos(rad)
        y = 5.25 + 23.75 * math.sin(rad)
        if y <= 47:
            arc_x.append(x)
            arc_y.append(y)
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines',
                              line=dict(color='white', width=2), hoverinfo='skip',
                              showlegend=False))

    # Hoop
    court_shapes.append(dict(type='circle', x0=-0.75, y0=4.5, x1=0.75, y1=6,
                              line=dict(color='orange', width=2)))

    fig.update_layout(shapes=court_shapes)

    zones = list(ZONE_MARKER_XY.keys())
    xs = [ZONE_MARKER_XY[z][0] for z in zones]
    ys = [ZONE_MARKER_XY[z][1] for z in zones]
    colors = [zone_fg_for_color[z] for z in zones]

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers+text',
        text=zones, textposition='top center', textfont=dict(color='white', size=10),
        marker=dict(size=32, color=colors, colorscale='RdYlGn', cmin=0.2, cmax=0.7,
                    line=dict(color='black', width=1),
                    colorbar=dict(title='Slight<br>Contest<br>FG%')),
        customdata=zones,
        hovertemplate='%{customdata}<br>Click to see the breakdown<extra></extra>',
        name='Zones'
    ))

    fig.update_layout(
        plot_bgcolor='#1a5f3f', paper_bgcolor='#1a5f3f',
        xaxis=dict(visible=False, range=[-27, 27]),
        yaxis=dict(visible=False, range=[-2, 49], scaleanchor='x', scaleratio=1),
        height=560, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=f"{selected_player} — Click a zone for shot probabilities",
                    font=dict(color='white'))
    )
    return fig


st.markdown("### 🏀 Shot-Zone Probability Explorer")
st.caption(
    "Estimated make probability by zone and by how contested the shot is. These are "
    "modeled from the player's True Shooting % relative to the league/position average "
    "that season, applied to realistic zone/contest baselines — not real shot-log data "
    "(see code comments for the full explanation)."
)

if player_data.empty:
    st.write("No data available to build the shot chart for this player.")
else:
    eff_factor = player_efficiency_factor(df, player_data)

    zone_player_fg = {}
    for zone, contest_levels in ZONE_LEAGUE_AVG_FG.items():
        zone_player_fg[zone] = {
            level: float(np.clip(base * eff_factor, 0.10, 0.85))
            for level, base in contest_levels.items()
        }

    # color markers by each zone's "Slight Contest" number for the selected player
    zone_color_lookup = {z: zone_player_fg[z]['Slight Contest'] for z in ZONE_LEAGUE_AVG_FG}

    court_fig = build_court_figure(zone_color_lookup)

    selection = st.plotly_chart(
        court_fig, use_container_width=True,
        on_select='rerun', selection_mode='points', key='shot_zone_court'
    )

    clicked_zone = None
    if selection and selection.get('selection', {}).get('points'):
        point = selection['selection']['points'][0]
        clicked_zone = point.get('customdata')

    if clicked_zone is None:
        st.info("Click any zone marker on the court above to see the make-probability breakdown.")
    else:
        st.markdown(f"#### {clicked_zone}")
        contest_levels = ['Wide Open', 'Slight Contest', 'Heavy Contest']
        player_vals = [zone_player_fg[clicked_zone][c] for c in contest_levels]
        league_vals = [ZONE_LEAGUE_AVG_FG[clicked_zone][c] for c in contest_levels]

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(name=selected_player, x=contest_levels,
                                  y=[v * 100 for v in player_vals], marker_color='royalblue'))
        bar_fig.add_trace(go.Bar(name='League Average', x=contest_levels,
                                  y=[v * 100 for v in league_vals], marker_color='lightgray'))
        bar_fig.update_layout(barmode='group', yaxis_title='Make Probability (%)',
                               yaxis=dict(range=[0, 90]), height=380,
                               title=f"{clicked_zone}: {selected_player} vs. League Average")
        st.plotly_chart(bar_fig, use_container_width=True)

        cols = st.columns(3)
        for c, col in zip(contest_levels, cols):
            with col:
                st.metric(
                    label=c,
                    value=f"{zone_player_fg[clicked_zone][c] * 100:.1f}%",
                    delta=f"{(zone_player_fg[clicked_zone][c] - ZONE_LEAGUE_AVG_FG[clicked_zone][c]) * 100:+.1f} pts vs league"
                )

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
