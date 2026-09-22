import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import plotly.graph_objects as go


@st.cache_data
def load_data():
    return pd.read_csv('player_advanced_stats.csv')


@st.cache_data
def load_zone_data():
    """Real shot data backing the Shot-Zone Probability Explorer:
    - player_zone_shooting.csv: real per-player, per-season, per-zone FGA/FGM,
      aggregated from ~2.8M actual shot attempts (2005-06 through 2018-19),
      sourced from stats.nba.com shot chart detail.
    - league_zone_contest_fg_2014_15.csv: real league-wide FG% by zone AND by
      closest-defender distance (Wide Open / Slight Contest / Heavy Contest),
      sourced from the NBA's 2014-15 SportVU shot-log tracking data (the only
      season this kind of defender-distance data is publicly available).
    - player_zone_contest_fg_2014_15.csv: the same defender-distance splits,
      but per player, for players with at least 10 shots in that zone during
      the 2014-15 season specifically.
    """
    zone_shooting = pd.read_csv('player_zone_shooting.csv')
    league_contest = pd.read_csv('league_zone_contest_fg_2014_15.csv')
    player_contest = pd.read_csv('player_zone_contest_fg_2014_15.csv')
    player_contest['player_name'] = player_contest['player_name'].str.lower()
    return zone_shooting, league_contest, player_contest


df = load_data()
zone_shooting_df, league_contest_df, player_contest_df = load_zone_data()

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
# NEW FEATURE: Interactive shot-zone hot map (real shot data)
# ===========================================================================
#
# DATA SOURCES (see load_zone_data() above for full detail):
#   1. player_zone_shooting.csv — REAL per-player, per-season FG% in each of
#      6 zones, aggregated from ~2.8 million actual shot attempts (2005-06
#      through 2018-19), pulled from stats.nba.com shot chart data.
#   2. league_zone_contest_fg_2014_15.csv — REAL league-wide FG% by zone AND
#      by closest-defender distance (Wide Open 6ft+, Slight Contest 4-6ft,
#      Heavy Contest 0-4ft), from the NBA's 2014-15 SportVU tracking data.
#   3. player_zone_contest_fg_2014_15.csv — the same defender-distance splits
#      computed per player (min. 10 shots in that zone), 2014-15 only.
#
# HOW THE NUMBER SHOWN IS CHOSEN, per zone: see build_zone_estimate() below —
# real player tracking data first, then real season FG% spread by the real
# league contest-level shape, then a league-average fallback.

CONTEST_LEVELS = ['Wide Open', 'Slight Contest', 'Heavy Contest']

# Corner side can't be recovered from the 2014-15 tracking data (no shot
# coordinates there, only distance) so both corners share one real shape.
CONTEST_ZONE_LOOKUP = {
    'Restricted Area (Layup)':  'Restricted Area (Layup)',
    'Free Throw / Short Range': 'Free Throw / Short Range',
    'Mid-Range':                'Mid-Range',
    'Left Corner 3':            'Corner 3',
    'Right Corner 3':           'Corner 3',
    'Above the Break 3':        'Above the Break 3',
}

HOOP_XY = (0, 5.25)


def classify_zone(x, y):
    """Maps any (x, y) court coordinate to one of the 6 shot zones, using the
    same real NBA zone boundaries (restricted-area radius, paint width,
    corner-3 line, three-point arc radius) as the shot-log data itself."""
    dist_hoop = math.sqrt(x ** 2 + (y - HOOP_XY[1]) ** 2)
    if dist_hoop <= 4:
        return 'Restricted Area (Layup)'
    if abs(x) <= 8 and 0 <= y <= 19:
        return 'Free Throw / Short Range'
    if abs(x) >= 22 and y <= 14.1:
        return 'Left Corner 3' if x < 0 else 'Right Corner 3'
    if dist_hoop >= 23.75:
        return 'Above the Break 3'
    return 'Mid-Range'


def league_contest_shape(contest_zone):
    rows = league_contest_df[league_contest_df['Zone'] == contest_zone]
    by_level = {r['Contest']: r['FG_PCT'] for _, r in rows.iterrows()}
    total_fga = rows['FGA'].sum()
    overall = (rows['FGM'].sum() / total_fga) if total_fga else np.nan
    return by_level, overall


def player_real_zone_fg(player, season, zone):
    row = zone_shooting_df[(zone_shooting_df['Player'] == player) &
                            (zone_shooting_df['Season'] == season) &
                            (zone_shooting_df['Zone'] == zone)]
    if row.empty or row['FGA'].values[0] < 5:
        return None
    return float(row['FG_PCT'].values[0])


def player_real_contest_split(player, contest_zone):
    rows = player_contest_df[(player_contest_df['player_name'] == player.lower()) &
                              (player_contest_df['Zone'] == contest_zone)]
    if rows.empty or rows['FGA'].sum() < 10:
        return None
    return {r['Contest']: r['FG_PCT'] for _, r in rows.iterrows()}


def build_zone_estimate(player, season, zone):
    """Returns (dict of contest-level -> FG%, source label) for one zone."""
    contest_zone = CONTEST_ZONE_LOOKUP[zone]
    league_shape, league_overall = league_contest_shape(contest_zone)

    real_split = player_real_contest_split(player, contest_zone)
    if real_split and all(c in real_split for c in CONTEST_LEVELS):
        return real_split, 'Real 2014-15 tracking data for this player'

    real_zone_fg = player_real_zone_fg(player, season, zone)
    if real_zone_fg is not None and league_overall and not pd.isna(league_overall):
        est = {c: float(np.clip(real_zone_fg * (league_shape.get(c, league_overall) / league_overall), 0.05, 0.95))
               for c in CONTEST_LEVELS}
        return est, "Player's real season FG% in this zone, spread using real league contest shape"

    return {c: league_shape.get(c, 0.35) for c in CONTEST_LEVELS}, 'League average (no shot data for this player/zone)'


@st.cache_data
def build_hot_zone_grid(player, season, cell_size=1.0):
    """Builds the heatmap grid: every cell colored by the player's real
    Slight-Contest FG% for whichever zone that cell falls in, and tagged with
    the zone name so a click anywhere resolves to a zone + exact coordinate."""
    xs = np.arange(-25, 25, cell_size) + cell_size / 2
    ys = np.arange(0, 40, cell_size) + cell_size / 2  # no real shots that deep

    zone_fg_lookup = {}
    for zone in CONTEST_ZONE_LOOKUP:
        est, _ = build_zone_estimate(player, season, zone)
        zone_fg_lookup[zone] = est['Slight Contest']

    z = np.empty((len(ys), len(xs)))
    customdata = np.empty((len(ys), len(xs)), dtype=object)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            zone = classify_zone(x, y)
            z[i, j] = zone_fg_lookup[zone]
            customdata[i, j] = zone
    return xs, ys, z, customdata


def build_court_figure(player, season):
    fig = go.Figure()

    xs, ys, z, customdata = build_hot_zone_grid(player, season)
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=z, customdata=customdata,
        colorscale='YlOrRd', zmin=0.20, zmax=0.75, showscale=False, zsmooth=False,
        hovertemplate='%{customdata}<br>Slight-contest FG%%: %{z:.1%}<extra></extra>',
        name='hotzones'
    ))

    court_shapes = [
        dict(type='rect', x0=-25, y0=0, x1=25, y1=47, line=dict(color='white', width=2)),
        dict(type='rect', x0=-8, y0=0, x1=8, y1=19, line=dict(color='white', width=2)),
        dict(type='circle', x0=-6, y0=13, x1=6, y1=25, line=dict(color='white', width=2)),
        dict(type='circle', x0=-4, y0=1.25, x1=4, y1=9.25, line=dict(color='white', width=2)),
        dict(type='line', x0=-3, y0=4, x1=3, y1=4, line=dict(color='white', width=3)),
        dict(type='line', x0=-22, y0=0, x1=-22, y1=14.2, line=dict(color='white', width=2)),
        dict(type='line', x0=22, y0=0, x1=22, y1=14.2, line=dict(color='white', width=2)),
        dict(type='circle', x0=-0.75, y0=4.5, x1=0.75, y1=6, line=dict(color='black', width=2)),
    ]
    arc_x, arc_y = [], []
    for deg in range(0, 181):
        rad = math.radians(deg)
        arc_x.append(23.75 * math.cos(rad))
        arc_y.append(5.25 + 23.75 * math.sin(rad))
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines',
                              line=dict(color='white', width=2), hoverinfo='skip',
                              showlegend=False, name='arc'))
    fig.update_layout(shapes=court_shapes)

    for zone, (zx, zy) in {
        'Restricted Area (Layup)': (0, 3), 'Free Throw / Short Range': (0, 14),
        'Mid-Range': (14, 16), 'Left Corner 3': (-23.5, 6), 'Right Corner 3': (23.5, 6),
        'Above the Break 3': (0, 32),
    }.items():
        fig.add_annotation(x=zx, y=zy, text=zone, showarrow=False,
                            font=dict(color='black', size=9), bgcolor='rgba(255,255,255,0.55)')

    fig.update_layout(
        plot_bgcolor='#1a5f3f', paper_bgcolor='#1a5f3f',
        xaxis=dict(visible=False, range=[-27, 27]),
        yaxis=dict(visible=False, range=[-2, 42], scaleanchor='x', scaleratio=1),
        height=560, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=f"{selected_player} ({season}) — click anywhere on the court",
                    font=dict(color='white'))
    )
    return fig


def build_shot_animation(x0, y0, make_probability, seed):
    """Animated top-down shot: the ball travels from the clicked spot to the
    hoop; marker size humps up then down to suggest the arc's height since
    this is a 2D bird's-eye view, not a side view. Outcome (make/miss) is a
    single simulated draw weighted by the zone's Slight-Contest probability —
    a fun simulation of one shot, not a real recorded result."""
    rng = np.random.default_rng(seed)
    made = bool(rng.random() < make_probability)

    n_frames = 18
    t = np.linspace(0, 1, n_frames)
    ball_x = x0 + t * (HOOP_XY[0] - x0)
    ball_y = y0 + t * (HOOP_XY[1] - y0)
    size = 12 + 20 * np.sin(np.pi * t)

    fig = go.Figure(
        data=[
            go.Scatter(x=[x0], y=[y0], mode='markers+text', text=['🏀'],
                       textfont=dict(size=22), marker=dict(size=1, color='rgba(0,0,0,0)'),
                       name='shooter', showlegend=False),
            go.Scatter(x=[ball_x[0]], y=[ball_y[0]], mode='markers',
                       marker=dict(size=size[0], color='#d35400', line=dict(color='black', width=1)),
                       name='ball', showlegend=False),
        ],
        layout=go.Layout(
            plot_bgcolor='#1a5f3f', paper_bgcolor='#1a5f3f',
            xaxis=dict(visible=False, range=[-27, 27]),
            yaxis=dict(visible=False, range=[-2, 20], scaleanchor='x', scaleratio=1),
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            shapes=[dict(type='rect', x0=-8, y0=0, x1=8, y1=19, line=dict(color='white', width=1)),
                    dict(type='circle', x0=-0.75, y0=4.5, x1=0.75, y1=6, line=dict(color='white', width=2))],
            updatemenus=[dict(type='buttons', showactive=False, y=1, x=1.15,
                               buttons=[dict(label='▶ Play Shot', method='animate',
                                             args=[None, dict(frame=dict(duration=60, redraw=True),
                                                               fromcurrent=True, transition=dict(duration=0))])])],
        ),
        frames=[
            go.Frame(
                data=[go.Scatter(x=[ball_x[k]], y=[ball_y[k]],
                                  marker=dict(size=size[k],
                                              color=('#27ae60' if (made and k == n_frames - 1) else '#d35400')),
                traces=[1],
            ) for k in range(n_frames)
        ] + [
            go.Frame(
                data=[go.Scatter(x=[HOOP_XY[0]], y=[HOOP_XY[1]],
                                  marker=dict(size=26, color='#27ae60' if made else '#c0392b'))],
                traces=[1],
                layout=go.Layout(annotations=[dict(
                    x=0, y=10, text=('SWISH! 🎉' if made else 'CLANK ❌'),
                    showarrow=False, font=dict(size=22, color='white'))])
            )
        ]
    )
    return fig, made


st.markdown("### 🏀 Shot-Zone Probability Explorer")
st.caption(
    "Court shaded by the player's real shooting %% in each zone (yellow = colder, red = hotter). "
    "Click anywhere on the court to see the make-probability breakdown for that zone and watch a "
    "simulated shot from that exact spot."
)

if player_data.empty:
    st.write("No data available to build the shot chart for this player.")
else:
    available_seasons = sorted(player_data['Season'].dropna().unique())
    zone_season = st.selectbox('Season for shot-zone breakdown:', available_seasons,
                                index=len(available_seasons) - 1, key='zone_season')

    court_fig = build_court_figure(selected_player, zone_season)
    selection = st.plotly_chart(
        court_fig, use_container_width=True,
        on_select='rerun', selection_mode='points', key='shot_zone_court'
    )

    clicked = None
    if selection and selection.get('selection', {}).get('points'):
        point = selection['selection']['points'][0]
        clicked = (point.get('customdata'), point.get('x'), point.get('y'))

    if clicked is None:
        st.info("Click anywhere on the court above to see the make-probability breakdown for that spot.")
    else:
        clicked_zone, click_x, click_y = clicked
        player_vals, source_label = build_zone_estimate(selected_player, zone_season, clicked_zone)
        league_shape, _ = league_contest_shape(CONTEST_ZONE_LOOKUP[clicked_zone])

        st.markdown(f"#### {clicked_zone}")
        st.caption(f"Source: {source_label}")

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(name=selected_player, x=CONTEST_LEVELS,
                                  y=[player_vals[c] * 100 for c in CONTEST_LEVELS], marker_color='royalblue'))
        bar_fig.add_trace(go.Bar(name='League Average', x=CONTEST_LEVELS,
                                  y=[league_shape.get(c, 0) * 100 for c in CONTEST_LEVELS], marker_color='lightgray'))
        bar_fig.update_layout(barmode='group', yaxis_title='Make Probability (%)',
                               yaxis=dict(range=[0, 100]), height=380,
                               title=f"{clicked_zone}: {selected_player} vs. League Average")
        st.plotly_chart(bar_fig, use_container_width=True)

        cols = st.columns(3)
        for c, col in zip(CONTEST_LEVELS, cols):
            with col:
                league_val = league_shape.get(c, 0)
                st.metric(
                    label=c,
                    value=f"{player_vals[c] * 100:.1f}%",
                    delta=f"{(player_vals[c] - league_val) * 100:+.1f} pts vs league"
                )

        st.markdown("##### Simulated shot from this spot")
        st.caption(
            "One randomized shot from the clicked spot, weighted by the Slight-Contest probability "
            "above — press Play. This is a fun single-shot simulation, not a recorded or predicted result."
        )
        anim_fig, made = build_shot_animation(
            click_x, click_y, player_vals['Slight Contest'],
            seed=hash((selected_player, zone_season, clicked_zone, round(click_x, 1), round(click_y, 1))) % (2**32)
        )
        st.plotly_chart(anim_fig, use_container_width=True, key='shot_animation')

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
