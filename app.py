import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.read_csv('player_advanced_stats.csv')

# Filter by position
position_list = df['Pos'].unique()
selected_position = st.sidebar.selectbox('Select a position:', position_list)
filtered_df = df[df['Pos'] == selected_position]
player_list = filtered_df['Player'].unique()
selected_player = st.sidebar.selectbox('Select a player:', player_list)

# user chooses metric
metric_list = ['PER', 'TS%', 'FTr', 'ORB%', 'DRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']
selected_metric = st.sidebar.selectbox('Select a metric:', metric_list)

# show data accordingly
player_data = filtered_df[filtered_df['Player'] == selected_player]

#explain metrics
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

#title
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

# Calculate average
average_metric = filtered_df.groupby('Age')[selected_metric].mean().reset_index()

# Player vs age
plt.figure(figsize=(10, 6))
plt.bar(player_data['Age'], player_data[selected_metric], color='skyblue')
plt.title(f'{selected_player} - {selected_metric} vs Age')
plt.xlabel('Age')
plt.ylabel(selected_metric)
plt.grid(True)

# graph
###st.pyplot()

# Title for average data
###st.header(f"Average Data for {selected_position}")

# average vs age
plt.figure(figsize=(10, 6))
plt.bar(average_metric['Age'], average_metric[selected_metric], color='skyblue')
plt.title(f'Average {selected_metric} vs Age for {selected_position}')
plt.xlabel('Age')
plt.ylabel(f'Average {selected_metric}')
plt.grid(True)

# graph
###st.pyplot()

# Calculate median
median_metric = filtered_df.groupby('Age')[selected_metric].median().reset_index()

# median title
###st.header(f"Median Data for {selected_position}")

# median vs age
plt.figure(figsize=(10, 6))
plt.bar(median_metric['Age'], median_metric[selected_metric], color='skyblue')
plt.title(f'Median {selected_metric} vs Age for {selected_position}')
plt.xlabel('Age')
plt.ylabel(f'Median {selected_metric}')
plt.grid(True)

# graph
###st.pyplot()

# line graph, player and average vs age
plt.figure(figsize=(10, 6))
plt.plot(player_data['Age'], player_data[selected_metric], marker='o', linestyle='-', color='blue', label=f'{selected_player}')
plt.plot(average_metric['Age'], average_metric[selected_metric], marker='o', linestyle='--', color='orange', label='Average')
plt.title(f'{selected_player} vs Average - {selected_metric} vs Age')
plt.xlabel('Age')
plt.ylabel(selected_metric)
plt.legend()
plt.grid(True)

# graph
###st.pyplot()

# line graph, player and median vs age
plt.figure(figsize=(10, 6))
plt.plot(player_data['Age'], player_data[selected_metric], marker='o', linestyle='-', color='blue', label=f'{selected_player}')
plt.plot(median_metric['Age'], median_metric[selected_metric], marker='o', linestyle='--', color='green', label='Median')
plt.title(f'{selected_player} vs Median - {selected_metric} vs Age')
plt.xlabel('Age')
plt.ylabel(selected_metric)
plt.legend()
plt.grid(True)

# graph
###st.pyplot()

# player's  vs Age and average  vs Age and median  vs Age
plt.figure(figsize=(10, 6))
plt.plot(player_data['Age'], player_data[selected_metric], marker='o', linestyle='-', color='blue', label=f'{selected_player}')
plt.plot(average_metric['Age'], average_metric[selected_metric], marker='o', linestyle='--', color='orange', label='Average')
plt.plot(median_metric['Age'], median_metric[selected_metric], marker='o', linestyle='--', color='green', label='Median')

plt.title(f'{selected_player} vs Average and Median - {selected_metric} vs Age')
plt.xlabel('Age')
plt.ylabel(selected_metric)
plt.legend()
plt.grid(True)

# graph
st.pyplot()

#note about suvivorship bias
st.write("Note: the player efficiency rating sometimes spikes in the later years because typically, only star players continue to play at an older age. The sample size is therefore smaller including only very good players, and this leads to a higher efficiency rating than what otherwise may be expected with an increase in age.")
