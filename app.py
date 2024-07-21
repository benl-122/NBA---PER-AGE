import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the CSV file
df = pd.read_csv('player_advanced_stats.csv')

st.title("Player Efficiency Rating")
st.divider()

# Filter unique positions
position_list = df['Pos'].unique()
selected_position = st.sidebar.selectbox('Select a position:', position_list)

# Filter data based on selected position
filtered_df = df[df['Pos'] == selected_position]

# Display the list of players for the selected position in the sidebar
player_list = filtered_df['Player'].unique()
selected_player = st.sidebar.selectbox('Select a player:', player_list)

# Add a metric selector to the sidebar
metric_list = ['PER', 'TS%', 'FTr', 'ORB%', 'DRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']
selected_metric = st.sidebar.selectbox('Select a metric:', metric_list)

# Filter the data for the selected player
player_data = filtered_df[filtered_df['Player'] == selected_player]

# Reorder columns to insert 'Season' as the 4th column
if 'Season' in player_data.columns:
    cols = player_data.columns.tolist()
    # Remove 'Season' from its current position
    cols.remove('Season')
    # Insert 'Season' as the 4th column
    cols.insert(3, 'Season')
    player_data = player_data[cols]
    
# Display the selected player's data
st.header(f"Player: {selected_player} ({selected_position})")
st.write(player_data)

# Calculate average metric and Age for the selected position
average_metric = filtered_df.groupby('Age')[selected_metric].mean().reset_index()

# Plotting the bar graph for player's metric vs Age
plt.figure(figsize=(10, 6))
plt.bar(player_data['Age'], player_data[selected_metric], color='skyblue')
plt.title(f'{selected_player} - {selected_metric} vs Age')
plt.xlabel('Age')
plt.ylabel(selected_metric)
plt.grid(True)

# Display the plot for player's metric vs Age within Streamlit
st.pyplot()

# Display header for average data
st.header(f"Average Data for {selected_position}")

# Plotting the bar graph for average metric vs Age for the selected position
plt.figure(figsize=(10, 6))
plt.bar(average_metric['Age'], average_metric[selected_metric], color='skyblue')
plt.title(f'Average {selected_metric} vs Age for {selected_position}')
plt.xlabel('Age')
plt.ylabel(f'Average {selected_metric}')
plt.grid(True)

# Display the plot for average metric vs Age within Streamlit
st.pyplot()

# Calculate median metric and Age for the selected position
median_metric = filtered_df.groupby('Age')[selected_metric].median().reset_index()

# Display header for median data
st.header(f"Median Data for {selected_position}")

# Plotting the bar graph for median metric vs Age for the selected position
plt.figure(figsize=(10, 6))
plt.bar(median_metric['Age'], median_metric[selected_metric], color='skyblue')
plt.title(f'Median {selected_metric} vs Age for {selected_position}')
plt.xlabel('Age')
plt.ylabel(f'Median {selected_metric}')
plt.grid(True)

# Display the plot for median metric vs Age within Streamlit
st.pyplot()

# Plotting the line graph for player's metric vs Age and average metric vs Age
plt.figure(figsize=(10, 6))
plt.plot(player_data['Age'], player_data[selected_metric], marker='o', linestyle='-', color='blue', label=f'{selected_player}')
plt.plot(average_metric['Age'], average_metric[selected_metric], marker='o', linestyle='--', color='orange', label='Average')
plt.title(f'{selected_player} vs Average - {selected_metric} vs Age')
plt.xlabel('Age')
plt.ylabel(selected_metric)
plt.legend()
plt.grid(True)

# Display the plot within Streamlit
st.pyplot()

# Plotting the line graph for player's metric vs Age and median metric vs Age
plt.figure(figsize=(10, 6))
plt.plot(player_data['Age'], player_data[selected_metric], marker='o', linestyle='-', color='blue', label=f'{selected_player}')
plt.plot(median_metric['Age'], median_metric[selected_metric], marker='o', linestyle='--', color='green', label='Median')
plt.title(f'{selected_player} vs Median - {selected_metric} vs Age')
plt.xlabel('Age')
plt.ylabel(selected_metric)
plt.legend()
plt.grid(True)

# Display the plot within Streamlit
st.pyplot()

st.write("Note: the player efficiency rating sometimes spikes in the later years because typically, only star players continue to play at an older age. The sample size is therefore smaller including only very good players, and this leads to a higher efficiency rating than what otherwise may be expected with an increase in age.")
