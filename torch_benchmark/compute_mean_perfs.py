import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_mean_perfs(input_file, output_file, group_column):
    df = pd.read_csv(input_file) # Read the input CSV file
    df = df.drop('run', axis=1) # Remove the 'run' column
    
    grouped_df = df.groupby(group_column).mean() # Compute the mean of the grouped column
    grouped_df.to_csv(output_file) # Save the result to the output CSV file

def format_scenario_name(scenario_name):
    # Remove 'model_best_' at beginning of scenario name
    scenario_name = scenario_name[11:]
    # replace '_' by ' ' in scenario names
    scenario_name = scenario_name.replace('_', ' ')
    # replace last ' ' by '\n' in scenario names
    scenario_name = scenario_name[::-1].replace(' ', '\n', 1)[::-1]
    return scenario_name

def plot_repartition(input_file, column, score_column):
    df = pd.read_csv(input_file) # Read the input CSV file
    df = df.drop('run', axis=1) # Remove the 'run' column
    grouped_df = df.groupby(column) # Group the dataframe by the given column
    
    data = [] # Create a list to store the data for each group
    scenario_names = []
    # Iterate over each group and extract the values for the column
    for group_name, group_data in grouped_df:
        data.append(group_data[score_column].values)
        scenario_names.append(group_name)
    
    # format scenario names
    scenario_names = [format_scenario_name(scenario_name) for scenario_name in scenario_names]

    # FIXME: sorted version of the plot
    # Sort the data based on the mean value of each group
    # sorted_data = sorted(data, key=lambda x: np.mean(x), reverse=True)
    # sorted_groups = [group_name for _, group_name in sorted(zip(sorted_data, grouped_df.groups.keys()), key=lambda x: np.mean(x[0]), reverse=True)]
    # Create a boxplot using the sorted data
    # plt.boxplot(sorted_data)
    # # Set the x-axis tick labels to the sorted group names
    # plt.xticks(range(1, len(grouped_df.groups) + 1), sorted_groups)

    bp = plt.boxplot(data, widths=0.8, patch_artist=True)  # Store the returned dictionary in 'bp'

    # Set thicker lines for boxplot components
    components = ['boxes', 'whiskers', 'caps', 'medians', 'fliers']
    for component in components:
        for element in bp[component]:
            if component == 'boxes':
                element.set_facecolor('none')  # Make the inside of the box transparent
                element.set_linewidth(2)  # Set the line width
            elif component == 'medians':
                element.set_color('red')
                element.set_linewidth(1.5)
            elif component == 'fliers':
                element.set_markeredgecolor('red')
                element.set_markersize(7.5)
            else:
                element.set_linewidth(1.5)
    # Set the x-axis tick labels to the group names
    plt.xticks(range(1, len(grouped_df.groups) + 1), scenario_names)
    
    # Set the title and labels
    plt.title(score_column)
    plt.xlabel(column)
    plt.ylabel('Score')
    
    # Show the plot
    plt.show()

# Example usages
# input_file = 'torch_benchmark/results.csv'
input_file = 'torch_benchmark/results_main_scenarios.csv'
output_file = 'torch_benchmark/results_mean.csv'
group_column = 'model'

compute_mean_perfs(input_file, output_file, group_column)
plot_repartition(input_file, group_column, score_column='Species_F1-score')
plot_repartition(input_file, group_column, score_column='Genus_F1-score')
plot_repartition(input_file, group_column, score_column='Family_F1-score')