def best_parameter_combination_across_families(df):

    df.columns = ['source', 'family', 'number_of_bins', 'l1_vs_l2', 'c_param', 'accuracies', 'mean_accuracy']
    df['accuracies'] = df['accuracies'].apply(eval)
    # Group data by the parameter combination
    grouped = df.groupby(['number_of_bins', 'l1_vs_l2', 'c_param'])
    # Calculate mean of 'mean_accuracy' across all families for each group
    best_combination = grouped['mean_accuracy'].mean().reset_index()
    # Identify the combination with the highest mean accuracy
    best_parameters = best_combination.loc[best_combination['mean_accuracy'].idxmax()]

    # Print the best parameters
    print("Best Parameters:")
    print("Number of Bins:", best_parameters['number_of_bins'])
    print("L1 vs L2:", best_parameters['l1_vs_l2'])
    print("C Param:", best_parameters['c_param'])
    print("Mean Accuracy:", best_parameters['mean_accuracy'])