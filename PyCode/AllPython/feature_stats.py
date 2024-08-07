import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def feature_plots(presabs, dur, count):
    eps_features = {
        'contour following': ['local shape'],
        'enclosure': ['global shape', 'weight'],  # 'enclosure' mapped to both 'global shape' and 'weight'
        'enclosure part': ['local shape'],
        'function test': ['function'],
        'pressure': ['hardness'],
        'rotation': ['local shape'],
        'translation': ['local shape'],
        'weighting': ['weight']
    }

    # Create a copy of dur, including the 'Family' column in the new DataFrame
    new_dur = dur.copy()
    # Ensure 'Family' is copied directly to the expanded_dur
    expanded_dur = pd.DataFrame(index=new_dur.index)
    expanded_dur['Family'] = new_dur['Family']

    expected_numeric_columns = [col for col in new_dur.columns if col in eps_features]
    for col in expected_numeric_columns:
        new_dur[col] = pd.to_numeric(new_dur[col], errors='coerce')
        for feature in eps_features[col]:
            if feature in expanded_dur:
                expanded_dur[feature] += new_dur[col]  # Add to existing feature column
            else:
                expanded_dur[feature] = new_dur[col]  # Create new feature column if it doesn't exist

    # Group by 'Family' and sum up features within each family
    expanded_dur = expanded_dur.groupby('Family').sum()

    # Normalize the data to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    numeric_cols = expanded_dur.select_dtypes(include=[np.number]).columns
    expanded_dur[numeric_cols] = scaler.fit_transform(expanded_dur[numeric_cols])

    # Plotting
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(expanded_dur[numeric_cols], annot=False, cmap='gray_r')  # Use reversed grayscale
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=45, size=12)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, size=12)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Normalized time', rotation=270, labelpad=20)
    plt.title('Time Spent Exploring Each Feature')
    plt.tight_layout()
    plt.savefig('./results/feature_Duration.svg', format='svg', dpi=600)
    plt.close()
