import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def get_dist_heatmap():

    sns.set(font_scale=2)

    # Data from the table with all red 'Yes' values converted to 'No' and new columns 'Global shape' and 'Local Shape'
    data = {
        'Pair': [
            'Fork - Spoon', 'Fork - Knife', 'Spoon - Knife', 'Cutlery',
            'Metal Mug - Ceramic Mug', 'Metal Mug - Glass', 'Ceramic Mug - Glass', 'Mugs',
            'Ceramic Plate - Metal Plate', 'Ceramic Plate - Plastic Plate', 'Metal Plate - Plastic Plate', 'Plates',
            'Squash Ball - Tennis Ball', 'Squash Ball - Ping-Pong Ball', 'Tennis Ball - Ping-Pong Ball', 'Sport Balls',
            'Cylinder - Cube', 'Cylinder - Prism', 'Cube - Prism', 'Geometric'
        ],
        'Weight': [
            'Feasible', 'Feasible', 'Feasible', 'Feasible',
            'Yes', 'Yes', 'No', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes',
            'Feasible', 'Feasible', 'Feasible', 'Feasible'
        ],
        'Size': [
            'Feasible', 'Feasible', 'Feasible', 'Feasible',
            'Feasible', 'Feasible', 'Feasible', 'Feasible',
            'Feasible', 'Feasible', 'Feasible', 'Feasible',
            'Yes', 'No', 'Yes', 'Yes',
            'Feasible', 'Feasible', 'Feasible', 'Feasible'
        ],
        'Hardness': [
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes',
            'No', 'No', 'No', 'No'
        ],
        'Global shape': [
            'Feasible', 'Feasible', 'Feasible', 'Feasible',
            'No', 'Yes', 'Yes', 'Yes',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'Yes', 'Yes', 'Yes', 'Yes'
        ],
        'Local Shape': [
            'Yes', 'Yes', 'Yes', 'Yes',
            'No', 'Yes', 'Yes', 'Yes',
            'No', 'No', 'No', 'No',
            'Feasible', 'No', 'Feasible', 'Feasible',
            'Yes', 'Yes', 'Yes', 'Yes'
        ]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Convert 'Yes'/'No' to 1/0
    plt.figure(figsize=(12, 12))
    ternary_df = df.set_index('Pair').applymap(lambda x: 1 if x == 'Yes' else (2 if x == 'Feasible' else 0))
    cmap = ListedColormap(['white', 'grey', 'black'])
    sns.heatmap(
        ternary_df,
        cmap=cmap,
        cbar=True,
        linewidths=0.5,
        cbar_kws={'boundaries': [0, 0.5, 1.5, 2.5], 'ticks': [0.25, 1, 1.75], 'spacing': 'proportional'},
        square=False
    )
    colorbar = plt.gca().collections[0].colorbar
    colorbar.set_ticks([0.25, 1, 1.75])  # Two ticks, corresponding to the two colors
    colorbar.set_ticklabels(['No', 'Feasible', 'Yes'])  # Labels for the ticks
    ax = plt.gca()
    yticks = ax.get_yticks()
    ytick_labels = [label.get_text() for label in ax.get_yticklabels()]
    ax.set_yticklabels(
        [f'$\mathbf{{{label}}}$' if label in ['Cutlery', 'Mugs','Plates', 'Sport Balls', 'Geometric'] else label for label in ytick_labels])
    plt.ylabel('')
    # colorbar.set_label('Response')
    # plt.title('Binary Heatmap of Yes/No Table with Global and Local Shape')
    # plt.show()
    plt.tight_layout()
    plt.savefig('./results/distinguish_heatmap.svg', format='svg', dpi=600)
    a=1