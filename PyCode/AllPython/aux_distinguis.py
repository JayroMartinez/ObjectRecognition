import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

def get_dist_heatmap():

    sns.set(font_scale=2)

    # Data from the table with all red 'Yes' values converted to 'No' and new columns 'Global shape' and 'Local Shape'
    data = {
        'Pair': [
            'Fork - Knife', 'Fork - Spoon', 'Knife - Spoon', 'Cutlery',
            'Ceramic Mug - Glass', 'Ceramic Mug - Metal Mug', 'Glass - Metal Mug', 'Vessels',
            'Ceramic Plate - Metal Plate', 'Ceramic Plate - Plastic Plate', 'Metal Plate - Plastic Plate', 'Plates',
            'Ping-Pong Ball - Squash Ball', 'Ping-Pong Ball - Tennis Ball', 'Squash Ball - Tennis Ball', 'Sport Balls',
            'Cube - Cylinder', 'Cube - Prism', 'Cylinder - Prism',  'Geometric'
        ],
        'Weight': [
            'Yes', 'Yes', 'No', 'Yes',
            'No', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes',
        ],
        'Size': [
            'Yes', 'Yes', 'Yes', 'Yes',
            'No', 'No', 'Yes', 'No',
            'Yes', 'Yes', 'Yes', 'Yes',
            'No', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes',
        ],
        'Hardness': [
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'No', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes',
            'No', 'No', 'No', 'No',
        ],
        'Global shape': [
            'Yes', 'No', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes',
            'No', 'No', 'No', 'No',
            'No', 'No', 'No', 'No',
            'Yes', 'Yes', 'Yes', 'Yes',
        ],
        'Local Shape': [
            'Yes', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes',
            'No', 'No', 'No', 'No',
            'No', 'Yes', 'Yes', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes',
        ]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    df.set_index('Pair', inplace=True)
    df.replace({'Yes': 1, 'No': 0}, inplace=True)

    cmap = ListedColormap(['black', 'white'])

    plt.figure(figsize=(12, 12))
    sns.heatmap(
        df,
        cmap=cmap,
        cbar=True,
        linewidths=0.5,
        cbar_kws={'boundaries': [-0.5, 0.5, 1.5], 'ticks': [0, 1]},
        square=False
    )

    colorbar = plt.gca().collections[0].colorbar
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(['No', 'Yes'])

    ax = plt.gca()
    ytick_labels = [label.get_text() for label in ax.get_yticklabels()]
    ax.set_yticklabels(
        [f'$\mathbf{{{label}}}$' if label in ['Cutlery', 'Vessels', 'Plates', 'Sport Balls', 'Geometric'] else label for
         label in ytick_labels]
    )

    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('./results/distinguish_heatmap.svg', format='svg', dpi=600)

    # # Convert 'Yes'/'No' to 1/0
    # plt.figure(figsize=(12, 12))
    # ternary_df = df.set_index('Pair').applymap(lambda x: 1 if x == 'Yes' else (2 if x == 'Feasible' else 0))
    # cmap = ListedColormap(['white', 'grey', 'black'])
    # sns.heatmap(
    #     ternary_df,
    #     cmap=cmap,
    #     cbar=True,
    #     linewidths=0.5,
    #     cbar_kws={'boundaries': [0, 0.5, 1.5, 2.5], 'ticks': [0.25, 1, 1.75], 'spacing': 'proportional'},
    #     square=False
    # )
    # colorbar = plt.gca().collections[0].colorbar
    # colorbar.set_ticks([0.25, 1, 1.75])  # Two ticks, corresponding to the two colors
    # colorbar.set_ticklabels(['No', 'Feasible', 'Yes'])  # Labels for the ticks
    # ax = plt.gca()
    # yticks = ax.get_yticks()
    # ytick_labels = [label.get_text() for label in ax.get_yticklabels()]
    # ax.set_yticklabels(
    #     [f'$\mathbf{{{label}}}$' if label in ['Cutlery', 'Mugs','Plates', 'Sport Balls', 'Geometric'] else label for label in ytick_labels])
    # plt.ylabel('')
    # # colorbar.set_label('Response')
    # # plt.title('Binary Heatmap of Yes/No Table with Global and Local Shape')
    # # plt.show()
    # plt.tight_layout()
    # plt.savefig('./results/distinguish_heatmap.svg', format='svg', dpi=600)
    # a=1