# def weber_frac():
#
#     families_dict = {
#         "cutlery": {"Fork", "Spoon", "Knife"},
#         "vessels": {"Metal Mug", "Ceramic Mug", "Glass"},
#         "plates": {"Ceramic Plate", "Metal Plate", "Plastic Plate"},
#         "balls": {"Squash Ball", "Tennis Ball", "Ping-Pong Ball"},
#         "geometric": {"Cylinder", "Cube", "Prism"},
#     }
#
#
#     object_dict = {
#         "Fork": {"weight": 28, "volume": 34.8},
#         "Spoon": {"weight": 35, "volume": 45.8},
#         "Knife": {"weight": 40, "volume": 10.3},
#         "Metal Mug": {"weight": 150, "volume": 498.9},
#         "Ceramic Mug": {"weight": 265, "volume": 479.8},
#         "Glass": {"weight": 265, "volume": 438.9},
#         "Ceramic Plate": {"weight": 375, "volume": 1051.2},
#         "Metal Plate": {"weight": 153, "volume": 618.0},
#         "Plastic Plate": {"weight": 14, "volume": 800.6},
#         "Squash Ball": {"weight": 24, "volume": 34.6},
#         "Tennis Ball": {"weight": 58, "volume": 160.5},
#         "Ping-Pong Ball": {"weight": 3, "volume": 34.6},
#         "Cylinder": {"weight": 30, "volume": 98.2},
#         "Cube": {"weight": 40, "volume": 125.3},
#         "Prism": {"weight": 19, "volume": 54.4},
#     }


import pandas as pd
from itertools import combinations


# Step 1: Define comparison functions
def compare_volume(volume_A, volume_B):
    """
    Compare volumes between two objects based on percentage thresholds:
    - < 3.5: 16%
    - 3.5 to 12: 13%
    - > 12: 10%
    """
    if volume_B < 3.5:
        percentage = 0.16
    elif 3.5 <= volume_B <= 12:
        percentage = 0.13
    else:
        percentage = 0.10

    lower_bound = volume_A * (1 - percentage)
    upper_bound = volume_A * (1 + percentage)
    return "no" if lower_bound <= volume_B <= upper_bound else "yes"


def compare_weight(weight_A, weight_B):
    """
    Compare weights between two objects:
    - Threshold: 29% difference (±29%).
    """
    lower_bound = weight_A * (1 - 0.29)
    upper_bound = weight_A * (1 + 0.29)
    return "no" if lower_bound <= weight_B <= upper_bound else "yes"

def compare_weight_restrictive(weight_A, weight_B):
    """
    Compare weights between two objects:
    - Threshold: 18% difference (±18%).
    """
    lower_bound = weight_A * (1 - 0.18)
    upper_bound = weight_A * (1 + 0.18)
    return "no" if lower_bound <= weight_B <= upper_bound else "yes"

def weber_frac():

    # Step 2: The object dictionary
    families_dict = {
            "cutlery": {"Fork", "Spoon", "Knife"},
            "vessels": {"Metal Mug", "Ceramic Mug", "Glass"},
            "plates": {"Ceramic Plate", "Metal Plate", "Plastic Plate"},
            "balls": {"Squash Ball", "Tennis Ball", "Ping-Pong Ball"},
            "geometric": {"Cylinder", "Cube", "Prism"},
        }


    object_dict = {
        "Fork": {"weight": 28, "volume": 34.8},
        "Spoon": {"weight": 35, "volume": 45.8},
        "Knife": {"weight": 40, "volume": 10.3},
        "Metal Mug": {"weight": 150, "volume": 498.9},
        "Ceramic Mug": {"weight": 265, "volume": 479.8},
        "Glass": {"weight": 265, "volume": 438.9},
        "Ceramic Plate": {"weight": 375, "volume": 1051.2},
        "Metal Plate": {"weight": 153, "volume": 618.0},
        "Plastic Plate": {"weight": 14, "volume": 800.6},
        "Squash Ball": {"weight": 24, "volume": 34.6},
        "Tennis Ball": {"weight": 58, "volume": 160.5},
        "Ping-Pong Ball": {"weight": 3, "volume": 34.6},
        "Cylinder": {"weight": 30, "volume": 98.2},
        "Cube": {"weight": 40, "volume": 125.3},
        "Prism": {"weight": 19, "volume": 54.4},
    }

    # Step 3: Generate combinations within families and compute results
    results = []

    for objects in families_dict.values():

        sorted_objects = sorted(objects)
        # Generate unique combinations for the current family
        family_pairs = list(combinations(sorted_objects, 2))

        for obj_A, obj_B in family_pairs:
            weight_A, volume_A = object_dict[obj_A]["weight"], object_dict[obj_A]["volume"]
            weight_B, volume_B = object_dict[obj_B]["weight"], object_dict[obj_B]["volume"]

            # Compute comparisons
            results.append({
                "Object_A": obj_A,
                "Object_B": obj_B,
                "Volume_AtoB": compare_volume(volume_A, volume_B),
                "Volume_BtoA": compare_volume(volume_B, volume_A),
                # "Weight_AtoB": compare_weight(weight_A, weight_B),
                # "Weight_BtoA": compare_weight(weight_B, weight_A),
                "Weight_AtoB": compare_weight_restrictive(weight_A, weight_B),
                "Weight_BtoA": compare_weight_restrictive(weight_B, weight_A),
            })

    # Step 4: Create the DataFrame
    df = pd.DataFrame(results)

    # Step 5: Display the final DataFrame
    print("\nFinal DataFrame with Object Pairs and Comparison Results:")
    print(df)
