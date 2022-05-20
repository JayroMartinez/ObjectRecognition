import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def main():
    # READING FILES AND CREATING DATASET
    file = '../Data/features.csv'
    objects = pd.read_csv(file, sep=';', engine='python')
    repeated = pd.concat([objects] * 100, ignore_index=True)

    labels = repeated["Object"]  # labels

    # ALL FEATURES
    features = repeated.drop(['Object'], axis=1)
    numerical_features = features.drop(['Hardness'], axis=1)

    noise_values = np.arange(0.0, 0.26, 0.01)
    cv_val = 10
    all_score_array = []
    weight_score_array = []
    size_score_array = []

    aux_score = []

    for noise in noise_values:
        noise_feat = numerical_features.applymap(lambda x: x + (x * noise * np.random.normal(0, 1)))

        # ONLY WEIGHT
        features_weight = noise_feat["Weight"]
        features_weight = features_weight[:, None]

        # ONLY SIZE
        features_size = noise_feat[["X", "Y", "Z"]]

        # CLASSIFIER
        clf = SVC(C=1, kernel='rbf', gamma='scale', probability=True, decision_function_shape="ovo")

        # SCORES ALL FEATURES
        aux = cross_val_score(clf, noise_feat, labels, cv=cv_val)
        aux_score.append(aux)
        noise_feat = noise_feat.join(features['Hardness'])  # add 'Hardness' as feature (transformed to categorical int)
        lab_encoder = LabelEncoder()
        noise_feat['Hardness'] = lab_encoder.fit_transform(noise_feat['Hardness'])
        scores_all = cross_val_score(clf, noise_feat, labels, cv=cv_val)
        all_score_array.append(scores_all)

        # SCORES ONLY WEIGHT
        scores_weight = cross_val_score(clf, features_weight, labels, cv=cv_val)
        weight_score_array.append(scores_weight)

        # SCORES ONLY SIZE
        scores_size = cross_val_score(clf, features_size, labels, cv=cv_val)
        size_score_array.append(scores_size)

    size_max = np.max(size_score_array, axis=-1)
    size_min = np.min(size_score_array, axis=-1)
    size_mean = np.mean(size_score_array, axis=-1)
    weight_max = np.max(weight_score_array, axis=-1)
    weight_min = np.min(weight_score_array, axis=-1)
    weight_mean = np.mean(weight_score_array, axis=-1)
    all_max = np.max(all_score_array, axis=-1)
    all_min = np.min(all_score_array, axis=-1)
    all_mean = np.mean(all_score_array, axis=-1)
    max_range = max(max(all_max), max(weight_max), max(size_max)) * 100
    min_range = min(min(all_min), min(weight_min), min(size_min)) * 100
    number_of_ticks = 5 * round(((max_range - min_range) / 1) / 5)

    diff = np.mean(aux_score, axis=-1) - all_mean
    print("Accuracy difference between models with and without 'Hardness': ", round(np.mean(diff), 3))

    plt.figure(0)
    plt.hist(np.array(noise_feat / numerical_features).reshape(-1), 100)
    plt.show()

    plt.figure(1)
    plt.plot(noise_values * 100, size_mean * 100, '-r', label='Using Size Features')
    plt.plot(noise_values * 100, weight_mean * 100, '-b', label='Using Weight Feature')
    plt.plot(noise_values * 100, all_mean * 100, '-k', label='Using All Features')
    plt.fill_between(noise_values * 100, weight_max * 100, weight_min * 100, facecolor='blue', alpha=0.2)
    plt.fill_between(noise_values * 100, size_max * 100, size_min * 100, facecolor='red', alpha=0.2)
    plt.fill_between(noise_values * 100, all_max * 100, all_min * 100, facecolor='black', alpha=0.2)
    plt.grid(axis='y')
    plt.yticks(range(int(max_range - number_of_ticks), int(max_range) + 1, 5))
    plt.xlabel("Noise (%)")
    plt.ylabel("Model Accuracy (%)")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    main()
