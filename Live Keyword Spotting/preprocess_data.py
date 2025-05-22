import os
import numpy as np
import librosa

FEATURES_SAVE_PATH = 'features_data.npy'
LABELS_SAVE_PATH = 'labels_data.npy'

def extract_mfcc_features(file_path, n_mfcc=40):
    """
    Extragem MFCC (media + std), vector de dimensiune 2 * n_mfcc = 80.
    """
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    feature_vector = np.concatenate((mfcc_mean, mfcc_std))
    return feature_vector

def main():
    data_dir = 'subset_augmented'  # folderul cu fișiere augmentate

    if not os.path.exists(data_dir):
        print(f"Eroare: Folderul '{data_dir}' nu există. Ai rulat augment_data.py?")
        return

    features_list = []
    labels_list = []

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Dacă vreodată apare _background_noise_ în subset_augmented, îl marcăm ca 'noise'
        if folder_name.startswith('_'):
            label = 'noise'
        else:
            label = folder_name  # ex. 'yes', 'no', 'up', 'down', 'wow'

        for fname in os.listdir(folder_path):
            if fname.endswith('.wav'):
                file_path = os.path.join(folder_path, fname)
                mfcc_vec = extract_mfcc_features(file_path, n_mfcc=40)
                features_list.append(mfcc_vec)
                labels_list.append(label)

    X = np.array(features_list)
    y = np.array(labels_list)

    np.save(FEATURES_SAVE_PATH, X)
    np.save(LABELS_SAVE_PATH, y)

    print("=== Preprocesare completă ===")
    print(f"Fișiere salvate: {FEATURES_SAVE_PATH} și {LABELS_SAVE_PATH}")
    print(f"Nr. total de eșantioane: {len(X)}")
    print(f"Dimensiunea unui vector de caracteristici (features): {X.shape[1]}")

if __name__ == '__main__':
    main()
