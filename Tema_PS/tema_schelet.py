import numpy as np
from get_features import create_gabor_filters, plot_spectrum, create_windows, get_features
from sklearn.neighbors import KNeighborsClassifier
import scipy


def main():
    # Load data from matfile 'data.mat'
    data = scipy.io.loadmat('data.mat')
    audio_train, audio_test = data['audio_train'].T, data['audio_test'].T
    labels_train, labels_test = data['labels_train'], data['labels_test']
    fs = data['fs'][0, 0]

    # Preprocesare: utilizăm doar o porțiune a semnalului audio pentru calcul rapid
    alpha = 1.0
    start1 = audio_train.shape[1] // 2 - int(alpha * audio_train.shape[1] // 2) + 1
    end1 = audio_train.shape[1] // 2 + int(alpha * audio_train.shape[1] // 2)
    audio_train_small = audio_train[:, start1:end1]

    start2 = audio_test.shape[1] // 2 - int(alpha * audio_test.shape[1] // 2) + 1
    end2 = audio_test.shape[1] // 2 + int(alpha * audio_test.shape[1] // 2)
    audio_test_small = audio_test[:, start2:end2]


    # Creăm setul de filtre Gabor și salvăm primele filtre
    num_filters = 12
    filter_size = 1102
    gabor_filters = create_gabor_filters(fs, num_filters=num_filters, filter_size=filter_size)
    print(f"Am creat {len(gabor_filters)} filtre Gabor.")

    # Creare ferestre (cerința 4.a)
    print("Creăm ferestre pentru semnalul audio...")
    windows_train = [create_windows(audio, filter_size) for audio in audio_train_small]
    windows_test = [create_windows(audio, filter_size) for audio in audio_test_small]
    print(
        f"Am creat {len(windows_train)} ferestre pentru setul de antrenare și {len(windows_test)} pentru setul de testare.")

    # Afișăm spectrul filtrelor
    plot_spectrum(gabor_filters, fs)

    # Extragem trăsături folosind funcția get_features
    feat_train = get_features(audio_train_small, fs)
    feat_test = get_features(audio_test_small, fs)

    # Pregătim etichetele
    labels_train = labels_train[:, 0]
    labels_test = labels_test[:, 0]

    # Antrenare și testare KNN
    clf = KNeighborsClassifier()
    clf.fit(feat_train, labels_train)
    pred_train = clf.predict(feat_train)
    pred_test = clf.predict(feat_test)

    # Calculăm acuratețea
    acc_train = np.mean(pred_train == labels_train)
    acc_test = np.mean(pred_test == labels_test)
    print(f'Accuracy on train: {acc_train:.2f}')
    print(f'Accuracy on test:  {acc_test:.2f}')


# Main function
if __name__ == "__main__":
    main()
