import os

def main():
    data_dir = 'subset'
    if not os.path.exists(data_dir):
        print(f"Eroare: Folderul '{data_dir}' nu există.")
        return

    print(f"Verific fișierele din directorul '{data_dir}':\n")
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            wav_count = sum(1 for f in os.listdir(folder_path) if f.endswith('.wav'))
            print(f" - {folder_name}: {wav_count} fișiere .wav")

    data_dir = 'subset_augmented'

    # Verificăm dacă folderul există
    if not os.path.exists(data_dir):
        print(
            f"Eroare: Folderul '{data_dir}' nu există! Asigură-te că ai rulat augment_data.py sau că folderul e corect.")
        return

    print(f"Verific fișierele .wav din directorul '{data_dir}':\n")

    # Parcurgem fiecare subfolder din subset_augmented
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)

        # Verificăm dacă e subdirector (ignorăm fișiere)
        if os.path.isdir(folder_path):
            # Numărăm fișierele .wav
            wav_count = sum(1 for fname in os.listdir(folder_path) if fname.endswith('.wav'))
            print(f" - {folder_name}: {wav_count} fișiere .wav")


if __name__ == '__main__':
    main()
