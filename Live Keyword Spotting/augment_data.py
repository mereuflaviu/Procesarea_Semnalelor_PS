import os
import numpy as np
import librosa
import soundfile as sf
import random


def time_shift(y, shift_max=0.05):
    """
    y: semnal audio (numpy array)
    shift_max: fracțiune maximă din lungime (ex. 0.05 = ±5%)
    Returnează semnalul time-shifted (circular).
    """
    shift = int(random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)


def pitch_shift(y, sr, semitones):
    """
    Schimbă tonalitatea (pitch) cu 'semitones' semitonuri.
    Apel cu parametrii numiți pentru compatibilitate cu diverse versiuni librosa.
    """
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitones)


def speed_change(y, speed_factor):
    """
    Schimbă viteza semnalului. speed_factor > 1.0 => mai rapid, <1 => mai lent.
    """
    return librosa.effects.time_stretch(y=y, rate=speed_factor)


def load_random_noise(noise_folder, sr):
    """
    Încarcă aleator un fișier de zgomot din noise_folder și îl returnează (y_noise).
    """
    noise_files = [f for f in os.listdir(noise_folder) if f.endswith('.wav')]
    if not noise_files:
        return None

    chosen_noise = random.choice(noise_files)
    noise_path = os.path.join(noise_folder, chosen_noise)
    y_noise, sr_noise = librosa.load(noise_path, sr=sr)
    return y_noise


def inject_noise(y, noise_data, noise_factor=0.1):
    """
    Amestecă semnalul y cu noise_data, scalând noise_data cu noise_factor.
    """
    if len(noise_data) > len(y):
        noise_data = noise_data[:len(y)]
    elif len(noise_data) < len(y):
        repeats = (len(y) // len(noise_data)) + 1
        noise_data = np.tile(noise_data, repeats)
        noise_data = noise_data[:len(y)]

    return y + noise_factor * noise_data


def augment_file(file_path, output_folder, sr_desired=16000, noise_folder=None):
    """
    Încarcă fișierul .wav din file_path, aplică augmentări și salvează
    fișierele rezultate în output_folder.
    noise_folder: calea la folderul cu zgomot (ex. subset/_background_noise_).
    """
    y, sr = librosa.load(file_path, sr=sr_desired)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # 1. Salvăm originalul (opțional)
    sf.write(os.path.join(output_folder, f"{base_name}_orig.wav"), y, sr_desired)

    # 2. Time shifting
    y_shifted = time_shift(y, shift_max=0.05)  # ±5% din lungime
    sf.write(os.path.join(output_folder, f"{base_name}_timeshift.wav"), y_shifted, sr_desired)

    # 3. Pitch shifting (±1 și ±2 semitonuri)
    for semitones in [1, -1, 2, -2]:
        y_ps = pitch_shift(y, sr_desired, semitones)
        out_name = f"{base_name}_pitch{semitones}.wav"
        sf.write(os.path.join(output_folder, out_name), y_ps, sr_desired)

    # 4. Speed change (±10%)
    for factor in [0.9, 1.1]:
        y_speed = speed_change(y, factor)
        out_name = f"{base_name}_speed{factor}.wav"
        sf.write(os.path.join(output_folder, out_name), y_speed, sr_desired)

    # 5. Injectare de zgomot (dacă avem noise_folder)
    if noise_folder and os.path.exists(noise_folder):
        noise_data = load_random_noise(noise_folder, sr_desired)
        if noise_data is not None:
            noise_factor = random.uniform(0.05, 0.2)
            y_noisy = inject_noise(y, noise_data, noise_factor=noise_factor)
            sf.write(os.path.join(output_folder, f"{base_name}_noise.wav"), y_noisy, sr_desired)


def main():
    data_dir = 'subset'
    output_dir = 'subset_augmented'
    noise_folder = os.path.join(data_dir, '_background_noise_')
    sr_desired = 16000

    if not os.path.exists(data_dir):
        print(f"Eroare: folderul '{data_dir}' nu există.")
        return

    # Dacă vrei să regenerezi totul de la zero, poți șterge manual folderul subset_augmented/
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parcurgem subfolderele (yes, no, etc.), sărim peste _background_noise_
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if folder_name.startswith('_'):
            # e folderul de noise, nu-l augmentăm
            continue

        out_subfolder = os.path.join(output_dir, folder_name)
        if not os.path.exists(out_subfolder):
            os.makedirs(out_subfolder)

        print(f"--- Augmentare pentru folderul: {folder_name} ---")

        for fname in os.listdir(folder_path):
            if fname.endswith('.wav'):
                file_path = os.path.join(folder_path, fname)
                # Aplicăm augmentări
                augment_file(file_path, out_subfolder, sr_desired, noise_folder)

    print("Augmentare finalizată! Verifică folderul:", output_dir)


if __name__ == '__main__':
    main()
