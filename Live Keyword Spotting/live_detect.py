import pyaudio
import numpy as np
import librosa
import time
import joblib
from collections import deque  # sau queue.Queue

MODEL_PATH = 'rf_model.pkl'
SCALER_PATH = 'scaler.pkl'

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Scoatem 'down' din setul de cuvinte țintă
TARGET_WORDS = {'yes', 'no', 'up', 'wow'}

# Eliminăm și din THRESHOLDS
THRESHOLDS = {
    'yes': 0.60,
    'no': 0.39,
    'up': 0.56,
    'wow': 0.22,
    'noise': 0.50
}

AUDIO_BUFFER = deque()  # colectăm bytes aici

def simple_vad(audio_np, vad_energy_thresh=0.001):
    energy = np.mean(audio_np**2)
    return (energy > vad_energy_thresh)

def amplitude_normalize(audio_np, target_rms=0.1):
    rms_current = np.sqrt(np.mean(audio_np**2))
    if rms_current < 1e-8:
        return audio_np
    gain = target_rms / rms_current
    audio_out = audio_np * gain
    audio_out = np.clip(audio_out, -1.0, 1.0)
    return audio_out

def process_audio_block(raw_block):
    """
    raw_block: bytes (aprox. 1s = 16000 semnale * 2 bytes = 32000 bytes)
    """
    audio_np = np.frombuffer(raw_block, dtype=np.int16).astype(np.float32)
    audio_np = audio_np / 32768.0

    # VAD
    if not simple_vad(audio_np):
        return None, None

    # amplitude normalize
    audio_np = amplitude_normalize(audio_np, target_rms=0.1)

    mfcc = librosa.feature.mfcc(y=audio_np, sr=RATE, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    features = np.concatenate((mfcc_mean, mfcc_std))
    return features, True

def callback(in_data, frame_count, time_info, status_flags):
    """
    Această funcție e apelată automat de PyAudio la fiecare CHUNK de date.
    in_data: bytes
    """
    # Adăugăm in_data în buffer
    AUDIO_BUFFER.extend(in_data)  # punem byte cu byte
    return (None, pyaudio.paContinue)

def main():
    print("=== Detecție live cu streaming continuu (callback) ===")
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=callback)

    stream.start_stream()
    print("Microfon deschis, callback activat. CTRL+C pt. a opri.\n")

    classes = model.classes_

    try:
        while True:
            time.sleep(0.1)  # mică pauză

            # Verificăm dacă avem >= 32000 bytes (aprox. 1 sec la 16kHz * 2 bytes)
            if len(AUDIO_BUFFER) >= 32000:
                # Scoatem exact 32000 bytes -> 1s
                block = bytearray()
                for _ in range(32000):
                    block.append(AUDIO_BUFFER.popleft())

                # convertim la bytes
                block_bytes = bytes(block)
                # Procesăm
                features, valid = process_audio_block(block_bytes)
                if valid:
                    # predict
                    features_scaled = scaler.transform([features])
                    probs = model.predict_proba(features_scaled)[0]
                    label_index = np.argmax(probs)
                    pred_label = classes[label_index]
                    confidence = probs[label_index]

                    threshold_label = THRESHOLDS.get(pred_label, 0.5)
                    if confidence >= threshold_label:
                        # Dacă modelul încă prezice 'down', nu apare
                        if pred_label != 'noise' and pred_label in TARGET_WORDS:
                            print(f"[Detectat] {pred_label} (conf={confidence:.2f})")

    except KeyboardInterrupt:
        print("\nOprim streaming-ul.")
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    main()
