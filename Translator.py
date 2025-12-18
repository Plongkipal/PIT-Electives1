# live_captions_webcam.py
import cv2
import whisper
import sounddevice as sd
import numpy as np
import queue
import threading

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # seconds per audio chunk
CHANNELS = 1

audio_queue = queue.Queue()
model = whisper.load_model("base")

current_text = []

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy().flatten())

def transcribe_loop():
    global current_text
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        result = model.transcribe(chunk, fp16=False)
        text = result.get("text", "").strip()
        if text:
            current_text.append(text)
            if len(current_text) > 5:
                current_text = current_text[-5:]

def main():
    # Start transcription thread
    thread = threading.Thread(target=transcribe_loop, daemon=True)
    thread.start()

    # Start audio stream
    stream = sd.InputStream(callback=audio_callback, channels=CHANNELS,
                            samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE*CHUNK_DURATION))
    stream.start()

    cap = cv2.VideoCapture(0)  # Open default webcam

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        caption = " ".join(current_text[-3:])
        if caption:
            cv2.rectangle(frame, (0, frame.shape[0]-50), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
            cv2.putText(frame, caption, (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Live Webcam Captions", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    stream.stop()
    cv2.destroyAllWindows()
    audio_queue.put(None)

if __name__ == "__main__":
    main()