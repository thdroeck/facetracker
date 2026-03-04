import numpy as np
import sounddevice as sd
import cv2
import threading


class AudioVisualizer:
    def __init__(self, device=None, samplerate=44100, blocksize=1024):
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize

        self.volume = 0
        self.running = False
        self.stream = None
        self.thread = None

    def _audio_callback(self, indata, frames, time, status):
        # RMS volume calculation
        self.volume = np.sqrt(np.mean(indata**2))

    def _run(self):
        while self.running:
            frame = np.zeros((300, 600, 3), dtype=np.uint8)

            display_volume = self.volume * 1000
            bar_width = min(int(display_volume), 500)

            cv2.rectangle(frame, (50, 150), (50 + bar_width, 200), (0, 255, 0), -1)

            cv2.putText(
                frame,
                f"Volume: {display_volume:.2f}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Audio Visualizer", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.close()

        cv2.destroyAllWindows()

    def open(self):
        if self.running:
            return

        self.running = True

        self.stream = sd.InputStream(
            device=self.device,
            callback=self._audio_callback,
            channels=2,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
        )

        self.stream.start()

        self._run()

    def close(self):
        if not self.running:
            return

        self.running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_volume(self):
        return self.volume


visualizer = AudioVisualizer(device=2)

visualizer.open()

# Now it's running in background

# You can access volume anytime:
print(visualizer.get_volume())

# When done:
visualizer.close()
