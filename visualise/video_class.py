import cv2


class VideoVisualizer:
    def __init__(self, device=None, samplerate=44100, blocksize=1024):
        self.cap = cv2.VideoCapture(0)
        self.running = False

    def _run(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame")
                break

            current_frame = frame.copy()

            # # Draw a rectangle
            # cv2.rectangle(current_frame, (50, 50), (300, 200), (0, 255, 0), 2)

            # # Draw a circle
            # cv2.circle(current_frame, (400, 200), 50, (255, 0, 0), 3)

            # Add text
            cv2.putText(
                current_frame,
                "Live Camera",
                (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            # Show the frame
            cv2.imshow("My Camera", current_frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def open(self):
        if self.running:
            return

        self.running = True

        self._run()

    def close(self):
        if not self.running:
            return

        self.running = False


visualizer = VideoVisualizer()

visualizer.open()

# Now it's running in background

# When done:
visualizer.close()
