import cv2
import torch
from torchvision import transforms
from PIL import Image

from model.architectures.net import Net
from model.architectures.vgg19 import VGG19


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")


class VideoVisualizer:
    def __init__(self, path, device=None, samplerate=44100, blocksize=1024):
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.model = VGG19()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((416, 416)),
                transforms.ToTensor(),
            ]
        )

    def _run(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame")
                break

            current_frame = frame.copy()
            frame_h, frame_w, _ = current_frame.shape
            # transform frame to tensor and pass through model

            # --- Preprocessing ---
            # Convert BGR (OpenCV) to RGB (PIL)
            # 1. Start with the PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 2. Apply the transform (This turns it into a Tensor)
            image_tensor = self.transform(pil_img)

            # 3. NOW you can unsqueeze it
            input_tensor = image_tensor.unsqueeze(0)  # This works!

            # --- Inference ---
            with torch.no_grad():
                prediction = self.model(input_tensor)
                # Assuming model outputs [cx, cy, w, h] normalized 0-1
                cx, cy, w, h = prediction[0].tolist()
                print(
                    f"Model Output (normalized): cx={cx:.2f}, cy={cy:.2f}, w={w:.2f}, h={h:.2f}"
                )

            # --- Post-processing (Scaling back to frame size) ---
            pixel_cx = cx * frame_w
            pixel_cy = cy * frame_h
            pixel_w = w * frame_w
            pixel_h = h * frame_h

            # 4. Convert Center-XY to Top-Left (x1, y1) and Bottom-Right (x2, y2)
            # This is what cv2.rectangle needs
            x1 = int(pixel_cx - (pixel_w / 2))
            y1 = int(pixel_cy - (pixel_h / 2))
            x2 = int(pixel_cx + (pixel_w / 2))
            y2 = int(pixel_cy + (pixel_h / 2))

            # --- Draw Box ---
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # print(f"Predicted Box: ({x1}, {y1}), ({x2}, {y2})")
            cv2.putText(
                current_frame,
                "Face Detected",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # Draw a rectangle
            cv2.rectangle(current_frame, (50, 50), (300, 200), (0, 255, 0), 2)

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


visualizer = VideoVisualizer(path="checkpoints/face_landmark_model_10.pth")

visualizer.open()
visualizer.close()
