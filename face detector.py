import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor


class FaceDetector:
    """Enhanced face detection system with multiple detection models and features."""

    def __init__(self):
        # Load multiple face detectors for better accuracy
        self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Load DNN-based face detector (more accurate but slower)
        model_file = os.path.join(os.path.dirname(__file__), "models", "deploy.prototxt")
        weights_file = os.path.join(os.path.dirname(__file__), "models", "res10_300x300_ssd_iter_140000.caffemodel")

        # Check if DNN model files exist before loading
        self.use_dnn = False
        if os.path.exists(model_file) and os.path.exists(weights_file):
            self.dnn_face_detector = cv2.dnn.readNetFromCaffe(model_file, weights_file)
            self.use_dnn = True

        # Parameters for detection (can be adjusted)
        self.detection_method = "haar"  # Options: "haar", "dnn", "both"
        self.confidence_threshold = 0.5
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_face_size = (30, 30)

        # Options for display
        self.show_fps = True
        self.show_face_count = True
        self.draw_landmarks = False
        self.blur_faces = False

        # For fps calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0

        # Initialize face landmark detector
        if self.draw_landmarks:
            self.landmark_detector = cv2.face.createFacemarkLBF()
            landmark_model = os.path.join(os.path.dirname(__file__), "models", "lbfmodel.yaml")
            if os.path.exists(landmark_model):
                self.landmark_detector.loadModel(landmark_model)
            else:
                self.draw_landmarks = False

    def detect_faces_haar(self, gray_img):
        """Detect faces using Haar cascade classifier."""
        return self.haar_cascade.detectMultiScale(
            gray_img,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size
        )

    def detect_faces_dnn(self, frame):
        """Detect faces using DNN-based detector."""
        if not self.use_dnn:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.dnn_face_detector.setInput(blob)
        detections = self.dnn_face_detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                faces.append((x, y, x2 - x, y2 - y))

        return faces

    def detect_faces(self, frame):
        """Detect faces using selected method."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.detection_method == "haar":
            return self.detect_faces_haar(gray), gray
        elif self.detection_method == "dnn" and self.use_dnn:
            return self.detect_faces_dnn(frame), gray
        elif self.detection_method == "both" and self.use_dnn:
            # Use both methods and combine results
            haar_faces = self.detect_faces_haar(gray)
            dnn_faces = self.detect_faces_dnn(frame)
            return list(haar_faces) + list(dnn_faces), gray
        else:
            return self.detect_faces_haar(gray), gray

    def draw_face_landmarks(self, frame, gray, faces):
        """Draw facial landmarks if enabled and model is available."""
        if not self.draw_landmarks:
            return

        # Convert faces to tuple format expected by landmark detector
        faces_for_landmarks = [(x, y, w, h) for (x, y, w, h) in faces]
        if faces_for_landmarks:
            _, landmarks = self.landmark_detector.fit(gray, np.array(faces_for_landmarks))
            for face_landmarks in landmarks:
                for landmark in face_landmarks[0]:
                    pt = tuple(map(int, landmark))
                    cv2.circle(frame, pt, 1, (0, 255, 255), -1)

    def process_frame(self, frame):
        """Process a single frame for face detection."""
        # Calculate FPS
        if self.show_fps:
            self.new_frame_time = time.time()
            fps = 1 / (self.new_frame_time - self.prev_frame_time + 0.001)
            self.prev_frame_time = self.new_frame_time
            fps = int(fps)

        # Detect faces
        faces, gray = self.detect_faces(frame)

        # Process detected faces
        for (x, y, w, h) in faces:
            if self.blur_faces:
                # Apply blur to the face region
                face_region = frame[y:y + h, x:x + w]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y:y + h, x:x + w] = blurred_face
            else:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw facial landmarks if enabled
        self.draw_face_landmarks(frame, gray, faces)

        # Add information overlay
        if self.show_fps:
            cv2.putText(frame, f"FPS: {fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self.show_face_count:
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def process_image(self, image_path):
        """Process an image file."""
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return

        try:
            image = cv2.imread(image_path)
            processed_image = self.process_frame(image)

            # Show the output
            cv2.imshow('Face Detection - Image', processed_image)

            # Add option to save the processed image
            print("Press 's' to save the image, any other key to close")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                output_path = f"detected_{os.path.basename(image_path)}"
                cv2.imwrite(output_path, processed_image)
                print(f"Image saved as {output_path}")

            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error processing image: {e}")

    def process_webcam(self):
        """Process webcam feed for real-time detection."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Set higher resolution if supported
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Face Detection Started - Press 'q' to quit")
        print("Controls:")
        print("  'h': Toggle between detection methods")
        print("  'l': Toggle facial landmarks")
        print("  'b': Toggle face blurring")
        print("  'f': Toggle FPS display")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)

                # Show the output video
                cv2.imshow('Face Detection - Webcam', processed_frame)

                # Handle key presses for options
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    # Cycle through detection methods
                    methods = ["haar", "dnn", "both"]
                    current_idx = methods.index(self.detection_method)
                    self.detection_method = methods[(current_idx + 1) % len(methods)]
                    print(f"Detection method changed to: {self.detection_method}")
                elif key == ord('l'):
                    self.draw_landmarks = not self.draw_landmarks
                    print(f"Facial landmarks: {'On' if self.draw_landmarks else 'Off'}")
                elif key == ord('b'):
                    self.blur_faces = not self.blur_faces
                    print(f"Face blurring: {'On' if self.blur_faces else 'Off'}")
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                    print(f"FPS display: {'On' if self.show_fps else 'Off'}")

        except Exception as e:
            print(f"Error in webcam processing: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def batch_process_images(self, folder_path):
        """Process multiple images in a folder using multi-threading."""
        if not os.path.exists(folder_path):
            print(f"Error: Folder not found at {folder_path}")
            return

        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print(f"No image files found in {folder_path}")
            return

        print(f"Processing {len(image_files)} images...")

        output_dir = os.path.join(folder_path, "detected")
        os.makedirs(output_dir, exist_ok=True)

        def process_single_image(img_path):
            try:
                image = cv2.imread(img_path)
                processed = self.process_frame(image)
                output_path = os.path.join(output_dir, os.path.basename(img_path))
                cv2.imwrite(output_path, processed)
                return True
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                return False

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_single_image, image_files))

        success_count = results.count(True)
        print(f"Successfully processed {success_count} out of {len(image_files)} images")
        print(f"Results saved in {output_dir}")


# Main execution
if __name__ == "__main__":
    detector = FaceDetector()

    print("Face Detection System")
    print("1. Detect faces in an image")
    print("2. Real-time detection using webcam")
    print("3. Batch process images in a folder")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        image_path = input("Enter image file path: ").strip()
        detector.process_image(image_path)
    elif choice == "2":
        detector.process_webcam()
    elif choice == "3":
        folder_path = input("Enter folder path containing images: ").strip()
        detector.batch_process_images(folder_path)
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")