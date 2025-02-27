# Face Detection System

An easy-to-use face detection application using OpenCV and Haar cascade classifiers.

## Features

- Image-based face detection: Detect faces in any image file
- Real-time webcam detection: Live face detection using your webcam
- Simple user interface: Choose between image or webcam mode via console
- Green rectangle highlighting for detected faces

## Requirements

- Python 3.6+
- OpenCV library (`cv2`)

## Installation

1. Make sure you have Python installed on your system
2. Install OpenCV:
```
pip install opencv-python
```
3. Download or clone this repository

## Usage

Run the script:
```
python face_detector.py
```

The program will prompt you to choose between:
- `image`: For detecting faces in a single image file
- `webcam`: For real-time face detection using your webcam

### Image Mode
If you select `image` mode, you'll be asked to enter the path to your image file. The program will display the image with rectangles around detected faces.

### Webcam Mode
If you select `webcam` mode, your webcam will turn on and display a live feed with real-time face detection. Press 'q' to exit.

## How It Works

This program uses the OpenCV library and pre-trained Haar cascade classifiers to detect faces. Haar cascade is a machine learning-based approach where a cascade function is trained on positive images (with faces) and negative images (without faces).

## Code Overview

```python
import cv2

# Load pre-trained face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function for detecting faces in an image
def detect_faces_in_image(image_path):
    image = cv2.imread(image_path)  # Load image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Show the output image
    cv2.imshow('Face Detection - Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function for real-time face detection using webcam
def detect_faces_webcam():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Show the output video
        cv2.imshow('Face Detection - Webcam', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Choose mode: Image or Webcam
mode = input("Enter 'image' for face detection in an image or 'webcam' for real-time detection: ").strip().lower()

if mode == 'image':
    image_path = input("Enter image file path: ").strip()
    detect_faces_in_image(image_path)
elif mode == 'webcam':
    detect_faces_webcam()
else:
    print("Invalid option. Please enter 'image' or 'webcam'.")
```

## Planned Future Improvements

1. **Multiple Detection Methods**:
   - Add DNN-based face detector for improved accuracy
   - Allow users to switch between detection methods

2. **Performance Enhancements**:
   - Add FPS counter to monitor performance
   - Implement multi-threading for faster processing

3. **Additional Features**:
   - Face count display
   - Facial landmark detection
   - Face blurring option (for privacy)
   - Keyboard shortcuts for controlling the application

4. **User Experience**:
   - Ability to save processed images
   - Adjustable detection parameters via UI
   - Interactive controls in webcam mode

5. **Batch Processing**:
   - Support for processing multiple images at once
   - Directory output for batch results

6. **Advanced Recognition**:
   - Face recognition capabilities (identify specific individuals)
   - Age and gender estimation
   - Emotion detection

## Limitations

- The Haar cascade classifier may produce false positives or miss faces in certain lighting conditions
- Performance may vary based on image quality, face orientation, and occlusion
- Limited to frontal face detection by default

## License

This project is available under the MIT License.

## Acknowledgments

- OpenCV library and community
- Haar cascade classifiers
