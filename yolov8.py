from ultralytics import YOLO
import cv2

# Load the YOLOv8 face model
model = YOLO("yolov8n-face.pt")  # Use the smaller model for faster inference

# Open webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Get the frame width, height, and FPS from the webcam
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up the video writer to save the annotated video
output = cv2.VideoWriter("Videos/Web.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize the frame to a lower resolution to speed up inference
    frame_resized = cv2.resize(frame, (640, 320))  # Resize to 320x640 (you can adjust this)

    # Perform object detection on the resized frame
    results = model.predict(source=frame_resized, show=True, save=False)  # Don't save individual frames, only the output video

    # Get the annotated frame and write it to the output video
    annotated_frame = results[0].plot()  # Annotated frame with boxes, labels, etc.
    output.write(annotated_frame)  # Save the annotated frame to the video file

    # Display the frame in a window
    cv2.imshow("Webcam - YOLOv8 Detection", annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
output.release()
cv2.destroyAllWindows()
