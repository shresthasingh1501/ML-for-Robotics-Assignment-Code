import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use YOLOv8n model (or larger if needed)

# Define video capture
cap = cv2.VideoCapture(0)  # 0 for webcam, or provide video file path

# Screen and size parameters
LEFT_BORDER_THRESHOLD = 50  # Pixels from the left border
RIGHT_BORDER_THRESHOLD = 50  # Pixels from the right border
FULL_POWER_THRESHOLD = 0.2   # Percentage of frame area for medium size bounding box
STOP_MOTOR_THRESHOLD = 0.5  # Percentage of frame area for a large bounding box

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame)

    # Ensure results contain bounding boxes
    if len(results) > 0 and hasattr(results[0], 'boxes'):
        # Extract bounding boxes for "person" class (class 0)
        person_boxes = [box for box in results[0].boxes if int(box.cls[0]) == 0]

        # Define frame dimensions and area for thresholds
        frame_height, frame_width, _ = frame.shape
        frame_area = frame_height * frame_width

        # Initialize command as "Power Motor" (default if no other condition met)
        command = "Power Motor" 

        for box in person_boxes:
            # Extract bounding box details
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

            # Calculate the width, height, and area of the bounding box
            box_width = x_max - x_min
            box_height = y_max - y_min
            box_area = box_width * box_height


            # Check for a large bounding box first (highest priority)
            if box_area >= STOP_MOTOR_THRESHOLD * frame_area:
                command = "Stop Motor"
            else:  # Only check other conditions if the bounding box is NOT too large
                if x_min <= LEFT_BORDER_THRESHOLD:
                    command = "Low Power Right Motor"
                elif x_max >= (frame_width - RIGHT_BORDER_THRESHOLD):
                    command = "Low Power Left Motor"
                elif box_area < FULL_POWER_THRESHOLD * frame_area: #prioritize this over default power motor
                    command = "Full Power Both Motors"

            # Draw bounding box and display command
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, command, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            break  # Process only the first detected person box

    # If no person is detected the for loop is skipped and the initialized command is shown
    else:
        cv2.putText(frame, command, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with bounding box and command
    cv2.imshow("YOLOv8 Tracking", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
