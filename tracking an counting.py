import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize list of trackers
trackers = []
tracker_ids = []
next_id = 0

# Tracker initialization function
def create_tracker():
    return cv2.TrackerMIL_create()  # Use MIL tracker

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Prepare the frame for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detection results
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        detection = out.reshape(-1, 85)  # Reshape detection output
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                b_w = int(obj[2] * width)
                b_h = int(obj[3] * height)
                x = int(center_x - b_w / 2)
                y = int(center_y - b_h / 2)
                boxes.append([x, y, b_w, b_h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    if len(boxes) > 0:
        # Apply Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indexes) > 0:
            indexes = indexes.flatten()

            # Initialize new trackers with new detections
            new_trackers = []
            new_tracker_ids = []
            for i in indexes:
                bbox = boxes[i]
                tracker = create_tracker()  # Use the MIL tracker
                tracker.init(frame, tuple(bbox))
                new_trackers.append(tracker)
                new_tracker_ids.append(next_id)
                next_id += 1

            # Update all trackers and draw bounding boxes
            for i, tracker in enumerate(trackers):
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {tracker_ids[i]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Replace old trackers with new trackers
            trackers = new_trackers
            tracker_ids = new_tracker_ids

    # Display the resulting frame
    cv2.imshow("Image", frame)

    # Break the loop when 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
