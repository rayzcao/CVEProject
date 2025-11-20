import cv2
from ultralytics import YOLO
import numpy as np

def track_soccer_ball(video_path, output_path):
    # 1. Load the YOLOv8 model
    # 'yolov8n.pt' is the "nano" model (fastest). 
    # If detection is poor, try 'yolov8m.pt' (medium) or 'yolov8x.pt' (extra large).
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt') 

    # 2. Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties for the output file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 3. Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Run YOLO tracking on the frame
        # persist=True tells the model to track objects between frames
        results = model.track(frame, persist=True, verbose=False)

        # 5. Process detection results
        # The COCO dataset class ID for 'sports ball' is 32
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                
                # Filter: Only look for Class ID 32 (Sports Ball)
                if cls_id == 32:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Calculate center of the ball
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Calculate radius (half of the largest dimension of the box)
                    radius = int(max(x2 - x1, y2 - y1) / 2)

                    # 6. Draw the circle
                    # Color: Bright Green (BGR format), Thickness: 3
                    cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 3)
                    
                    # Optional: Add a label
                    cv2.putText(frame, "Ball", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Write the processed frame to output video
        out.write(frame)

        # Display the frame (optional, press 'q' to quit)
        cv2.imshow('Soccer Ball Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Output saved to {output_path}")

# --- Usage ---
# Replace 'soccer_match.mp4' with the path to your video file
if __name__ == "__main__":
    track_soccer_ball('vid.mp4', 'output_tracked.mp4')