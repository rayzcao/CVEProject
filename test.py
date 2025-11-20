import cv2
import sys
import numpy as np

# --- CONFIGURATION AREA ---
# Adjust these if the "Debug Mask" window doesn't show the ball clearly white.
# For a standard white ball in daylight:
LOWER_COLOR = np.array([0, 0, 200]) 
UPPER_COLOR = np.array([180, 50, 255])
MIN_RADIUS = 5
MAX_RADIUS = 25 # IMPORTANT: Reduce this if it selects players, increase if ball is close to camera
# ---------------------------

def find_ball_advanced(frame, roi=None):
    """
    Searches for a ball-like object. If roi is provided, searches only within that region.
    Returns (global_bbox, center_point) or (None, None).
    """
    # If an ROI is active, crop the frame
    if roi is not None:
        x_off, y_off, w_roi, h_roi = roi
        # Ensure ROI is within frame boundaries
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x_off), max(0, y_off)
        x2, y2 = min(w_img, x_off + w_roi), min(h_img, y_off + h_roi)
        search_area = frame[y1:y2, x1:x2]
        if search_area.size == 0: return None, None
        roi_offset = (x1, y1)
    else:
        search_area = frame
        roi_offset = (0, 0)

    blurred = cv2.GaussianBlur(search_area, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    # Clean up noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Show what we are searching (for debugging purposes)
    if roi is None:
         cv2.imshow("Debug Mask (Full Screen)", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_score = 0
    best_candidate = (None, None) # Initialize as a tuple

    for c in contours:
        area = cv2.contourArea(c)
        if area < 10: continue
        
        ((cx, cy), radius) = cv2.minEnclosingCircle(c)
        if radius < MIN_RADIUS or radius > MAX_RADIUS: continue

        x, y, w, h = cv2.boundingRect(c)

        # --- ADVANCED CHECKS ---
        # 1. Aspect Ratio: A ball should be roughly square (1.0)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.7 or aspect_ratio > 1.3: continue

        # 2. Circularity: 4*pi*area/perimeter^2. Perfect circle is 1.0
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.6: continue

        # 3. Solidity: Area of contour / Area of bounding box. Circle is roughly pi/4 (0.785)
        solidity = area / float(w * h)
        if solidity < 0.5: continue 

        # Score candidates based on how "circular" and "solid" they are
        score = circularity * solidity
        if score > best_score:
            best_score = score
            # Adjust coordinates back to global frame if we used ROI
            global_x = int(x + roi_offset[0])
            global_y = int(y + roi_offset[1])
            best_candidate = ((global_x, global_y, w, h), (int(cx + roi_offset[0]), int(cy + roi_offset[1])))

    return best_candidate

def create_tracker():
    try: return cv2.TrackerCSRT_create()
    except: return cv2.TrackerCSRT()

def run_smart_tracker(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return

    # --- 1. Scrubber for start frame ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow("Setup")
    cv2.createTrackbar('Frame', "Setup", 0, total_frames - 1, lambda x: None)
    start_frame = None
    while True:
        pos = cv2.getTrackbarPos('Frame', "Setup")
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow("Setup", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == ord(' '): 
            start_frame = frame.copy()
            break
        if k == ord('q'): return
    cv2.destroyWindow("Setup")
    if start_frame is None: return

    # --- 2. Select Object ---
    bbox = cv2.selectROI("Select Ball", start_frame, False, True)
    cv2.destroyWindow("Select Ball")
    if all(v == 0 for v in bbox): return

    tracker = create_tracker()
    tracker.init(start_frame, bbox)
    last_good_bbox = bbox

    while True:
        ret, frame = cap.read()
        if not ret: break

        success, box = tracker.update(frame)

        if success:
            last_good_bbox = box
            (x, y, w, h) = [int(v) for v in box]
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            radius = int((w + h) / 4)
            
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            # --- LOST SIGNAL: INTELLIGENT RE-ACQUISITION ---
            cv2.putText(frame, "Signal Lost", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 1. Try searching LOCALLY first (2x size of last known position)
            lx, ly, lw, lh = [int(v) for v in last_good_bbox]
            margin_x, margin_y = int(lw * 2), int(lh * 2)
            roi_local = (lx - margin_x, ly - margin_y, lw + 2*margin_x, lh + 2*margin_y)
            
            # Draw search area for debug
            cv2.rectangle(frame, (roi_local[0], roi_local[1]), (roi_local[0]+roi_local[2], roi_local[1]+roi_local[3]), (255,0,0), 1)
            
            candidate_bbox, _ = find_ball_advanced(frame, roi=roi_local)

            if candidate_bbox is None:
                 # 2. If local fails, search GLOBALLY
                 cv2.putText(frame, "Scanning Full Screen...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                 candidate_bbox, _ = find_ball_advanced(frame, roi=None)

            if candidate_bbox:
                print("Ball re-acquired!")
                tracker = create_tracker()
                tracker.init(frame, candidate_bbox)
                last_good_bbox = candidate_bbox
                # Flash yellow to indicate re-acquisition
                (cx, cy, cw, ch) = candidate_bbox
                cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (0, 255, 255), 3)

        cv2.imshow("Smart Tracker", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_smart_tracker('vid.mp4')