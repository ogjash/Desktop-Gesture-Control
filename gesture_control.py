import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import subprocess
import os

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# System control helpers

def control_media(action):
    """Control media playback"""
    if action == "play_pause":
        pyautogui.press("playpause")
    elif action == "next":
        pyautogui.press("nexttrack")
    elif action == "previous":
        pyautogui.press("prevtrack")

def adjust_volume(direction):
    """Adjust system volume"""
    if direction == "up":
        pyautogui.press("volumeup")
    elif direction == "down":
        pyautogui.press("volumedown")
    elif direction == "mute":
        pyautogui.press("volumemute")

def mouse_click():
    """Simulate mouse click"""
    pyautogui.click()

def move_cursor(x, y, frame_width, frame_height):
    """Move cursor based on normalized hand coordinates with enhanced smoothing to eliminate shakiness"""
    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    
    # Convert the normalized coordinates to screen coordinates
    screen_x = int(x * screen_width)
    screen_y = int(y * screen_height)
    
    # Get current position
    current_x, current_y = pyautogui.position()
    
    # Enhanced smoothing with higher factor for greater stability
    smoothing_factor = 0.85  # Higher = more stable but less responsive
    
    smoothed_x = int(current_x * smoothing_factor + screen_x * (1 - smoothing_factor))
    smoothed_y = int(current_y * smoothing_factor + screen_y * (1 - smoothing_factor))
    
    # Increased noise threshold to prevent small unintentional movements
    noise_threshold = 6  # Increased from 3 to further reduce jitter
    if abs(smoothed_x - current_x) < noise_threshold and abs(smoothed_y - current_y) < noise_threshold:
        return  # Skip movement if it's below threshold
    
    # Additional stability: round to reduce micro-movements
    smoothed_x = round(smoothed_x / 2) * 2
    smoothed_y = round(smoothed_y / 2) * 2
    
    # Move the cursor
    pyautogui.moveTo(smoothed_x, smoothed_y)

def press_windows_key():
    """Simulate pressing the Windows key"""
    pyautogui.press('win')

def keyboard_typing(key):
    """Simulate keyboard typing"""
    pyautogui.press(key)

def scroll(direction):
    """Scroll page"""
    if direction == "up":
        pyautogui.scroll(10)
    elif direction == "down":
        pyautogui.scroll(-10)

# Gesture recognition and mapping
class GestureDetector:
    def __init__(self):
        self.prev_gesture = None
        self.gesture_start_time = 0
        self.gesture_delay = 1.0  # seconds to wait before recognizing same gesture again
        self.keyboard_mode = False
        self.keyboard_letters = {
            "play_pause": "c",
            "next_track": "d",
            "volume_up": "e",
            "volume_down": "f",
            "mouse_click": "g",
            "scroll_up": "h",
            "scroll_down": "i",
            "windows_key": "j"
        }
        self.cursor_mode = False
        self.frame_width = 0
        self.frame_height = 0
        
    def detect_gesture(self, hand_landmarks, frame_width, frame_height):
        if not hand_landmarks:
            return None
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        landmarks = hand_landmarks.landmark
        
        # Calculate fingertip positions
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
        
        # Calculate finger knuckle positions
        index_knuckle = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_knuckle = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_knuckle = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_knuckle = landmarks[mp_hands.HandLandmark.PINKY_MCP]
        wrist = landmarks[mp_hands.HandLandmark.WRIST]
        
        # Check if fingers are extended
        thumb_extended = thumb_tip.x < index_knuckle.x
        index_extended = index_tip.y < index_knuckle.y
        middle_extended = middle_tip.y < middle_knuckle.y
        ring_extended = ring_tip.y < ring_knuckle.y
        pinky_extended = pinky_tip.y < pinky_knuckle.y
        
        # Check for pinch gesture (thumb and index finger close together)
        thumb_index_distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        pinch_detected = thumb_index_distance < 0.05
        
        # Check for C shape gesture (all fingers and thumb forming a C)
        if thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            # Calculate distance between thumb and index tips - should be farther than pinch but closer than fully apart
            thumb_index_distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            # Check if the distance indicates a C shape (not a pinch, not fully extended)
            c_shape_detected = 0.07 < thumb_index_distance < 0.15
            
            # Also check the angle to ensure it's really a C shape - thumb and index should be facing each other
            if c_shape_detected:
                self.cursor_mode = not self.cursor_mode
                return "toggle_cursor_mode"
        
        # In cursor mode, use index finger movement to control cursor
        if self.cursor_mode:
            # Only track index finger position if extended
            if index_extended:
                # Use absolute positioning for cursor movement
                move_cursor(index_tip.x, index_tip.y, frame_width, frame_height)
                
                # Detect pinch for clicking
                if pinch_detected:
                    return "mouse_click"
                
            return "cursor_tracking"
            
        # Windows key gesture: Thumb, index, and pinky extended (like the "I love you" sign)
        if thumb_extended and index_extended and not middle_extended and not ring_extended and pinky_extended:
            return "windows_key"
        
        # If in keyboard mode, interpret gestures as keyboard inputs
        if self.keyboard_mode:
            # Simple version: use pre-defined gestures as different letters
            
            # A - All fingers extended
            if thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
                return "key_a"
            
            # B - Index and middle finger extended
            if not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
                return "key_b"
                
            # C - Only thumb extended
            if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                return "key_c"
            
            # D - Thumb and index extended
            if thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
                return "key_d"
            
            # Space - Flat palm with fingers together
            if index_extended and middle_extended and ring_extended and pinky_extended:
                finger_spread = abs(index_tip.x - pinky_tip.x)
                if finger_spread < 0.1:  # Fingers together
                    return "key_space"
            
            # Backspace - Swipe left
            if thumb_tip.x > wrist.x and index_extended:
                return "key_backspace"
        
        else:
            # Normal system control gestures
            
            # Play/Pause: Only thumb extended
            if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                return "play_pause"
            
            # Next track: Thumb and index extended in a pinch
            if thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
                if thumb_index_distance < 0.05: # pinch
                    return "next_track"
            
            # Volume up: Index, middle, ring extended
            if not thumb_extended and index_extended and middle_extended and ring_extended and not pinky_extended:
                return "volume_up"
            
            # Volume down: Index and pinky extended
            if not thumb_extended and index_extended and not middle_extended and not ring_extended and pinky_extended:
                return "volume_down"
            
            # Scroll up: All fingers extended upward
            if not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
                return "scroll_up"
            
            # Scroll down: All fingers pointing down
            if not thumb_extended and index_tip.y > index_knuckle.y and middle_tip.y > middle_knuckle.y and ring_tip.y > ring_knuckle.y and pinky_tip.y > pinky_knuckle.y:
                return "scroll_down"
        
        return None
    
    def process_gesture(self, gesture):
        current_time = time.time()
        
        # Handle toggle mode gestures and cursor tracking immediately
        if gesture in ["toggle_keyboard_mode", "toggle_cursor_mode", "cursor_tracking"]:
            self.prev_gesture = gesture
            self.gesture_start_time = current_time
            return
        
        # Prevent repeated triggers for the same gesture
        if gesture == self.prev_gesture and current_time - self.gesture_start_time < self.gesture_delay:
            return
        
        if gesture != self.prev_gesture:
            self.prev_gesture = gesture
            self.gesture_start_time = current_time
            
            # Execute action based on gesture
            if gesture == "windows_key":
                press_windows_key()
            elif gesture == "mouse_click":
                mouse_click()
            elif gesture.startswith("key_"):
                if gesture == "key_space":
                    keyboard_typing("space")
                elif gesture == "key_backspace":
                    keyboard_typing("backspace")
                else:
                    # Extract the letter from the gesture name
                    letter = gesture.split("_")[1]
                    keyboard_typing(letter)
            elif gesture == "play_pause":
                control_media("play_pause")
            elif gesture == "next_track":
                control_media("next")
            elif gesture == "volume_up":
                adjust_volume("up")
            elif gesture == "volume_down":
                adjust_volume("down")
            elif gesture == "scroll_up":
                scroll("up")
            elif gesture == "scroll_down":
                scroll("down")

# Main function
def main():
    cap = cv2.VideoCapture(0)
    gesture_detector = GestureDetector()
    
    # Text positions
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    thickness = 2
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        # Get frame dimensions
        frame_height, frame_width, _ = image.shape
        
        # Convert image to RGB and process with MediaPipe
        image = cv2.flip(image, 1)  # Flip for mirror effect
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Detect and process gesture
                gesture = gesture_detector.detect_gesture(hand_landmarks, frame_width, frame_height)
                if gesture:
                    gesture_detector.process_gesture(gesture)
                    if gesture not in ["cursor_tracking"]:  # Don't display "cursor_tracking" as a detected gesture
                        cv2.putText(image, f"Gesture: {gesture}", (10, 30), 
                                    fontFace, fontScale, (0, 255, 0), thickness)
        
        # Display mode information
        if gesture_detector.cursor_mode:
            mode_text = "Mode: CURSOR CONTROL"
        elif gesture_detector.keyboard_mode:
            mode_text = "Mode: KEYBOARD"
        else:
            mode_text = "Mode: SYSTEM CONTROL"
            
        cv2.putText(image, mode_text, (10, 60), fontFace, fontScale, (0, 255, 255), thickness)
        
        # Display instructions
        y_offset = 90
        
        if gesture_detector.cursor_mode:
            instructions = [
                "Move index finger: Control cursor movement (absolute positioning)",
                "Pinch index and thumb: Click",
                "C shape with all fingers: Exit cursor mode"
            ]
        elif gesture_detector.keyboard_mode:
            instructions = [
                "Thumb+Index+Middle+Ring+Pinky: Letter A",
                "Index+Middle: Letter B",
                "Thumb only: Letter C",
                "Thumb+Index: Letter D",
                "Flat palm with fingers together: Space",
                "Swipe left: Backspace",
                "Pinch + extend other fingers: Toggle keyboard mode"
            ]
        else:
            instructions = [
                "Thumb only: Play/Pause",
                "Thumb & index pinch: Next track",
                "Thumb & index & pinky extended: Windows key",
                "Index, middle & ring: Volume up",
                "Index & pinky: Volume down",
                "All fingers up: Scroll up",
                "All fingers down: Scroll down",
                "C shape with all fingers: Enter cursor mode",
                "Pinch + extend other fingers: Toggle keyboard mode"
            ]
        
        for instruction in instructions:
            cv2.putText(image, instruction, (10, y_offset), 
                        fontFace, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Display the image
        cv2.imshow('Gesture Control', image)
        
        if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
