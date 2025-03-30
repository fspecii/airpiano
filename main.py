import cv2
import mediapipe as mp
import numpy as np
import math
import time
import pygame
import pygame.midi
import json
from typing import List, Tuple, Optional, Dict, Any, Set
from collections import defaultdict

# --- Configuration Constants ---

# Webcam/Display
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# MediaPipe Hands
MP_HANDS_MAX_HANDS = 2
MP_HANDS_MIN_DET_CONF = 0.7
MP_HANDS_MIN_TRACK_CONF = 0.6

# --- Piano Layout & Appearance ---
PIANO_X = 50              # X position of the start of the piano bottom edge
PIANO_Y = FRAME_HEIGHT - 50  # Y position of the *bottom* edge of the white keys (closer to user)
PIANO_WIDTH = FRAME_WIDTH - 2 * PIANO_X # Width of the piano bottom edge
WHITE_KEY_HEIGHT = 180     # Visual height of the white keys (vertical distance)
BLACK_KEY_HEIGHT_FACTOR = 0.6 # Black key height relative to white key height
BLACK_KEY_WIDTH_FACTOR = 0.55  # How wide black keys are compared to white keys
NUM_OCTAVES = 3            # Number of octaves to display
START_MIDI_NOTE = 48       # Starting note (e.g., 48=C3, 60=C4)
PIANO_ROTATION = 0       # Rotation angle in degrees (0: normal, 180: upside down)

# Pseudo-3D Appearance
PERSPECTIVE_FACTOR = 0.9   # How much narrower the top edge is (0.0 to 1.0). 1.0 = no perspective
KEY_FRONT_HEIGHT = 12      # Height of the darker "front face" of the keys
KEY_TOP_Y_OFFSET = 15      # How much higher the black keys sit visually on white keys

# Key Press Detection
# How far *into* the key vertically (from the top edge) a fingertip must be
KEY_PRESS_Y_ENTRY_FACTOR_WHITE = 0.10 # Relaxed slightly (was 0.15)
KEY_PRESS_Y_ENTRY_FACTOR_BLACK = 0.15 # Relaxed slightly (was 0.2)
# Optional: Z-Depth Press Threshold (Relative to Wrist)
PRESS_DEPTH_THRESHOLD = 0.035 # Relaxed slightly (was 0.04)
# PRESS_DEPTH_THRESHOLD = None # Disable Z-checking

# Debouncing: Require condition to hold for this many frames
DEBOUNCE_FRAMES_ON = 2        # Frames to confirm a press
DEBOUNCE_FRAMES_OFF = 2       # Frames to confirm a release

# MIDI Configuration
MIDI_INSTRUMENT = 0           # 0=Acoustic Grand, 1=Bright Acoustic, 5=Electric Piano 1
MIDI_NOTE_VELOCITY = 90      # Loudness of piano notes (0-127)
MIDI_CHECK_INTERVAL = 15.0    # Check MIDI health every 15 seconds

# Note names for display
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SHOW_NOTE_LABELS = True       # Display note names on white keys

# Colors
COLOR_WHITE_KEY = (255, 255, 255)
COLOR_WHITE_KEY_TOP = (255, 255, 255) # Slightly brighter top for gradient
COLOR_WHITE_KEY_BOTTOM = (235, 235, 235) # Slightly darker bottom for gradient
COLOR_WHITE_KEY_FRONT = (200, 200, 200) # Darker front face
COLOR_BLACK_KEY = (20, 20, 20)
COLOR_BLACK_KEY_FRONT = (10, 10, 10)
COLOR_KEY_OUTLINE = (80, 80, 80)
COLOR_PRESSED_WHITE_FILL_TOP = (180, 220, 255) # Light blue gradient top
COLOR_PRESSED_WHITE_FILL_BOTTOM = (150, 200, 245) # Slightly darker blue bottom
COLOR_PRESSED_BLACK_FILL = (100, 180, 230)
COLOR_PRESSED_OUTLINE = (0, 120, 255)     # Blue outline for pressed
COLOR_TEXT = (255, 255, 255)
COLOR_CALIBRATE_TEXT = (0, 255, 0)
COLOR_ERROR_TEXT = (255, 0, 0)
COLOR_INFO_TEXT = (255, 220, 0)
COLOR_FINGERTIP = (255, 0, 255) # Magenta for fingertips
COLOR_NOTE_LABEL = (50, 50, 50)
COLOR_FINGERTIP_PRESS_EFFECT = (255, 255, 0) # Yellow circle on press

# UI Enhancement Colors & Constants
COLOR_UI_PANEL_BG = (30, 30, 50, 180)  # Semi-transparent dark blue background for panels
COLOR_UI_PANEL_BORDER = (100, 140, 200)  # Light blue border for panels
COLOR_UI_HEADER = (140, 200, 255)  # Light blue for headers
COLOR_UI_SECTION_TITLE = (200, 220, 255)  # Very light blue for section titles
COLOR_UI_HIGHLIGHT = (255, 220, 100)  # Yellowish highlight
COLOR_UI_KEY_LABEL = (220, 220, 220)  # Light gray for key labels

# Panel styling
UI_PANEL_PADDING = 15  # Padding inside panels
UI_PANEL_CORNER_RADIUS = 10  # Rounded corner radius
UI_PANEL_BORDER_THICKNESS = 2  # Border thickness
UI_PANEL_OPACITY = 0.85  # Panel opacity

# Game States
STATE_POSITIONING = 0      # Initial positioning of piano
STATE_CALIBRATING = 1
STATE_RUNNING = 2

# Fingertip Landmark Indices
FINGERTIP_INDICES = [
    mp.solutions.hands.HandLandmark.THUMB_TIP,
    mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
    mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
    mp.solutions.hands.HandLandmark.PINKY_TIP
]
WRIST_INDEX = mp.solutions.hands.HandLandmark.WRIST

# --- Helper Functions ---
def get_note_name(midi_note: int) -> str:
    """Converts MIDI note number to note name (e.g., C4)."""
    if not 0 <= midi_note <= 127:
        return "N/A"
    octave = midi_note // 12 - 1
    note_index = midi_note % 12
    return f"{NOTE_NAMES[note_index]}{octave}"

def draw_panel(image: np.ndarray, x: int, y: int, width: int, height: int, 
               title: Optional[str] = None, alpha: float = UI_PANEL_OPACITY) -> None:
    """Draws a semi-transparent panel with optional title."""
    # Create overlay for semi-transparent panel
    overlay = image.copy()
    
    # Draw filled rectangle with rounded corners if possible
    try:
        # Use rounded rectangle if OpenCV version supports it
        cv2.rectangle(overlay, (x, y), (x + width, y + height), 
                      COLOR_UI_PANEL_BG[:3], -1, 
                      lineType=cv2.LINE_AA, 
                      shift=0)
    except:
        # Fallback to regular rectangle
        cv2.rectangle(overlay, (x, y), (x + width, y + height), 
                      COLOR_UI_PANEL_BG[:3], -1)
    
    # Apply transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw border
    cv2.rectangle(image, (x, y), (x + width, y + height), 
                  COLOR_UI_PANEL_BORDER, UI_PANEL_BORDER_THICKNESS)
    
    # Draw title if provided
    if title:
        # Title background
        cv2.rectangle(image, (x, y), (x + width, y + 30), 
                     COLOR_UI_PANEL_BORDER, -1)
        
        # Title text
        (text_w, text_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(image, title, 
                    (x + (width - text_w) // 2, y + 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_UI_KEY_LABEL, 2, cv2.LINE_AA)

def draw_enhanced_text(image: np.ndarray, text: str, position: Tuple[int, int], 
                       font_scale: float = 0.7, color: Tuple[int, int, int] = COLOR_TEXT, 
                       thickness: int = 1, shadow: bool = True) -> None:
    """Draws text with an optional shadow for better visibility."""
    x, y = position
    if shadow:
        # Draw shadow
        cv2.putText(image, text, (x+2, y+2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # Draw main text
    cv2.putText(image, text, (x, y), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# --- Classes ---

class HandTracker:
    """Handles MediaPipe hand detection and landmark extraction."""
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=MP_HANDS_MAX_HANDS,
            min_detection_confidence=MP_HANDS_MIN_DET_CONF,
            min_tracking_confidence=MP_HANDS_MIN_TRACK_CONF
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.results = None

    def process_frame(self, image_rgb: np.ndarray) -> None:
        """Processes the RGB frame to find hands."""
        image_rgb.flags.writeable = False # Performance opt
        self.results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

    def get_landmarks(self, frame_width: int, frame_height: int) -> List[List[Tuple[int, int, float]]]:
        """Extracts scaled landmarks (x, y, z) for each detected hand."""
        all_landmarks = []
        # Use multi_hand_world_landmarks for potentially more stable Z relative to the hand center
        # But fall back to multi_hand_landmarks if world landmarks aren't available/reliable enough
        landmarks_source = self.results.multi_hand_landmarks # Use image coordinates for XY

        if self.results and landmarks_source:
            for i, hand_landmarks in enumerate(landmarks_source):
                landmarks = []
                # Get corresponding world landmarks just for Z depth if available
                hand_world_landmarks = None
                if self.results.multi_hand_world_landmarks and len(self.results.multi_hand_world_landmarks) > i:
                   hand_world_landmarks = self.results.multi_hand_world_landmarks[i]

                for j, lm in enumerate(hand_landmarks.landmark):
                    pixel_x = int(lm.x * frame_width)
                    pixel_y = int(lm.y * frame_height)

                    # Use world landmark Z if possible, otherwise use image landmark Z (less ideal)
                    z_depth = lm.z
                    if hand_world_landmarks and j < len(hand_world_landmarks.landmark):
                         # World landmarks Z is often negative, smaller means further away usually
                         # We might need to invert or scale it depending on desired behavior
                         # Using raw world Z relative to wrist's world Z might be better
                         z_depth = hand_world_landmarks.landmark[j].z

                    landmarks.append((pixel_x, pixel_y, z_depth))
                all_landmarks.append(landmarks)
        return all_landmarks


    def get_fingertips_and_wrists(self, landmarks_list: List[List[Tuple[int, int, float]]]) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
        """Extracts only fingertip and wrist coordinates."""
        fingertips = []
        wrists = []
        for hand_landmarks in landmarks_list:
            if not hand_landmarks or len(hand_landmarks) <= max(FINGERTIP_INDICES + [WRIST_INDEX]):
                continue # Skip if landmarks are incomplete for this hand

            wrist_data = hand_landmarks[WRIST_INDEX]
            wrists.append(wrist_data)
            for index in FINGERTIP_INDICES:
                 fingertips.append(hand_landmarks[index])

        return fingertips, wrists

    def draw_landmarks(self, image: np.ndarray) -> None:
        """Draws the detected hand landmarks and connections on the image."""
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw connections first (behind landmarks)
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())
                # Draw landmarks on top
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    None, # No connections here
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style())


class Piano:
    """Represents the virtual piano keyboard with pseudo-3D perspective."""
    def __init__(self, x, y_bottom, width, white_h, black_h_factor, black_w_factor,
                 perspective, key_front_h, key_top_y_offset, start_note, num_octaves):
        self.x_base = x
        self.y_bottom = y_bottom
        self.width_base = width
        self.white_key_h = white_h
        self.black_key_h = int(white_h * black_h_factor)
        self.black_w_factor = black_w_factor
        self.perspective_factor = perspective
        self.key_front_h = key_front_h
        self.key_top_y_offset = key_top_y_offset # Black keys visually higher
        self.start_note = start_note
        self.num_octaves = num_octaves
        self.rotation = PIANO_ROTATION  # Add rotation parameter

        # Initialize the keys list
        self.keys: List[Dict[str, Any]] = []
        self._recalculate_layout()

    def _recalculate_layout(self):
        """Calculates polygons and MIDI notes for all piano keys with perspective."""
        # For non-rotated piano
        if self.rotation == 0:
            self.y_top_white = self.y_bottom - self.white_key_h
            self.y_top_black = self.y_bottom - self.black_key_h - self.key_top_y_offset
        # For rotated piano (upside down)
        else:
            self.y_top_white = self.y_bottom + self.white_key_h
            self.y_top_black = self.y_bottom + self.black_key_h + self.key_top_y_offset

        # Store key data including polygons for drawing and hit detection
        # {'midi': int, 'is_white': bool, 'note_name': str,
        #  'poly_points': np.array, 'poly_top_y': int, 'poly_bottom_y': int,
        #  'front_rect': Optional[Tuple[int, int, int, int]]}
        self.keys = []
        self.rect = (self.x_base, self.y_top_white, self.width_base, self.white_key_h) # Overall bounding box approx
        self._calculate_layout()
        
    def update_position(self, x=None, y=None, rotation=None):
        """Updates the piano position and/or rotation."""
        if x is not None:
            self.x_base = x
        if y is not None:
            self.y_bottom = y
        if rotation is not None:
            self.rotation = rotation
        
        # Recalculate layout with new parameters
        self._recalculate_layout()
        
        # Log the current position
        print(f"Piano position updated: x={self.x_base}, y={self.y_bottom}, rotation={self.rotation}°")
    
    def update_size(self, width=None, height=None, num_octaves=None):
        """Updates the piano size."""
        if width is not None:
            self.width_base = width
        if height is not None:
            self.white_key_h = height
            self.black_key_h = int(height * BLACK_KEY_HEIGHT_FACTOR)
        if num_octaves is not None:
            self.num_octaves = num_octaves
            
        # Recalculate layout with new parameters
        self._recalculate_layout()
        
        # Log the current size
        print(f"Piano size updated: width={self.width_base}, height={self.white_key_h}, " +
              f"octaves={self.num_octaves}")
        
    def _calculate_layout(self):
        """Calculates polygons and MIDI notes for all piano keys with perspective."""
        self.keys = []
        num_notes = self.num_octaves * 12
        total_white_keys = 0
        notes_in_layout = []

        for i in range(num_notes):
            midi_note = self.start_note + i
            note_in_octave = midi_note % 12
            is_white = note_in_octave in [0, 2, 4, 5, 7, 9, 11]
            note_name = get_note_name(midi_note)
            notes_in_layout.append({'midi': midi_note, 'is_white': is_white, 'note_name': note_name})
            if is_white:
                total_white_keys += 1

        if total_white_keys == 0: return

        white_key_width_base = self.width_base / total_white_keys
        black_key_width_base = white_key_width_base * self.black_w_factor

        current_x_base = self.x_base
        white_key_indices = [] # Store index of white keys in self.keys

        # --- Pass 1: Create all key entries (initially placeholder rects) ---
        for i, note_info in enumerate(notes_in_layout):
            midi = note_info['midi']
            is_white = note_info['is_white']
            name = note_info['note_name']
            key_data: Dict[str, Any] = {
                'midi': midi, 'is_white': is_white, 'note_name': name,
                'poly_points': None, 'poly_top_y': 0, 'poly_bottom_y': 0,
                'front_rect': None
            }

            if is_white:
                w_base = white_key_width_base
                x_base = current_x_base
                
                if self.rotation == 0:  # Normal orientation
                    y_top = self.y_top_white
                    y_bot = self.y_bottom
                else:  # Rotated upside down
                    y_top = self.y_top_white
                    y_bot = self.y_bottom
                
                key_data.update({
                     'x_base': x_base, 'w_base': w_base,
                     'y_top': y_top, 'y_bot': y_bot,
                     'h': self.white_key_h
                })
                self.keys.append(key_data)
                white_key_indices.append(len(self.keys) - 1) # Store index
                current_x_base += w_base # Move base position for next white key
            else:
                # Black keys are initially positioned relative to the *previous* white key
                # We'll refine their position and polygon later
                if self.rotation == 0:  # Normal orientation
                    y_top = self.y_top_black
                    y_bot = self.y_bottom - self.key_top_y_offset  # Sits on white key
                else:  # Rotated upside down
                    y_top = self.y_top_black
                    y_bot = self.y_bottom + self.key_top_y_offset  # Sits on white key (but inverted)
                
                key_data.update({
                     'x_base': 0, # Placeholder
                     'w_base': black_key_width_base,
                     'y_top': y_top,
                     'y_bot': y_bot,
                     'h': self.black_key_h
                 })
                self.keys.append(key_data)


        # --- Pass 2: Calculate polygons and positions ---
        current_white_key_idx_in_all = -1
        white_key_counter = -1

        for i, key_data in enumerate(self.keys):
            is_white = key_data['is_white']
            x_base = key_data['x_base']
            w_base = key_data['w_base']
            y_top = key_data['y_top']
            y_bot = key_data['y_bot']

            # Calculate perspective width at the top
            w_top = w_base * self.perspective_factor
            x_offset_top = (w_base - w_top) / 2

            # Define polygon points
            if self.rotation == 0:  # Normal orientation
                # (bottom-left, bottom-right, top-right, top-left)
                pt1 = (int(x_base), y_bot)
                pt2 = (int(x_base + w_base), y_bot)
                pt3 = (int(x_base + x_offset_top + w_top), y_top)
                pt4 = (int(x_base + x_offset_top), y_top)
            else:  # Rotated upside down
                # For 180 rotation, we flip the order to: (top-left, top-right, bottom-right, bottom-left)
                pt1 = (int(x_base + x_offset_top), y_top)
                pt2 = (int(x_base + x_offset_top + w_top), y_top)
                pt3 = (int(x_base + w_base), y_bot)
                pt4 = (int(x_base), y_bot)

            poly_points = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)

            key_data['poly_points'] = poly_points
            key_data['poly_top_y'] = y_top
            key_data['poly_bottom_y'] = y_bot

            # Calculate front rectangle (a thin rect below the bottom edge of the polygon)
            if self.rotation == 0:  # Normal orientation
                front_y = y_bot
                front_h = self.key_front_h
                front_w = int(w_base)
                front_x = int(x_base)
            else:  # Rotated upside down
                front_y = y_bot - self.key_front_h  # Move up to be above the bottom edge
                front_h = self.key_front_h
                front_w = int(w_base)
                front_x = int(x_base)

            key_data['front_rect'] = (front_x, front_y, front_w, front_h)

            if is_white:
                white_key_counter += 1
                current_white_key_idx_in_all = i
            else: # Black Key - Position relative to the preceding white key
                if current_white_key_idx_in_all != -1:
                    prev_white_key = self.keys[current_white_key_idx_in_all]
                    # Center the black key over the boundary of the white keys, shifted back
                    black_x_base = prev_white_key['x_base'] + prev_white_key['w_base'] - (w_base / 2)
                    key_data['x_base'] = black_x_base

                    # Recalculate black key polygon based on its final position
                    w_top_b = w_base * self.perspective_factor
                    x_offset_top_b = (w_base - w_top_b) / 2
                    
                    if self.rotation == 0:  # Normal orientation
                        pt1_b = (int(black_x_base), y_bot)
                        pt2_b = (int(black_x_base + w_base), y_bot)
                        pt3_b = (int(black_x_base + x_offset_top_b + w_top_b), y_top)
                        pt4_b = (int(black_x_base + x_offset_top_b), y_top)
                    else:  # Rotated upside down
                        pt1_b = (int(black_x_base + x_offset_top_b), y_top)
                        pt2_b = (int(black_x_base + x_offset_top_b + w_top_b), y_top)
                        pt3_b = (int(black_x_base + w_base), y_bot)
                        pt4_b = (int(black_x_base), y_bot)
                    
                    key_data['poly_points'] = np.array([pt1_b, pt2_b, pt3_b, pt4_b], dtype=np.int32)
                    key_data['poly_top_y'] = y_top
                    key_data['poly_bottom_y'] = y_bot

                    # Black key front rectangle
                    if self.rotation == 0:  # Normal orientation
                        b_front_y = y_bot
                        b_front_h = self.key_front_h
                        b_front_w = int(w_base)
                        b_front_x = int(black_x_base)
                    else:  # Rotated upside down
                        b_front_y = y_bot - self.key_front_h
                        b_front_h = self.key_front_h
                        b_front_w = int(w_base)
                        b_front_x = int(black_x_base)
                    
                    key_data['front_rect'] = (b_front_x, b_front_y, b_front_w, b_front_h)


        # Sort keys for drawing: Draw all white keys first, then all black keys
        self.keys.sort(key=lambda k: not k['is_white'])
        print(f"Piano layout calculated: {len([k for k in self.keys if k['is_white']])} white, {len([k for k in self.keys if not k['is_white']])} black keys.")

    def draw(self, image: np.ndarray, pressed_notes: Set[int]):
        """Draws the piano keyboard with perspective on the image."""
        img_h, img_w = image.shape[:2]

        # Draw white keys first
        for key_info in self.keys:
            if key_info['is_white']:
                midi_note = key_info['midi']
                poly = key_info['poly_points']
                front_rect = key_info['front_rect']
                is_pressed = midi_note in pressed_notes

                # Determine colors based on state
                fill_color = COLOR_PRESSED_WHITE_FILL_TOP if is_pressed else COLOR_WHITE_KEY_TOP
                outline_color = COLOR_PRESSED_OUTLINE if is_pressed else COLOR_KEY_OUTLINE
                front_color = COLOR_WHITE_KEY_FRONT # Keep front color consistent
                outline_thickness = 2 if is_pressed else 1

                # Define gradient colors based on state
                if is_pressed:
                    top_color = COLOR_PRESSED_WHITE_FILL_TOP
                    bottom_color = COLOR_PRESSED_WHITE_FILL_BOTTOM
                else:
                    top_color = COLOR_WHITE_KEY_TOP
                    bottom_color = COLOR_WHITE_KEY_BOTTOM

                # Draw key front (behind the main polygon)
                if front_rect:
                    fx, fy, fw, fh = front_rect
                    # Clip front rect to screen bounds before drawing
                    fy_draw = max(0, min(fy, img_h - 1))
                    fh_draw = max(0, min(fh, img_h - fy_draw))
                    if fh_draw > 0:
                         cv2.rectangle(image, (fx, fy_draw), (fx + fw, fy_draw + fh_draw), front_color, -1)
                         cv2.rectangle(image, (fx, fy_draw), (fx + fw, fy_draw + fh_draw), COLOR_KEY_OUTLINE, 1) # Thin outline

                # Draw main key polygon with gradient (top surface)
                min_x, min_y, w, h = cv2.boundingRect(poly)
                # Need the visual top/bottom Y from key_info for correct gradient direction
                visual_top_y = min(key_info['poly_top_y'], key_info['poly_bottom_y'])
                visual_bottom_y = max(key_info['poly_top_y'], key_info['poly_bottom_y'])
                visual_h = abs(visual_bottom_y - visual_top_y)

                # Create a temporary mask for the polygon
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)

                # Apply gradient line by line within the mask
                clamped_min_y = max(0, min_y)
                clamped_max_y = min(min_y + h, img_h) # Use bounding box height for loop

                for y_line in range(clamped_min_y, clamped_max_y):
                    # Calculate interpolation factor based on visual height
                    interp_factor = (y_line - visual_top_y) / visual_h if visual_h > 0 else 0.0
                    interp_factor = np.clip(interp_factor, 0.0, 1.0) # Ensure factor is [0, 1]

                    # Interpolate color (bottom color corresponds to interp_factor=1)
                    line_color = (
                        int(top_color[0] * (1 - interp_factor) + bottom_color[0] * interp_factor),
                        int(top_color[1] * (1 - interp_factor) + bottom_color[1] * interp_factor),
                        int(top_color[2] * (1 - interp_factor) + bottom_color[2] * interp_factor)
                    )
                    # Draw the line, masked by the polygon shape
                    row_mask = mask[y_line, :]
                    indices = np.where(row_mask > 0)[0]
                    if len(indices) > 0:
                        start_x = indices[0]
                        end_x = indices[-1]
                        # Clip X coords too for safety
                        start_x_draw = max(0, min(start_x, img_w - 1))
                        end_x_draw = max(0, min(end_x, img_w - 1))
                        if end_x_draw >= start_x_draw:
                             cv2.line(image, (start_x_draw, y_line), (end_x_draw, y_line), line_color, 1)

                # Draw outline after gradient
                cv2.polylines(image, [poly], isClosed=True, color=outline_color, thickness=outline_thickness)

                # Draw note labels on white keys
                if SHOW_NOTE_LABELS and not is_pressed: # Avoid drawing on pressed keys for clarity
                    note_name = key_info['note_name']
                    # Position label near the visual bottom (closest to user) center of the key
                    label_x = int(key_info['x_base'] + key_info['w_base'] * 0.5)
                    label_y_base = key_info['poly_bottom_y'] if self.rotation == 0 else key_info['poly_top_y']
                    y_offset = -8 if self.rotation == 0 else 20 # Adjust offset based on rotation

                    (text_w, text_h), _ = cv2.getTextSize(note_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.putText(image, note_name, (label_x - text_w // 2, label_y_base + y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_NOTE_LABEL, 1, cv2.LINE_AA)


        # Draw black keys on top
        for key_info in self.keys:
            if not key_info['is_white']:
                midi_note = key_info['midi']
                poly = key_info['poly_points']
                front_rect = key_info['front_rect']
                is_pressed = midi_note in pressed_notes

                fill_color = COLOR_PRESSED_BLACK_FILL if is_pressed else COLOR_BLACK_KEY
                outline_color = COLOR_PRESSED_OUTLINE if is_pressed else COLOR_KEY_OUTLINE
                front_color = COLOR_BLACK_KEY_FRONT
                outline_thickness = 2 if is_pressed else 1

                # Draw key front
                if front_rect:
                     fx, fy, fw, fh = front_rect
                     # Clip front rect to screen bounds before drawing
                     fy_draw = max(0, min(fy, img_h - 1))
                     fh_draw = max(0, min(fh, img_h - fy_draw))
                     if fh_draw > 0:
                          # Shift front slightly down visually relative to main key surface for depth
                          draw_offset = 2 if self.rotation == 0 else -2
                          cv2.rectangle(image, (fx, fy_draw + draw_offset), (fx + fw, fy_draw + fh_draw + draw_offset), front_color, -1)
                          cv2.rectangle(image, (fx, fy_draw + draw_offset), (fx + fw, fy_draw + fh_draw + draw_offset), COLOR_KEY_OUTLINE, 1)

                # Draw main key polygon
                cv2.fillPoly(image, [poly], fill_color)
                cv2.polylines(image, [poly], isClosed=True, color=outline_color, thickness=outline_thickness)


    def get_key_at_pos(self, x: int, y: int, z: float, wrist_z: float) -> Optional[int]:
        """
        Finds the MIDI note of the key at the given (x, y, z) position,
        using polygon hit testing and considering press depth.
        Checks black keys first due to drawing order.
        Returns MIDI note number or None.
        """
        potential_press = None
        # Iterate reverse (black keys first, as they are drawn last/on top)
        for key_info in reversed(self.keys):
            poly = key_info['poly_points']
            if poly is None: continue

            # 1. Basic Polygon Containment Check (XY plane)
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                midi_note = key_info['midi']
                is_white = key_info['is_white']
                # Use the *visual* top/bottom Y for calculations
                key_top_y = min(key_info['poly_top_y'], key_info['poly_bottom_y'])
                key_bottom_y = max(key_info['poly_top_y'], key_info['poly_bottom_y'])
                key_height = abs(key_bottom_y - key_top_y)

                if key_height <= 0: continue # Avoid division by zero

                # 2. Check Vertical Entry Threshold
                entry_factor = KEY_PRESS_Y_ENTRY_FACTOR_BLACK if not is_white else KEY_PRESS_Y_ENTRY_FACTOR_WHITE

                # Threshold is calculated from the visual top edge 'downwards' into the key
                press_y_threshold = key_top_y + key_height * entry_factor
                # Check if the finger Y is beyond this threshold (further into the key)
                valid_y_press = y >= press_y_threshold

                if valid_y_press:
                    # 3. Optional: Check Z-Depth Threshold
                    if PRESS_DEPTH_THRESHOLD is not None:
                        # Compare fingertip Z to its corresponding wrist Z
                        # MediaPipe Z: Smaller values are closer to the camera
                        if z < wrist_z - PRESS_DEPTH_THRESHOLD:
                            potential_press = midi_note
                            break # Found a potential pressed key (black keys have priority)
                    else:
                        # Z-depth check disabled, press if within X/Y/Entry bounds
                        potential_press = midi_note
                        break # Found a potential pressed key

        return potential_press


class MidiPlayer:
    """Handles MIDI output initialization and playback."""
    def __init__(self, instrument=0):
        self.midi_out = None
        self.output_id = -1
        self.instrument = instrument
        self.last_check_time = 0
        self.device_name = "Not Initialized"
        self._initialize_midi()

    def _initialize_midi(self):
        """Initializes pygame MIDI and finds a suitable output device."""
        try:
            pygame.midi.quit() # Ensure clean state
            pygame.midi.init()
            print("Initializing MIDI...")

            found_preferred = False
            preferred_device_name = b"Microsoft GS Wavetable Synth" # Common on Windows

            # 1. Try preferred device name explicitly
            for i in range(pygame.midi.get_count()):
                info = pygame.midi.get_device_info(i)
                # info structure: (interf, name, input, output, opened)
                if info and len(info) >= 4 and info[3] == 1: # Check if it's an output device
                    try: # Defensive decoding
                        device_name_bytes = info[1]
                        if preferred_device_name in device_name_bytes:
                            self.output_id = i
                            found_preferred = True
                            print(f"Found preferred MIDI device: {device_name_bytes.decode()}")
                            break
                    except Exception as decode_err:
                         print(f"Warning: Could not decode name for device {i}: {decode_err}")


            # 2. Fallback to default output device if preferred not found
            if not found_preferred:
                default_id = pygame.midi.get_default_output_id()
                if default_id != -1:
                    self.output_id = default_id
                    print(f"Using default MIDI output ID: {default_id}")
                else: # 3. Fallback to first available output device
                     print("Default MIDI device not found. Searching for first available...")
                     for i in range(pygame.midi.get_count()):
                        info = pygame.midi.get_device_info(i)
                        if info and len(info) >= 4 and info[3] == 1:
                            self.output_id = i
                            print(f"WARN: Using first available MIDI output: ID {i}")
                            break

            if self.output_id != -1:
                self.midi_out = pygame.midi.Output(self.output_id)
                self.midi_out.set_instrument(self.instrument)
                info = pygame.midi.get_device_info(self.output_id)
                self.device_name = info[1].decode('utf-8', 'ignore') if info and info[1] else f"Device ID {self.output_id}"
                print(f"MIDI Output '{self.device_name}' (ID: {self.output_id}) initialized successfully.")
                print(f"Instrument set to: {self.instrument}")

            else:
                self.device_name = "No Device Found"
                print("ERROR: No MIDI output device found. Sound will not play.")
                self.midi_out = None

        except pygame.midi.MidiException as midi_err:
            print(f"ERROR during MIDI initialization (Pygame MIDI): {midi_err}")
            print("Sound will likely not play. Ensure a MIDI synth is installed and configured.")
            self.midi_out = None
            self.device_name = f"MIDI Error: {midi_err}"
        except Exception as e:
            print(f"ERROR during MIDI initialization (General): {e}")
            print("Sound will likely not play.")
            self.midi_out = None
            self.device_name = f"Error: {e}"

    def set_instrument(self, instrument_id: int):
        """Sets the MIDI instrument."""
        if self.midi_out:
            try:
                self.midi_out.set_instrument(instrument_id)
                self.instrument = instrument_id
                print(f"MIDI Instrument changed to: {instrument_id}")
            except (pygame.midi.MidiException, AttributeError, TypeError) as e:
                print(f"ERROR setting MIDI instrument: {e}")
                self._handle_midi_error()

    def play_note(self, note: int, velocity: int):
        """Sends a MIDI note-on message."""
        if self.midi_out:
            try:
                self.midi_out.note_on(note, velocity)
            except (pygame.midi.MidiException, AttributeError, TypeError, OSError) as e: # Added OSError
                print(f"ERROR playing note {note}: {e}")
                self._handle_midi_error()

    def stop_note(self, note: int):
        """Sends a MIDI note-off message."""
        if self.midi_out:
            try:
                self.midi_out.note_off(note, 0) # Velocity 0 for note-off
            except (pygame.midi.MidiException, AttributeError, TypeError, OSError) as e: # Added OSError
                print(f"ERROR stopping note {note}: {e}")
                self._handle_midi_error()

    def stop_all_notes(self):
        """Stops all potentially sounding MIDI notes."""
        if self.midi_out:
            print("Stopping all MIDI notes...")
            for i in range(128):
                try:
                    self.midi_out.note_off(i, 0)
                except Exception:
                    pass # Ignore errors during mass stop
            # Optionally send All Notes Off MIDI message (CC 123) - can sometimes help
            try:
                 # Channel 1 (0) to Channel 16 (15): 0xB0 to 0xBF
                 for channel in range(16):
                    control_change = 0xB0 + channel
                    self.midi_out.write_short(control_change, 123, 0) # CC 123 = All Notes Off
            except Exception as e:
                print(f"Minor error sending All Notes Off: {e}")


    def play_test_sound(self):
        """Plays a scale and chord to verify MIDI output."""
        if not self.midi_out:
            print("Cannot play test sound: MIDI output not available.")
            return False

        print(f"Playing MIDI Test Sound on '{self.device_name}'...")
        try:
            # Ensure correct instrument is set
            self.set_instrument(self.instrument) # Use the method to handle potential errors
            if not self.midi_out: return False # set_instrument might have failed

            # Play a C major scale
            scale_notes = [60, 62, 64, 65, 67, 69, 71, 72] # C4 to C5
            for note in scale_notes:
                self.play_note(note, MIDI_NOTE_VELOCITY)
                pygame.time.wait(150)
                self.stop_note(note)
                pygame.time.wait(50)
                # Check if midi_out was closed due to an error during play/stop
                if not self.midi_out:
                    print("MIDI connection lost during test sound.")
                    return False

            # Play C Major Chord
            chord_notes = [60, 64, 67]
            for note in chord_notes:
                 self.play_note(note, MIDI_NOTE_VELOCITY)
            pygame.time.wait(500)
            for note in chord_notes:
                 self.stop_note(note)

            if not self.midi_out:
                print("MIDI connection lost during test sound.")
                return False

            print("Test sound finished.")
            return True # Indicate success

        except (pygame.midi.MidiException, AttributeError, TypeError, OSError) as e:
            print(f"ERROR playing test sound: {e}. Loading default settings.")
            self._handle_midi_error()
            return False # Indicate failure
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while playing test sound: {e}. Loading default settings.")
            self._handle_midi_error()
            return False # Indicate failure

    def check_health(self):
        """Periodically checks MIDI connection and attempts re-initialization if needed."""
        current_time = time.time()
        if current_time - self.last_check_time < MIDI_CHECK_INTERVAL:
            return # Not time yet

        self.last_check_time = current_time
        if not self.midi_out:
            # If it was previously closed due to error, try re-init periodically
            print("Attempting periodic MIDI re-initialization...")
            self._initialize_midi()
            return

        # Simple check: try setting the instrument again (low overhead)
        # If this fails, it will trigger _handle_midi_error
        self.set_instrument(self.instrument)


    def _handle_midi_error(self):
        """Closes the current MIDI output upon error to allow re-initialization."""
        print("MIDI Error detected. Closing current output.")
        if self.midi_out:
            try:
                # Try stopping notes before closing
                self.stop_all_notes()
                self.midi_out.close()
            except Exception as close_err:
                print(f"Error closing MIDI device: {close_err}")
            finally:
                 # Ensure midi_out is None even if close fails
                 self.midi_out = None
                 self.output_id = -1 # Reset ID too
                 self.device_name = "Error - Reinitializing..."
        # Next health check or operation will attempt _initialize_midi()

    def cleanup(self):
        """Cleans up MIDI resources."""
        if self.midi_out:
            try:
                self.stop_all_notes()
                self.midi_out.close()
                print("MIDI output closed.")
            except Exception as e:
                print(f"Minor error during MIDI cleanup: {e}")
            finally:
                 self.midi_out = None # Ensure it's None

        # Quit pygame.midi if initialized
        # Check explicitly because init might fail but leave pygame.midi imported
        try:
            if pygame.midi.get_init():
                pygame.midi.quit()
                print("Pygame MIDI quit.")
        except Exception as e:
            print(f"Error quitting pygame.midi: {e}")


class AirPianoGame:
    """Main class for the Air Piano game."""
    def __init__(self):
        print("Initializing Air Piano Game...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        # Verify frame dimensions after setting
        ret_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        ret_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if ret_w != FRAME_WIDTH or ret_h != FRAME_HEIGHT:
             print(f"Warning: Webcam does not support {FRAME_WIDTH}x{FRAME_HEIGHT}. Using {int(ret_w)}x{int(ret_h)}.")
             self.frame_width = int(ret_w)
             self.frame_height = int(ret_h)
        else:
             self.frame_width = FRAME_WIDTH
             self.frame_height = FRAME_HEIGHT
        print(f"Webcam opened ({self.frame_width}x{self.frame_height}).")

        # Camera flip settings
        self.flip_horizontal = True  # Mirror image horizontally by default for intuitive control
        self.flip_vertical = False   # Don't flip vertically by default
        print(f"Camera orientation: horizontal flip={self.flip_horizontal}, vertical flip={self.flip_vertical}")

        # Initialize Pygame core first (needed for time, events if used later)
        try:
            pygame.init()
        except Exception as e:
            print(f"Warning: Pygame core init failed: {e}")
            # MIDI relies on pygame, so we probably can't continue easily if this fails
            raise RuntimeError("Failed to initialize Pygame core.") from e


        self.hand_tracker = HandTracker()
        self.piano = Piano(PIANO_X, PIANO_Y, PIANO_WIDTH,
                           WHITE_KEY_HEIGHT, BLACK_KEY_HEIGHT_FACTOR, BLACK_KEY_WIDTH_FACTOR,
                           PERSPECTIVE_FACTOR, KEY_FRONT_HEIGHT, KEY_TOP_Y_OFFSET,
                           START_MIDI_NOTE, NUM_OCTAVES)
        self.midi_player = MidiPlayer(instrument=MIDI_INSTRUMENT)

        # Start in positioning mode
        self.game_state = STATE_POSITIONING
        self.pressed_keys: Set[int] = set() # MIDI notes currently down
        
        # Debounce dictionaries for key press/release
        self.key_press_potential = defaultdict(int)  # Counter for potential key presses
        self.key_release_potential = defaultdict(int)  # Counter for potential key releases

        # Position and size adjustment parameters
        self.position_step = 10  # Pixels to move per key press
        self.width_step = 20     # Pixels to adjust width per key press
        self.height_step = 10    # Pixels to adjust height per key press
        self.octave_step = 1     # Number of octaves to add/remove per key press
        
        # Store locations of fingertips that pressed a key this frame
        self.active_press_locations: List[Tuple[int, int]] = []

        # Calibration state variables
        self.hands_detected_time: Optional[float] = None
        self.calibration_countdown: float = 3.0

        # Preset filename
        self.preset_filename = "piano_preset.json"

        # Attempt to load preset on startup
        self._load_preset() # Load settings before first draw/log

        # Log initial/loaded piano position
        print(f"Initial piano position: x={self.piano.x_base}, y={self.piano.y_bottom}, rotation={self.piano.rotation}°")
        print(f"Initial piano size: width={self.piano.width_base}, height={self.piano.white_key_h}, octaves={self.piano.num_octaves}")

        self._initial_audio_check()

    def _initial_audio_check(self):
        """Performs and logs the initial audio test."""
        print("\n" + "="*30)
        print("   INITIAL AUDIO DEVICE CHECK")
        print("="*30)
        if self.midi_player.output_id != -1: # Check if an ID was assigned, even if midi_out is None temporarily
            print(f"Attempting test sound on: '{self.midi_player.device_name}' (ID: {self.midi_player.output_id})")
            # Ensure midi_out is potentially available before testing
            if not self.midi_player.midi_out:
                 print("MIDI output object not currently active, attempting re-init before test...")
                 self.midi_player._initialize_midi() # Try to get it back

            if self.midi_player.midi_out:
                success = self.midi_player.play_test_sound()
                if success:
                    print("-> If you heard the test sound, setup is likely correct.")
                else:
                     print("-> Test sound failed or MIDI connection lost during test.")
                     print("   MIDI output may not be working reliably.")
            else:
                 print("-> Failed to initialize MIDI output for testing.")

            print("\nIf sound issues persist:")
            print("  1. Check system volume & speaker/headphone connection.")
            print("  2. Ensure the selected MIDI device routes to an audible output.")
            print("     (e.g., 'Microsoft GS Wavetable Synth' requires Windows audio).")
            print("  3. Install/configure a software MIDI synthesizer:")
            print("     - Windows: VirtualMIDISynth (recommended), loopMIDI + VST synth")
            print("     - macOS/Linux: FluidSynth + SoundFont, SimpleSynth")
            print("  4. Press 't' in the game window to test sound again.")
        else:
            print("   NO AUDIO OUTPUT DEVICE FOUND")
            print("No working MIDI output device was initialized.")
            print("The game will run visually, but without sound.")
            print("See steps above for enabling sound via MIDI synthesizer.")
        print("="*30 + "\n")

    def run(self):
        """Main game loop."""
        print("Starting game loop... Press 'ESC' in the window to quit.")
        frame_count = 0
        start_time = time.time()

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Warning: Ignoring empty camera frame.")
                pygame.time.wait(10) # Avoid busy-waiting if frames consistently fail
                continue

            # Apply flips based on settings
            image_bgr = frame
            if self.flip_horizontal:
                image_bgr = cv2.flip(image_bgr, 1)  # 1 = horizontal flip
            if self.flip_vertical:
                image_bgr = cv2.flip(image_bgr, 0)  # 0 = vertical flip
                
            # Convert BGR->RGB for MediaPipe
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Process hands
            self.hand_tracker.process_frame(image_rgb)

            # Get landmark data
            landmarks_list = self.hand_tracker.get_landmarks(self.frame_width, self.frame_height)
            fingertips, wrists = self.hand_tracker.get_fingertips_and_wrists(landmarks_list)

            # --- State Machine ---
            if self.game_state == STATE_POSITIONING:
                self._handle_positioning(image_bgr)
            elif self.game_state == STATE_CALIBRATING:
                self._handle_calibrating(image_bgr, landmarks_list)
            elif self.game_state == STATE_RUNNING:
                self._handle_running(image_bgr, landmarks_list, fingertips, wrists)

            # --- Draw Shared Elements (FPS, etc.) ---
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                cv2.putText(image_bgr, f"FPS: {fps:.1f}", (self.frame_width - 100, self.frame_height - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
                frame_count = 0
                start_time = time.time()

            # --- Handle Keyboard Input ---
            key = cv2.waitKey(5) & 0xFF # Use waitKey(5) for smoother feel
            
            # Global keys (always active)
            if key == 27:  # ESC key to quit
                print("ESC key pressed. Exiting...")
                break
            elif key == ord('t'):
                print("Testing sound...")
                self.midi_player.play_test_sound()
            elif key == ord('r'):
                # 'r' to recalibrate from any state
                self._enter_calibration_state()
            elif key == ord('p'):
                # 'p' to go back to positioning from any state
                self._enter_positioning_state()
                
            # State-specific keys
            elif self.game_state == STATE_POSITIONING:
                # Arrow keys aliases (optional, some systems send different codes)
                # 81: Left, 82: Up, 83: Right, 84: Down (common terminal codes)
                key_map = {81: 'a', 82: 'w', 83: 'd', 84: 's'}
                # Map if possible, else keep original
                mapped_key = key_map.get(key, key) 
                # If key is an integer (ASCII code), convert to character
                if isinstance(mapped_key, int):
                    mapped_key = chr(mapped_key)

                # Piano movement
                if mapped_key == 'w':
                    self.piano.update_position(y=self.piano.y_bottom - self.position_step)
                elif mapped_key == 's':
                    self.piano.update_position(y=self.piano.y_bottom + self.position_step)
                elif mapped_key == 'a':
                    self.piano.update_position(x=self.piano.x_base - self.position_step)
                elif mapped_key == 'd':
                    self.piano.update_position(x=self.piano.x_base + self.position_step)
                # Toggle rotation between 0 and 180 degrees
                elif mapped_key == 'z' or mapped_key == 'x':
                    new_rotation = 180 if self.piano.rotation == 0 else 0
                    self.piano.update_position(rotation=new_rotation)
                # Size adjustments
                elif mapped_key == '1':  # Decrease width
                    new_width = max(200, self.piano.width_base - self.width_step)
                    self.piano.update_size(width=new_width)
                elif mapped_key == '2':  # Increase width
                    new_width = min(self.frame_width - 2*PIANO_X, self.piano.width_base + self.width_step) # Ensure width fits
                    self.piano.update_size(width=new_width)
                elif mapped_key == '3':  # Decrease height
                    new_height = max(100, self.piano.white_key_h - self.height_step)
                    self.piano.update_size(height=new_height)
                elif mapped_key == '4':  # Increase height
                    new_height = min(300, self.piano.white_key_h + self.height_step)
                    self.piano.update_size(height=new_height)
                elif mapped_key == '5':  # Fewer octaves
                    new_octaves = max(1, self.piano.num_octaves - self.octave_step)
                    self.piano.update_size(num_octaves=new_octaves)
                elif mapped_key == '6':  # More octaves
                    new_octaves = min(5, self.piano.num_octaves + self.octave_step)
                    self.piano.update_size(num_octaves=new_octaves)
                # Save and Load Preset
                elif mapped_key == 'f':
                    self._save_preset()
                elif mapped_key == 'l':
                    self._load_preset()
                # Camera orientation adjustments
                elif mapped_key == 'h':
                    self.flip_horizontal = not self.flip_horizontal
                    print(f"Horizontal camera flip: {self.flip_horizontal}")
                elif mapped_key == 'v':
                    self.flip_vertical = not self.flip_vertical
                    print(f"Vertical camera flip: {self.flip_vertical}")
                # Continue to calibration
                elif mapped_key == 'c' or key == 13:  # 13 is Enter key
                    self._enter_calibration_state() # Transition to calibration

            # --- Display the final image ---
            cv2.imshow('Virtual Piano Keyboard', image_bgr)

        self.cleanup()

    def _handle_positioning(self, image: np.ndarray):
        """Handles logic and drawing for the piano positioning state."""
        # Dim the background for better visibility
        overlay = image.copy()
        cv2.rectangle(overlay, (0,0), (self.frame_width, self.frame_height), (10, 10, 30), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Draw guide piano
        self.piano.draw(image, set())
        
        # Draw main title with shadow
        title = "PIANO POSITIONING"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)[0]
        title_x = (self.frame_width - title_size[0]) // 2
        # Shadow
        cv2.putText(image, title, (title_x + 3, 63),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 60, 0), 2, cv2.LINE_AA)
        # Main text
        cv2.putText(image, title, (title_x, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, COLOR_CALIBRATE_TEXT, 2, cv2.LINE_AA)
        
        # Create instructions panel
        panel_width = 400
        panel_height = 430
        panel_x = self.frame_width // 2 - panel_width // 2
        panel_y = 90
        draw_panel(image, panel_x, panel_y, panel_width, panel_height, "Controls")
        
        # Organized instructions with sections
        instructions = [
            ("Position", [
                "W/S or ↑/↓: Move piano up/down",
                "A/D or ←/→: Move piano left/right",
                "Z/X: Flip piano orientation"
            ]),
            ("Size", [
                "1/2: Decrease/Increase width",
                "3/4: Decrease/Increase height",
                "5/6: Decrease/Increase octaves"
            ]),
            ("Camera", [
                "H/V: Toggle Horizontal/Vertical flip"
            ]),
            ("Presets", [
                f"F: Save layout to {self.preset_filename}",
                f"L: Load layout from file"
            ]),
            ("Navigation", [
                "C or Enter: Continue to calibration"
            ])
        ]
        
        # Draw instructions with sections
        y_pos = panel_y + 45
        for section_title, section_items in instructions:
            # Draw section title
            draw_enhanced_text(image, section_title + ":", 
                             (panel_x + 20, y_pos), 
                             0.7, COLOR_UI_SECTION_TITLE, 2)
            y_pos += 25
            
            # Draw section items
            for item in section_items:
                draw_enhanced_text(image, "• " + item, 
                                 (panel_x + 35, y_pos), 
                                 0.6, COLOR_UI_KEY_LABEL)
                y_pos += 22
            
            y_pos += 10  # Gap between sections
        
        # Create info panel for current settings
        info_panel_width = 350
        info_panel_height = 100
        info_panel_x = 20
        info_panel_y = 20
        draw_panel(image, info_panel_x, info_panel_y, info_panel_width, info_panel_height, "Current Settings")
        
        # Display position and size details
        info_y = info_panel_y + 45
        position_text = f"• Position: X={self.piano.x_base}, Y={self.piano.y_bottom}"
        rotation_text = f"• Rotation: {self.piano.rotation}°"
        size_text = f"• Size: {self.piano.width_base}×{self.piano.white_key_h} ({self.piano.num_octaves} octaves)"
        camera_text = f"• Camera: {'Mirrored' if self.flip_horizontal else 'Normal'}"
        
        draw_enhanced_text(image, position_text, (info_panel_x + 15, info_y), 0.6, COLOR_UI_KEY_LABEL)
        draw_enhanced_text(image, rotation_text, (info_panel_x + 15, info_y + 20), 0.6, COLOR_UI_KEY_LABEL)
        draw_enhanced_text(image, size_text, (info_panel_x + 15, info_y + 40), 0.6, COLOR_UI_KEY_LABEL)
        draw_enhanced_text(image, camera_text, (info_panel_x + 15, info_y + 60), 0.6, COLOR_UI_KEY_LABEL)

    def _handle_calibrating(self, image: np.ndarray, landmarks_list: List[List[Tuple[int, int, float]]]):
        """Handles logic and drawing for the calibration state."""
        # Dim the background slightly
        overlay = image.copy()
        cv2.rectangle(overlay, (0,0), (self.frame_width, self.frame_height), (10, 10, 40), -1)
        cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

        # Draw guide piano (no highlights)
        self.piano.draw(image, set())

        # Draw main title with shadow
        title = "CALIBRATION MODE"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)[0]
        title_x = (self.frame_width - title_size[0]) // 2
        # Shadow
        cv2.putText(image, title, (title_x + 3, 63),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 60, 0), 2, cv2.LINE_AA)
        # Main text
        cv2.putText(image, title, (title_x, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, COLOR_CALIBRATE_TEXT, 2, cv2.LINE_AA)
        
        # Create instructions panel
        panel_width = 500
        panel_height = 150
        panel_x = self.frame_width // 2 - panel_width // 2
        panel_y = 90
        draw_panel(image, panel_x, panel_y, panel_width, panel_height, "Instructions")

        # Add instructions text
        instructions_text = "Hold your hand(s) steady over the piano area"
        instruction_size = cv2.getTextSize(instructions_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
        draw_enhanced_text(image, instructions_text, 
                          (panel_x + (panel_width - instruction_size[0]) // 2, panel_y + 50), 
                          0.8, COLOR_UI_HIGHLIGHT, 1, True)

        current_time = time.time()
        if len(landmarks_list) >= 1: # Need at least one hand
            if self.hands_detected_time is None:
                print("Hand(s) detected. Starting calibration countdown...")
                self.hands_detected_time = current_time

            elapsed = current_time - self.hands_detected_time
            remaining = max(0, self.calibration_countdown - elapsed)

            # Draw countdown progress bar
            bar_x = panel_x + 50
            bar_y = panel_y + 80
            bar_w = panel_width - 100
            bar_h = 30
            progress = (self.calibration_countdown - remaining) / self.calibration_countdown
            
            # Draw bar background with rounded corners if possible
            try:
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                             (60, 60, 80), -1, cv2.LINE_AA)
            except:
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                             (60, 60, 80), -1)
            
            # Draw progress
            progress_width = int(bar_w * progress)
            if progress_width > 0:
                try:
                    cv2.rectangle(image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_h), 
                                 (0, 200, 100), -1, cv2.LINE_AA)
                except:
                    cv2.rectangle(image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_h), 
                                 (0, 200, 100), -1)
            
            # Draw border
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), 
                         COLOR_UI_PANEL_BORDER, 2, cv2.LINE_AA)
            
            # Draw countdown text
            countdown_txt = f"{remaining:.1f}s remaining"
            txt_size = cv2.getTextSize(countdown_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            draw_enhanced_text(image, countdown_txt, 
                           (bar_x + (bar_w - txt_size[0]) // 2, bar_y + bar_h - 8),
                           0.7, (255, 255, 255) if remaining > 0.1 else (0, 0, 0), 1, False)


            if remaining <= 0:
                print("Calibration Complete. Starting game.")
                self._enter_running_state() # Use method to ensure clean transition
        else:
            if self.hands_detected_time is not None:
                print("Hand(s) lost. Resetting calibration countdown.")
                self.hands_detected_time = None # Reset timer

            # Draw waiting message
            waiting_text = "POSITION HAND(S) TO START"
            txt_size = cv2.getTextSize(waiting_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            waiting_x = panel_x + (panel_width - txt_size[0]) // 2
            waiting_y = panel_y + 95
            
            # Draw with animated pulsing effect using time
            pulse = abs(math.sin(time.time() * 3)) * 50  # Pulsing factor
            pulse_color = (200 + int(pulse), 220, int(pulse))
            
            # Shadow for better visibility
            cv2.putText(image, waiting_text, (waiting_x + 2, waiting_y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, waiting_text, (waiting_x, waiting_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, pulse_color, 2, cv2.LINE_AA)

        # --- Draw Control Panel ---
        panel_width = 350
        panel_height = 60
        panel_x = (self.frame_width - panel_width) // 2
        panel_y = self.frame_height - panel_height - 20
        draw_panel(image, panel_x, panel_y, panel_width, panel_height)
        
        # Key control hints
        controls_text = "P: Back to Positioning  |  ESC: Quit"
        (text_w, _), _ = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        draw_enhanced_text(image, controls_text, 
                         (panel_x + (panel_width - text_w) // 2, panel_y + 35), 
                         0.6, COLOR_UI_HIGHLIGHT)

    def _enter_positioning_state(self):
        """Enters piano positioning mode."""
        print("Entering piano positioning mode...")
        self.game_state = STATE_POSITIONING
        self.midi_player.stop_all_notes()  # Stop any playing notes
        self.pressed_keys.clear()
        self._reset_debounce_state() # Clear debounce counters
        # Log current position
        print(f"Current piano position: x={self.piano.x_base}, y={self.piano.y_bottom}, rotation={self.piano.rotation}°")
        print(f"Current camera orientation: horizontal flip={self.flip_horizontal}, vertical flip={self.flip_vertical}")

    def _enter_calibration_state(self):
        """Resets state variables and enters calibration mode."""
        print("Entering calibration mode...")
        self.game_state = STATE_CALIBRATING
        self.midi_player.stop_all_notes() # Stop any playing notes
        self.pressed_keys.clear()
        self.hands_detected_time = None # Reset calibration timer
        # Log current position
        print(f"Current piano position: x={self.piano.x_base}, y={self.piano.y_bottom}, rotation={self.piano.rotation}°")

    def _enter_running_state(self):
        """Enters the main playing state."""
        print("Entering running state...")
        self.game_state = STATE_RUNNING
        self.pressed_keys.clear() # Ensure no stuck notes
        self._reset_debounce_state() # Clear debounce counters

    def _reset_debounce_state(self):
        """Resets the debounce counters."""
        self.key_press_potential.clear()
        self.key_release_potential.clear()
        print("Debounce state reset.")

    def cleanup(self):
        """Releases resources."""
        print("Cleaning up resources...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Webcam released.")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")
        self.midi_player.cleanup() # Handles MIDI closing and pygame.midi.quit
        # Quit pygame core if it was initialized
        if pygame.get_init():
             pygame.quit()
             print("Pygame quit.")
        print("Exited cleanly.")

    # --- Preset Save/Load ---

    def _save_preset(self):
        """Saves the current piano layout and camera settings to a JSON file."""
        preset_data = {
            'piano_x': self.piano.x_base,
            'piano_y': self.piano.y_bottom,
            'piano_rotation': self.piano.rotation,
            'piano_width': self.piano.width_base,
            'piano_height': self.piano.white_key_h,
            'piano_octaves': self.piano.num_octaves,
            'flip_horizontal': self.flip_horizontal,
            'flip_vertical': self.flip_vertical
        }
        try:
            with open(self.preset_filename, 'w') as f:
                json.dump(preset_data, f, indent=4)
            print(f"Preset saved successfully to {self.preset_filename}")
        except IOError as e:
            print(f"ERROR: Could not save preset to {self.preset_filename}: {e}")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while saving preset: {e}")

    def _load_preset(self):
        """Loads piano layout and camera settings from the JSON preset file."""
        try:
            with open(self.preset_filename, 'r') as f:
                preset_data = json.load(f)

            # Apply loaded settings
            self.flip_horizontal = preset_data.get('flip_horizontal', self.flip_horizontal)
            self.flip_vertical = preset_data.get('flip_vertical', self.flip_vertical)

            # Update piano size first (height, width, octaves)
            self.piano.update_size(
                width=preset_data.get('piano_width', self.piano.width_base),
                height=preset_data.get('piano_height', self.piano.white_key_h),
                num_octaves=preset_data.get('piano_octaves', self.piano.num_octaves)
            )
            # Then update position (x, y, rotation)
            # This automatically calls _recalculate_layout inside update_position
            self.piano.update_position(
                x=preset_data.get('piano_x', self.piano.x_base),
                y=preset_data.get('piano_y', self.piano.y_bottom),
                rotation=preset_data.get('piano_rotation', self.piano.rotation)
            )

            print(f"Preset loaded successfully from {self.preset_filename}")

        except FileNotFoundError:
            print(f"INFO: Preset file {self.preset_filename} not found. No preset loaded.")
        except (IOError, json.JSONDecodeError) as e:
            print(f"ERROR: Could not load or parse preset from {self.preset_filename}: {e}")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while loading preset: {e}")

    def _handle_running(self, image: np.ndarray, landmarks_list: List[List[Tuple[int, int, float]]], fingertips: List[Tuple[int, int, float]], wrists: List[Tuple[int, int, float]]):
        """Handles logic and drawing for the main running state."""
        # Reset active press locations for this frame
        self.active_press_locations = []
        
        # Create a slight darkened background for better contrast
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.frame_height), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
        
        # Draw the piano with current pressed keys
        self.piano.draw(image, self.pressed_keys)
        
        # Process hand landmarks if hands are detected
        if landmarks_list:
            # Draw hand landmarks
            self.hand_tracker.draw_landmarks(image)
            
            # Reset for collecting this frame's pressed notes
            current_frame_pressed_notes = {}  # Maps MIDI note to finger position
            
            # Process each fingertip
            for i, fingertip in enumerate(fingertips):
                if not wrists or i // 5 >= len(wrists):  # Make sure we have a wrist for this hand
                    continue
                    
                wrist = wrists[i // 5]  # Get the wrist for the current hand
                x, y, z = fingertip
                _, _, wrist_z = wrist
                
                # Check if fingertip is pressing a key
                pressed_midi_note = self.piano.get_key_at_pos(x, y, z, wrist_z)
                if pressed_midi_note is not None:
                    current_frame_pressed_notes[pressed_midi_note] = (x, y)
                    
                    # Save fingertip location for visual feedback
                    self.active_press_locations.append((x, y))
            
            # Convert dictionary keys to a set for set operations
            current_frame_keys = set(current_frame_pressed_notes.keys())
            
            # Implement debouncing logic
            for midi_note in current_frame_keys:
                if midi_note not in self.pressed_keys:
                    # Potential new press - increment counter
                    self.key_press_potential[midi_note] += 1
                    if self.key_press_potential[midi_note] >= DEBOUNCE_FRAMES_ON:
                        # Counter reached threshold - register key press
                        self.pressed_keys.add(midi_note)
                        self.midi_player.play_note(midi_note, MIDI_NOTE_VELOCITY)
                        self.key_press_potential[midi_note] = 0  # Reset counter
                else:
                    # Key already pressed - reset release counter
                    self.key_release_potential[midi_note] = 0
            
            # Check for key releases with debouncing
            for midi_note in list(self.pressed_keys):
                if midi_note not in current_frame_keys:
                    # Potential release - increment counter
                    self.key_release_potential[midi_note] += 1
                    if self.key_release_potential[midi_note] >= DEBOUNCE_FRAMES_OFF:
                        # Counter reached threshold - register key release
                        self.pressed_keys.remove(midi_note)
                        self.midi_player.stop_note(midi_note)
                        self.key_release_potential[midi_note] = 0  # Reset counter
                else:
                    # Key still pressed - reset press counter
                    self.key_press_potential[midi_note] = 0
            
            # Draw press indicators for fingertips that activated keys
            if self.active_press_locations:
                for x, y in self.active_press_locations:
                    # Enhanced visual feedback with concentric circles
                    cv2.circle(image, (x, y), 15, COLOR_FINGERTIP_PRESS_EFFECT, 2)  # Outer circle
                    cv2.circle(image, (x, y), 8, COLOR_FINGERTIP_PRESS_EFFECT, -1)  # Inner filled circle
                    cv2.circle(image, (x, y), 3, (255, 255, 255), -1)  # White center dot
        
        # Periodically check MIDI health
        self.midi_player.check_health()
        
        # --- Draw Status Panel ---
        panel_width = 250
        panel_height = 140
        panel_x = 20
        panel_y = 20
        draw_panel(image, panel_x, panel_y, panel_width, panel_height, "Status")
        
        # Display status information in the panel
        text_lines = [
            f"• Notes: {len(self.pressed_keys)} active",
            f"• MIDI: {self.midi_player.device_name}",
            f"• Piano: {self.piano.num_octaves} octaves",
            f"• Camera: {'Mirrored' if self.flip_horizontal else 'Normal'}"
        ]
        
        y_pos = panel_y + 45
        for line in text_lines:
            draw_enhanced_text(image, line, (panel_x + 15, y_pos), 0.6, COLOR_UI_KEY_LABEL)
            y_pos += 25
        
        # --- Draw Control Panel ---
        panel_width = 350
        panel_height = 60
        panel_x = (self.frame_width - panel_width) // 2
        panel_y = self.frame_height - panel_height - 20
        draw_panel(image, panel_x, panel_y, panel_width, panel_height)
        
        # Key control hints
        controls_text = "P: Positioning  |  R: Recalibrate  |  ESC: Quit"
        (text_w, _), _ = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        draw_enhanced_text(image, controls_text, 
                         (panel_x + (panel_width - text_w) // 2, panel_y + 35), 
                         0.6, COLOR_UI_HIGHLIGHT)

# --- Main Execution ---
if __name__ == "__main__":
    game_instance = None
    try:
        game_instance = AirPianoGame()
        game_instance.run()
    except IOError as e:
         print(f"\n--- FATAL I/O ERROR ---")
         print(f"Error: {e}")
         print("Could not access the webcam. Please check:")
         print("- If the webcam is connected and powered on.")
         print("- If another application is using the webcam.")
         print("- If the correct camera index (usually 0) is used.")
         print("- Webcam permissions for the application.")
    except RuntimeError as e:
         print(f"\n--- FATAL RUNTIME ERROR ---")
         print(f"Error: {e}")
         print("This often relates to core library initialization (like Pygame).")
    except Exception as e:
        print(f"\n--- UNEXPECTED FATAL ERROR ---")
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("--------------------\n")
        print("Common issues:")
        print("- Webcam access denied or device unavailable.")
        print("- Missing dependencies (opencv-python, mediapipe, pygame). Install with pip.")
        print("- Graphics driver issues affecting OpenCV display.")
        print("- MIDI device conflicts or errors.")
    finally:
        # Ensure cleanup runs even if game initialization failed partially
        if game_instance:
            # If run() wasn't reached or exited prematurely, cleanup might not have run
             print("Ensuring cleanup is called...")
             game_instance.cleanup()
        else:
             # If game object creation failed, try basic cleanup
             print("Attempting basic cleanup...")
             cv2.destroyAllWindows()
             # Attempt to quit pygame modules if they were ever initialized
             try:
                 if pygame.midi.get_init(): pygame.midi.quit()
             except Exception: pass
             try:
                 if pygame.get_init(): pygame.quit()
             except Exception: pass
             print("Basic cleanup attempted.")
        print("\nApplication finished.")