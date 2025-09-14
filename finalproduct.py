import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pygame
import wave
import os
from datetime import datetime
import tkinter as tk

# -------------- Responsive Window and Start Page Setup ---------------

def get_window_size():
    root = tk.Tk()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.destroy()
    return int(sw * 0.8), int(sh * 0.8)

w, h = get_window_size()
window_name = "Theremin Gesture Player"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, w, h)

# Load logo image for menu page
logo_img = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)  # Supports transparency

def draw_maestro_silhouette(img, center, size):
    cx, cy = center
    cv2.ellipse(img, (cx, cy-size//3), (size//7, size//5), 0, 0, 360, (255,255,255), -1)
    cv2.rectangle(img, (cx-size//18, cy-size//3), (cx+size//18, cy+size//2), (255,255,255), -1)
    cv2.line(img, (cx, cy-size//10), (cx+size//2, cy-size//2), (255,255,255), size//19)
    cv2.line(img, (cx, cy-size//10), (cx-size//2, cy-size//2), (255,255,255), size//19)
    cv2.line(img, (cx+size//2, cy-size//2), (cx+size//2+size//9, cy-size//2-size//9), (255,255,255), size//29)

def show_start_page(window_name, w, h):
    btn_radius = min(w, h) // 8
    center = (w // 2, h // 2)
    font_scale_btn = max(0.7, h / 900 * 0.9)
    font_scale_sub = max(0.40, h / 1200 * 0.53)
    clicked = False
    def mouse_cb(event, x, y, flags, param):
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            if (x-center[0])**2 + (y-center[1])**2 <= btn_radius**2:
                clicked = True
    cv2.setMouseCallback(window_name, mouse_cb)
    while True:
        display = np.zeros((h, w, 3), dtype=np.uint8)
        draw_maestro_silhouette(display, (w//2, h//2-btn_radius//2), btn_radius)
        cv2.circle(display, center, btn_radius, (255,255,255), -1)
        btnsize = cv2.getTextSize("MAESTRO", cv2.FONT_HERSHEY_SIMPLEX, font_scale_btn, 3)[0]
        cv2.putText(display, "MAESTRO", (center[0]-btnsize[0]//2, center[1]+btnsize[1]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_btn, (0,0,0), 3)
        subsize = cv2.getTextSize("hands on music", cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, 2)[0]
        cv2.putText(display, "hands on music", (center[0]-subsize[0]//2, center[1]+btn_radius+subsize[1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, (255,255,255), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(15)
        if key == 27: exit(0)  # Escape to quit
        if key == 32 or clicked: break  # Space or click to proceed
    cv2.setMouseCallback(window_name, lambda *args: None)
    galaxy_transition(window_name, display, center, btn_radius, w, h)
    show_instruction_page(window_name, w, h)

def galaxy_transition(window_name, display, center, radius, w, h):
    maxr = int(np.sqrt(w**2+h**2)*1.01)
    for r in range(radius, maxr, 33):
        frame = display.copy()
        cv2.circle(frame, center, r, (255,255,255), -1)
        alpha = min(1, (r-radius)/(maxr-radius))
        blended = cv2.addWeighted(display, 1-alpha, frame, alpha, 0)
        cv2.imshow(window_name, blended)
        cv2.waitKey(1)

def show_instruction_page(window_name, w, h):
    lines = [
        "Welcome to Maestro, where you are the master of your music.",
        "Place both hands in front of the camera to begin recording.",
        "Use right hand index finger to hover and change instruments.",
        "Use left hand, moving it up and down to control the volume/intensity of your sounds."
    ]
    font_scale = max(0.55, min(0.83, h / 980 * 0.73))
    line_spacing = int(font_scale * 45)
    run = True
    while run:
        page = np.zeros((h, w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        total_height = len(lines)*line_spacing
        for i, line in enumerate(lines):
            tsize = cv2.getTextSize(line, font, font_scale, 2)[0]
            tx = (w - tsize[0]) // 2
            ty = (h - total_height) // 2 + i*line_spacing + tsize[1]
            cv2.putText(page, line, (tx, ty), font, font_scale, (255,255,255), 2)
        cv2.imshow(window_name, page)
        if cv2.waitKey(21) != -1:
            run = False

# Show start and instruction at launch
show_start_page(window_name, w, h)

# -------------- Begin Your Existing Code ---------------

# Setup - Enable detection of both hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()

# Accurate pitch frequencies based on A4=440Hz standard
def note_to_freq(note, octave):
    notes = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
             'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
    a4 = 440.0
    semitones_from_a4 = notes[note] + (octave - 4) * 12
    return a4 * (2 ** (semitones_from_a4 / 12))

INSTRUMENTS = {
    0: {"name": "Piano", "notes": [("C", 4), ("C#", 4), ("D", 4), ("D#", 4),
                                   ("E", 4), ("F", 4), ("F#", 4), ("G", 4),
                                   ("G#", 4), ("A", 4), ("A#", 4), ("B", 4)],
        "color": (255, 255, 255)},
    1: {"name": "Guitar", "notes": [("E", 2), ("F", 2), ("F#", 2), ("G", 2),
                                    ("G#", 2), ("A", 2), ("A#", 2), ("B", 2),
                                    ("C", 3), ("C#", 3), ("D", 3), ("D#", 3)],
        "color": (150, 75, 0)},
    2: {"name": "Violin", "notes": [("G", 2), ("G#", 2), ("A", 2), ("A#", 2),
                                    ("B", 2), ("C", 3), ("C#", 3), ("D", 3),
                                    ("D#", 3), ("E", 3), ("F", 3), ("F#", 3)],
        "color": (139, 69, 19)},
    3: {"name": "Cello", "notes": [("C", 2), ("C#", 2), ("D", 2), ("D#", 2),
                                   ("E", 2), ("F", 2), ("F#", 2), ("G", 2),
                                   ("G#", 2), ("A", 2), ("A#", 2), ("B", 2)],
        "color": (101, 67, 33)},
    4: {"name": "Flute", "notes": [("C", 5), ("C#", 5), ("D", 5), ("D#", 5),
                                   ("E", 5), ("F", 5), ("F#", 5), ("G", 5),
                                   ("G#", 5), ("A", 5), ("A#", 5), ("B", 5)],
        "color": (192, 192, 192)},
}

current_instrument = 0
last_switch_time = 0
sounds = {}
tempo = 80  # Default tempo

metronome_sound = None
last_metronome_time = 0
metronome_enabled = True
loop_enabled = False  # New flag for looping on/off

last_stable_note = -1
note_stability_time = 0
stability_threshold = 0.15

APP_STATE = "MENU"
menu_buttons = []
settings_buttons = []

BACKGROUND_COLORS = {
    "Black": (0, 0, 0),
    "Dark Blue": (20, 20, 60),
    "Dark Green": (0, 40, 0),
    "Dark Purple": (40, 0, 40),
    "Dark Red": (40, 0, 0)
}
current_background = "Black"
mouse_x, mouse_y = 0, 0

# Add LoopState class, metronome, instrument sounds, detection functions, etc.
# (Keep all your detailed functions and loop code unchanged here)

# Updated draw_menu() with pink buttons and logo

def draw_menu(display, w, h):
    global menu_buttons
    menu_buttons = []

    display[:] = BACKGROUND_COLORS[current_background]

    # Draw logo image at top center
    if logo_img is not None:
        scale = (w * 0.5) / logo_img.shape[1]
        logo_resized = cv2.resize(logo_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        y_offset = 30
        x_offset = (w - logo_resized.shape[1]) // 2
        if logo_resized.shape[2] == 4:
            alpha_s = logo_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                display[y_offset:y_offset+logo_resized.shape[0], x_offset:x_offset+logo_resized.shape[1], c] = (
                  alpha_s * logo_resized[:, :, c] +
                  alpha_l * display[y_offset:y_offset+logo_resized.shape[0], x_offset:x_offset+logo_resized.shape[1], c])
        else:
            display[y_offset:y_offset+logo_resized.shape[0], x_offset:x_offset+logo_resized.shape[1]] = logo_resized
    else:
        # Fallback title text if no logo
        title = "Theremin Gesture Player"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(display, title, (title_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Buttons below the logo or title
    button_width = 200
    button_height = 60
    button_x = (w - button_width) // 2
    button_spacing = 80
    base_y = 100 + (logo_resized.shape[0] if logo_img is not None else 0) + 40

    pink_base = (180, 50, 120)
    pink_hover = (220, 80, 170)

    buttons = [
        {"text": "START", "y": base_y, "color": pink_base, "hover_color": pink_hover, "action": "THEREMIN"},
        {"text": "TUTORIAL", "y": base_y + button_spacing, "color": pink_base, "hover_color": pink_hover, "action": "TUTORIAL"},
        {"text": "SETTINGS", "y": base_y + button_spacing * 2, "color": pink_base, "hover_color": pink_hover, "action": "SETTINGS"}
    ]

    for button in buttons:
        y = button["y"]
        is_hovering = (button_x <= mouse_x <= button_x + button_width and y <= mouse_y <= y + button_height)
        button_color = button["hover_color"] if is_hovering else button["color"]
        cv2.rectangle(display, (button_x, y), (button_x + button_width, y + button_height), button_color, -1)
        cv2.rectangle(display, (button_x, y), (button_x + button_width, y + button_height), (255, 255, 255), 2)

        text_size = cv2.getTextSize(button["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = button_x + (button_width - text_size[0]) // 2
        text_y = y + (button_height + text_size[1]) // 2
        cv2.putText(display, button["text"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        menu_buttons.append({"rect": (button_x, y, button_width, button_height), "action": button["action"]})

    instructions = [
        "Click on a button to continue",
        "Press ESC to quit"
    ]
    for i, instruction in enumerate(instructions):
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(display, instruction, (text_x, h - 80 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

# Update draw_settings with looping toggle

#loop_enabled = False  # Global flag to enable/disable looping

def draw_settings(display, w, h):
    global settings_buttons
    settings_buttons = []

    display[:] = BACKGROUND_COLORS[current_background]

    title = "Settings"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    title_x = (w - title_size[0]) // 2
    cv2.putText(display, title, (title_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

    # Background color buttons
    color_button_width = 100
    color_button_height = 30
    color_start_x = 50
    color_start_y = 140

    for i, (color_name, color_value) in enumerate(BACKGROUND_COLORS.items()):
        x = color_start_x + (i % 3) * (color_button_width + 10)
        y = color_start_y + (i // 3) * (color_button_height + 10)

        is_hovering = (x <= mouse_x <= x + color_button_width and y <= mouse_y <= y + color_button_height)
        is_selected = (color_name == current_background)

        if is_selected:
            button_color = (100, 100, 100)
            border_color = (255, 255, 0)
            border_thickness = 3
        elif is_hovering:
            button_color = tuple(min(255, c+50) for c in color_value)
            border_color = (255, 255, 255)
            border_thickness = 2
        else:
            button_color = color_value
            border_color = (128,128,128)
            border_thickness = 1

        cv2.rectangle(display, (x,y), (x+color_button_width, y+color_button_height), button_color, -1)
        cv2.rectangle(display, (x,y), (x+color_button_width, y+color_button_height), border_color, border_thickness)

        text_size = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = x + (color_button_width - text_size[0]) // 2
        text_y = y + (color_button_height + text_size[1]) // 2
        text_color = (255,255,255) if sum(color_value) < 200 else (0,0,0)
        cv2.putText(display, color_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        settings_buttons.append({"rect": (x,y,color_button_width,color_button_height), "action": f"COLOR_{color_name}"})

    # Looping toggle button
    toggle_width = color_button_width  # 100
    toggle_height = color_button_height  # 30

    num_color_buttons = len(BACKGROUND_COLORS)
    rows = (num_color_buttons + 2) // 3
    toggle_x = (w - toggle_width) // 2
    toggle_y = color_start_y + rows * (color_button_height + 10) + 10  # 10 pixel spacing below color buttons



    toggle_text = f"Looping: {'On' if loop_enabled else 'Off'}"
    toggle_color = (0,180,0) if loop_enabled else (180,0,0)
    cv2.rectangle(display, (toggle_x, toggle_y), (toggle_x+toggle_width, toggle_y+toggle_height), toggle_color, -1)
    cv2.rectangle(display, (toggle_x, toggle_y), (toggle_x+toggle_width, toggle_y+toggle_height), (255,255,255), 2)

    text_size = cv2.getTextSize(toggle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_x = toggle_x + (toggle_width - text_size) // 2
    text_y = toggle_y + (toggle_height + text_size[1]) // 2
    cv2.putText(display, toggle_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    settings_buttons.append({"rect": (toggle_x, toggle_y, toggle_width, toggle_height), "action": "TOGGLE_LOOPING"})

    # Other settings info (keep your existing)

    settings_info = [
        "",
        "Current Settings:",
        "",
        f"Default Tempo: {tempo} BPM",
        f"Metronome: {'Enabled' if metronome_enabled else 'Disabled'}",
        f"Detection Confidence: 70%",
        f"Max Hands: 2",
        "",
        "Available Instruments:",
        "  - Piano (C4 octave)",
        "  - Guitar (E2-D#3 range)",
        "  - Violin (G2-F#3 range)",
        "  - Cello (C2-B2 octave)",
        "  - Flute (C5 octave)",
        "",
        "Note: Settings can be adjusted during play"
    ]

    y_start = toggle_y + toggle_height + 40
    for i, setting in enumerate(settings_info):
        if setting == "":
            continue
        color = (255,255,255) if not setting.startswith("  ") else (180,180,180)
        cv2.putText(display, setting, (50, y_start + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(display, "Press ESC to return to menu", (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,255,100), 2)

# Update mouse callback to handle looping toggle

def mouse_callback(event, x, y, flags, param):
    global tempo, mouse_dragging, slider_rect, APP_STATE, mouse_x, mouse_y, current_background, loop_enabled

    mouse_x, mouse_y = x, y

    if APP_STATE == "MENU" and event == cv2.EVENT_LBUTTONDOWN:
        for button in menu_buttons:
            bx, by, bw, bh = button["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                APP_STATE = button["action"]
                print(f"Switched to {APP_STATE}")
                break

    elif APP_STATE == "SETTINGS" and event == cv2.EVENT_LBUTTONDOWN:
        clicked_button = False
        for button in settings_buttons:
            bx, by, bw, bh = button["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if button["action"].startswith("COLOR_"):
                    color_name = button["action"].replace("COLOR_", "")
                    current_background = color_name
                    print(f"Background color changed to {color_name}")
                elif button["action"] == "TOGGLE_LOOPING":
                    loop_enabled = not loop_enabled
                    print(f"Looping {'enabled' if loop_enabled else 'disabled'}")
                clicked_button = True
                break
    # Comment out or remove this if you want clicks outside buttons to not exit settings
    # if not clicked_button:
    #     APP_STATE = "MENU"



    elif APP_STATE == "TUTORIAL" and event == cv2.EVENT_LBUTTONDOWN:
        APP_STATE = "MENU"

    elif APP_STATE == "THEREMIN":
        if slider_rect is None:
            return
        slider_x, slider_y, slider_w, slider_h = slider_rect

        if event == cv2.EVENT_LBUTTONDOWN:
            if slider_y <= y <= slider_y + slider_h and slider_x <= x <= slider_x + slider_w:
                mouse_dragging = True
                relative_x = x - slider_x
                tempo = int(55 + (relative_x / slider_w) * (125 - 55))
                tempo = max(55, min(125, tempo))
        elif event == cv2.EVENT_MOUSEMOVE and mouse_dragging:
            if slider_x <= x <= slider_x + slider_w:
                relative_x = x - slider_x
                tempo = int(55 + (relative_x / slider_w) * (125 - 55))
                tempo = max(55, min(125, tempo))
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_dragging = False

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

# (Place your existing full main loop and function definitions here)
# Remember to check 'loop_enabled' before executing loop logic where appropriate.

# Example snippet for gating loop in the theremin loop section:
# if loop_enabled:
#     # your loop recording/playing code here

# Your existing main while loop and additional code continues unchanged...




# Setup - Enable detection of both hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)  # Changed to 2 hands
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()

# Accurate pitch frequencies based on A4=440Hz standard/
def note_to_freq(note, octave):
    """Convert note name and octave to accurate frequency"""
    notes = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 
             'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
    a4 = 440.0
    semitones_from_a4 = notes[note] + (octave - 4) * 12
    return a4 * (2 ** (semitones_from_a4 / 12))

INSTRUMENTS = {
    0: {"name": "Piano", "notes": [("C", 4), ("C#", 4), ("D", 4), ("D#", 4), ("E", 4), ("F", 4), ("F#", 4), ("G", 4), ("G#", 4), ("A", 4), ("A#", 4), ("B", 4)], "color": (255, 255, 255)},
    1: {"name": "Guitar", "notes": [("E", 2), ("F", 2), ("F#", 2), ("G", 2), ("G#", 2), ("A", 2), ("A#", 2), ("B", 2), ("C", 3), ("C#", 3), ("D", 3), ("D#", 3)], "color": (150, 75, 0)},
    2: {"name": "Violin", "notes": [("G", 2), ("G#", 2), ("A", 2), ("A#", 2), ("B", 2), ("C", 3), ("C#", 3), ("D", 3), ("D#", 3), ("E", 3), ("F", 3), ("F#", 3)], "color": (139, 69, 19)},
    3: {"name": "Cello", "notes": [("C", 2), ("C#", 2), ("D", 2), ("D#", 2), ("E", 2), ("F", 2), ("F#", 2), ("G", 2), ("G#", 2), ("A", 2), ("A#", 2), ("B", 2)], "color": (101, 67, 33)},
    4: {"name": "Flute", "notes": [("C", 5), ("C#", 5), ("D", 5), ("D#", 5), ("E", 5), ("F", 5), ("F#", 5), ("G", 5), ("G#", 5), ("A", 5), ("A#", 5), ("B", 5)], "color": (192, 192, 192)},
    5: {"name": "Drums", "notes": [("Kick", 0), ("Snare", 0), ("Hi-Hat", 0), ("Open-Hat", 0), ("Crash", 0), ("Ride", 0), ("Tom1", 0), ("Tom2", 0), ("Tom3", 0), ("Clap", 0), ("Cowbell", 0), ("Shaker", 0)], "color": (255, 100, 100)},
}

current_instrument = 0
last_switch_time = 0
sounds = {}
tempo = 80  # Default tempo (BPM)

metronome_sound = None
last_metronome_time = 0
metronome_enabled = True

last_stable_note = -1  # Track stable note for right hand
note_stability_time = 0  # Time when note became stable
stability_threshold = 0.15  # Seconds to wait before playing new note

APP_STATE = "MENU"  # MENU, TUTORIAL, SETTINGS, THEREMIN
menu_buttons = []
settings_buttons = []

BACKGROUND_COLORS = {
    "Black": (0, 0, 0),
    "Dark Blue": (20, 20, 60),
    "Dark Green": (0, 40, 0),
    "Dark Purple": (40, 0, 40),
    "Dark Red": (40, 0, 0)
}
current_background = "Black"
mouse_x, mouse_y = 0, 0

class LoopState:
    def __init__(self):
        self.state = "IDLE"  # IDLE, ASKING_MEASURES, COUNTDOWN, RECORDING, PLAYING
        self.max_fingers_seen = 0
        
    def reset_fingers(self):
        self.max_fingers_seen = 0

LOOP_STATE = LoopState()

loop_audio_data = []
loop_measures = 0
loop_start_time = 0
countdown_start_time = 0
measure_ask_start_time = 0
current_loop_measure = 0
recorded_audio_buffer = []
sample_rate = 44100

recording_start_time = 0
last_audio_time = 0
audio_chunk_size = 1024  # Process audio in chunks for better continuity

last_gesture_time = {"thumbs_up": 0, "thumbs_down": 0, "peace_sign": 0}
gesture_cooldown = 1.0  # 1 second cooldown between gestures

def generate_metronome_sound():
    """Generate a simple click sound for metronome"""
    sample_rate = 44100
    duration = 0.1  # Short click
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # High frequency click with quick decay
    freq = 1000  # 1kHz click
    wave = np.sin(2 * np.pi * freq * t)
    envelope = np.exp(-t * 50)  # Very quick decay
    wave *= envelope
    
    wave = np.clip(wave, -1, 1)
    wave_int = (wave * 32767).astype(np.int16)
    stereo_wave = np.zeros((len(wave_int), 2), dtype=np.int16)
    stereo_wave[:, 0] = stereo_wave[:, 1] = wave_int
    
    return pygame.sndarray.make_sound(stereo_wave)

def generate_instrument_sound(freq, instrument_name, duration=1.0):
    """Generate realistic instrument sounds using accurate frequencies"""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    if instrument_name == "Piano":
        wave = np.sin(2 * np.pi * freq * t)
        wave += 0.5 * np.sin(2 * np.pi * freq * 2 * t)  # Harmonic
        wave += 0.25 * np.sin(2 * np.pi * freq * 3 * t)  # Third harmonic
        envelope = np.exp(-t * 2.0)  # Piano decay
        volume_multiplier = 0.3
    elif instrument_name == "Guitar":
        wave = np.sin(2 * np.pi * freq * t)
        wave += 0.6 * np.sin(2 * np.pi * freq * 2 * t)
        wave += 0.4 * np.sin(2 * np.pi * freq * 3 * t)
        envelope = np.exp(-t * 1.5)  # Guitar decay
        volume_multiplier = 0.3
    elif instrument_name == "Violin":
        wave = np.sin(2 * np.pi * freq * t)  # Fundamental
        wave += 0.4 * np.sin(2 * np.pi * freq * 2 * t)  # Second harmonic
        wave += 0.2 * np.sin(2 * np.pi * freq * 3 * t)  # Third harmonic
        wave += 0.1 * np.sin(2 * np.pi * freq * 4 * t)  # Fourth harmonic
        envelope = np.minimum(1, t * 8) * np.exp(-t * 0.3)  # Violin bow attack and sustain
        volume_multiplier = 0.35
    elif instrument_name == "Cello":
        wave = np.sin(2 * np.pi * freq * t)
        wave += 0.6 * np.sin(2 * np.pi * freq * 2 * t)
        wave += 0.4 * np.sin(2 * np.pi * freq * 3 * t)
        envelope = np.minimum(1, t * 4) * np.exp(-t * 0.5)  # Cello bow
        volume_multiplier = 0.5
    elif instrument_name == "Flute":
        wave = np.sin(2 * np.pi * freq * t)
        wave += 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        wave += 0.05 * (np.random.random(len(t)) - 0.5) * np.exp(-t * 10)  # Breath noise
        envelope = np.minimum(1, t * 6) * np.exp(-t * 0.2)
        volume_multiplier = 0.3
    elif instrument_name == "Drums":
        # Drums use shorter duration for quick hits
        drum_duration = 0.3
        t_drum = np.linspace(0, drum_duration, int(sample_rate * drum_duration), False)
    
    wave *= envelope * volume_multiplier
    wave = np.clip(wave, -1, 1)
    
    wave_int = (wave * 32767).astype(np.int16)
    stereo_wave = np.zeros((len(wave_int), 2), dtype=np.int16)
    stereo_wave[:, 0] = stereo_wave[:, 1] = wave_int
    
    return pygame.sndarray.make_sound(stereo_wave)

def generate_continuous_audio_chunk(freq, instrument_name, duration, phase_offset=0):
    """Generate a continuous audio chunk with proper phase continuity"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    t_with_phase = t + phase_offset
    
    if instrument_name == "Piano":
        wave = np.sin(2 * np.pi * freq * t_with_phase)
        wave += 0.5 * np.sin(2 * np.pi * freq * 2 * t_with_phase)
        wave += 0.25 * np.sin(2 * np.pi * freq * 3 * t_with_phase)
        volume_multiplier = 0.3
    elif instrument_name == "Guitar":
        wave = np.sin(2 * np.pi * freq * t_with_phase)
        wave += 0.6 * np.sin(2 * np.pi * freq * 2 * t_with_phase)
        wave += 0.4 * np.sin(2 * np.pi * freq * 3 * t_with_phase)
        volume_multiplier = 0.3
    elif instrument_name == "Violin":
        wave = np.sin(2 * np.pi * freq * t_with_phase)
        wave += 0.4 * np.sin(2 * np.pi * freq * 2 * t_with_phase)
        wave += 0.2 * np.sin(2 * np.pi * freq * 3 * t_with_phase)
        wave += 0.1 * np.sin(2 * np.pi * freq * 4 * t_with_phase)
        volume_multiplier = 0.35
    elif instrument_name == "Cello":
        wave = np.sin(2 * np.pi * freq * t_with_phase)
        wave += 0.6 * np.sin(2 * np.pi * freq * 2 * t_with_phase)
        wave += 0.4 * np.sin(2 * np.pi * freq * 3 * t_with_phase)
        volume_multiplier = 0.5
    elif instrument_name == "Flute":
        wave = np.sin(2 * np.pi * freq * t_with_phase)
        wave += 0.2 * np.sin(2 * np.pi * freq * 2 * t_with_phase)
        volume_multiplier = 0.3
    else:
        wave = np.sin(2 * np.pi * freq * t_with_phase)
        volume_multiplier = 0.3
    
    wave *= volume_multiplier
    wave = np.clip(wave, -1, 1)
    
    return wave

def load_instrument_sounds(instrument_id):
    """Load sounds for selected instrument"""
    global sounds
    sounds = {}
    instrument = INSTRUMENTS[instrument_id]
    
    for i, (note, octave) in enumerate(instrument["notes"]):
        freq = note_to_freq(note, octave)
        sound = generate_instrument_sound(freq, instrument["name"])
        sounds[i] = sound
        print(f"Loaded {instrument['name']} - {note}{octave} ({freq:.1f} Hz)")

def map_position_to_note(x, y, w, h):
    """Map hand position to note index (theremin style)"""
    # X position controls pitch (0-7 notes)
    note_index = int((x / w) * len(INSTRUMENTS[current_instrument]["notes"]))
    note_index = max(0, min(note_index, len(INSTRUMENTS[current_instrument]["notes"]) - 1))
    
    # Y position controls volume (top = loud, bottom = quiet)
    volume = 1.0 - (y / h)  # Invert Y axis
    volume = max(0.1, min(volume, 1.0))
    
    return note_index, volume

def detect_fist(hand_landmarks):
    """Simple fist detection for muting"""
    tips = [4, 8, 12, 16, 20]
    mcps = [3, 5, 9, 13, 17]
    closed = 0
    
    for i in range(5):
        if hand_landmarks.landmark[tips[i]].y > hand_landmarks.landmark[mcps[i]].y:
            closed += 1
    return closed >= 4

def detect_thumbs_up(hand_landmarks):
    """Detect thumbs up gesture with stricter criteria"""
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    thumb_cmc = hand_landmarks.landmark[1]
    
    thumb_up = (thumb_tip.y < thumb_ip.y < thumb_mcp.y < thumb_cmc.y and 
                thumb_tip.y < thumb_mcp.y - 0.05)  # Significant vertical separation
    
    # Check if other fingers are clearly closed
    fingers_closed = 0
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y + 0.02:  # More strict closure
            fingers_closed += 1
    
    return thumb_up and fingers_closed == 4

def detect_thumbs_down(hand_landmarks):
    """Detect thumbs down gesture with improved criteria"""
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    
    thumb_down = (thumb_tip.y > thumb_ip.y + 0.03 and 
                  thumb_ip.y > thumb_mcp.y + 0.02 and
                  thumb_tip.y > thumb_mcp.y + 0.05)
    
    # Check if other fingers are more closed (less strict)
    fingers_closed = 0
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y + 0.01:
            fingers_closed += 1
    
    return thumb_down and fingers_closed >= 1  # Less strict requirement

def detect_peace_sign(hand_landmarks):
    """Detect peace sign (V) gesture"""
    # Check if index and middle fingers are extended
    index_extended = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_extended = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    
    # Check if ring and pinky are closed
    ring_closed = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_closed = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    
    return index_extended and middle_extended and ring_closed and pinky_closed

def count_fingers(hand_landmarks):
    """Count the number of extended fingers"""
    fingers = 0
    
    # Thumb
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    
    if abs(thumb_tip.x - thumb_ip.x) > 0.04:  # Thumb is extended horizontally
        fingers += 1
    
    # Other fingers - check if tip is above PIP joint
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y - 0.02:  # Added threshold
            fingers += 1
    
    return fingers

def map_right_hand_to_pitch(y, h):
    """Map right hand Y position to instrument's natural note range"""
    instrument = INSTRUMENTS[current_instrument]
    # Normalize Y position with better precision for full 12-note access
    normalized_y = 1.0 - (y / h)  # 1.0 at top, 0.0 at bottom
    
    # Add small buffer zones at top and bottom to make extreme notes easier to hit
    buffer = 0.05
    if normalized_y > (1.0 - buffer):
        normalized_y = 1.0
    elif normalized_y < buffer:
        normalized_y = 0.0
    else:
        # Rescale the middle range to use full 0-1 range
        normalized_y = (normalized_y - buffer) / (1.0 - 2 * buffer)
    
    # Map to full note range with improved precision
    pitch_index = int(normalized_y * len(instrument["notes"]))
    pitch_index = max(0, min(pitch_index, len(instrument["notes"]) - 1))
    return pitch_index

def map_left_hand_to_volume(y, h):
    """Map left hand Y position to volume (top = loud, bottom = quiet)"""
    volume = 1.0 - (y / h)  # 1.0 at top, 0.0 at bottom
    volume = max(0.0, min(volume, 1.0))
    return volume

def export_loop_audio():
    """Export the recorded loop audio to a WAV file"""
    if not loop_audio_data:
        print("❌ No audio to export")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"theremin_loop_{timestamp}.wav"
    
    max_length = max(len(layer) for layer in loop_audio_data) if loop_audio_data else 0
    if max_length == 0:
        print("❌ No audio data to export")
        return False
    
    try:
        # Create mixed audio by adding all layers together
        mixed_audio = np.zeros(max_length, dtype=np.float64)
        for layer in loop_audio_data:
            if len(layer) > 0:
                layer_array = np.array(layer, dtype=np.float64)
                # Pad or trim to match max_length
                if len(layer_array) < max_length:
                    layer_array = np.pad(layer_array, (0, max_length - len(layer_array)))
                else:
                    layer_array = layer_array[:max_length]
                mixed_audio += layer_array
        
        # Normalize to prevent clipping with better headroom
        if np.max(np.abs(mixed_audio)) > 0:
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.7  # More conservative normalization
        
        audio_int = (mixed_audio * 32767).astype(np.int16)
        
        # Save as WAV file
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int.tobytes())
        
        print(f"✅ Loop exported as {filename}")
        return True
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def clear_loop():
    """Clear the current loop"""
    global LOOP_STATE, loop_audio_data, recorded_audio_buffer, current_loop_measure
    loop_audio_data = []
    recorded_audio_buffer = []
    current_loop_measure = 0
    print("🗑️ Loop cleared")

def draw_menu(display, w, h):
    global menu_buttons
    menu_buttons = []

    display[:] = BACKGROUND_COLORS[current_background]

    # Draw logo image at top center
    if logo_img is not None:
        scale = (w * 0.5) / logo_img.shape[1]
        logo_resized = cv2.resize(logo_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        y_offset = 30
        x_offset = (w - logo_resized.shape[1]) // 2
        if logo_resized.shape[2] == 4:
            alpha_s = logo_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                display[y_offset:y_offset+logo_resized.shape[0], x_offset:x_offset+logo_resized.shape[1], c] = (
                  alpha_s * logo_resized[:, :, c] +
                  alpha_l * display[y_offset:y_offset+logo_resized.shape[0], x_offset:x_offset+logo_resized.shape[1], c])
        else:
            display[y_offset:y_offset+logo_resized.shape[0], x_offset:x_offset+logo_resized.shape[1]] = logo_resized
    else:
        title = "Theremin Gesture Player"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(display, title, (title_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Buttons below the logo or title
    button_width = 200
    button_height = 60
    button_x = (w - button_width) // 2
    button_spacing = 80
    base_y = 100 + (logo_resized.shape[0] if logo_img is not None else 0) + 40

    # Shades of Gold in increasing brightness (BGR tuples)
    gold_shades = [
        (20, 120, 220),   # Dark Gold
        (40, 160, 255),   # Medium Gold
        (80, 200, 255)    # Light Gold
    ]

    buttons = [
        {"text": "START", "y": base_y, "color": gold_shades[0], "hover_color": (60, 220, 255), "action": "THEREMIN"},
        {"text": "TUTORIAL", "y": base_y + button_spacing, "color": gold_shades[1], "hover_color": (90, 240, 255), "action": "TUTORIAL"},
        {"text": "SETTINGS", "y": base_y + button_spacing * 2, "color": gold_shades[2], "hover_color": (120, 255, 255), "action": "SETTINGS"}
    ]

    for button in buttons:
        y = button["y"]
        is_hovering = (button_x <= mouse_x <= button_x + button_width and y <= mouse_y <= y + button_height)
        button_color = button["hover_color"] if is_hovering else button["color"]
        cv2.rectangle(display, (button_x, y), (button_x + button_width, y + button_height), button_color, -1)
        cv2.rectangle(display, (button_x, y), (button_x + button_width, y + button_height), (255, 255, 255), 2)
        text_size = cv2.getTextSize(button["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = button_x + (button_width - text_size[0]) // 2
        text_y = y + (button_height + text_size[1]) // 2
        cv2.putText(display, button["text"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        menu_buttons.append({"rect": (button_x, y, button_width, button_height), "action": button["action"]})

    instructions = [
        "Click on a button to continue",
        "Press ESC to quit"
    ]
    for i, instruction in enumerate(instructions):
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(display, instruction, (text_x, h - 80 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)



def draw_tutorial(display, w, h):
    """Draw the tutorial screen"""
    display[:] = BACKGROUND_COLORS[current_background]
    
    # Title
    title = "How to Play"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    title_x = (w - title_size[0]) // 2
    cv2.putText(display, title, (title_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Tutorial content
    instructions = [
        "LEFT HAND (Cyan): Controls volume only",
        "  - Move up for loud, down for quiet",
        "",
        "RIGHT HAND (Magenta): Controls pitch only", 
        "  - Move up for high notes, down for low notes",
        "  - Make a fist to mute",
        "",
        "INSTRUMENT SELECTION:",
        "  - Point either hand at top buttons to switch",
        "",
        "TEMPO CONTROL:",
        "  - Click and drag the tempo slider",
        "",
        "LOOP CONTROL:",
        "  - Thumbs up to start a new loop",
        "  - Thumbs down to export and clear loop",
        "  - Peace sign to clear loop",
        "",
        "CONTROLS:",
        "  - Press 'M' to toggle metronome",
        "  - Press ESC to return to menu"
    ]
    
    y_start = 120
    for i, instruction in enumerate(instructions):
        if instruction == "":
            continue
        color = (255, 255, 255) if not instruction.startswith("  ") else (180, 180, 180)
        cv2.putText(display, instruction, (50, y_start + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Back button
    cv2.putText(display, "Press anywhere on the screen to return to menu", (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

def draw_settings(display, w, h):
    """Draw the settings screen"""
    global settings_buttons
    settings_buttons = []
    
    display[:] = BACKGROUND_COLORS[current_background]
    
    # Title
    title = "Settings"
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    title_x = (w - title_size[0]) // 2
    cv2.putText(display, title, (title_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    cv2.putText(display, "Background Color:", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Color selection buttons
    color_button_width = 100
    color_button_height = 30
    color_start_x = 50
    color_start_y = 140
    
    for i, (color_name, color_value) in enumerate(BACKGROUND_COLORS.items()):
        x = color_start_x + (i % 3) * (color_button_width + 10)
        y = color_start_y + (i // 3) * (color_button_height + 10)
        
        # Check if mouse is hovering over color button
        is_hovering = (x <= mouse_x <= x + color_button_width and 
                      y <= mouse_y <= y + color_button_height)
        is_selected = (color_name == current_background)
        
        # Button color - brighter if hovering, highlighted if selected
        if is_selected:
            button_color = (100, 100, 100)
            border_color = (255, 255, 0)  # Yellow border for selected
            border_thickness = 3
        elif is_hovering:
            button_color = tuple(min(255, c + 50) for c in color_value)
            border_color = (255, 255, 255)
            border_thickness = 2
        else:
            button_color = color_value
            border_color = (128, 128, 128)
            border_thickness = 1
        
        # Draw color button
        cv2.rectangle(display, (x, y), (x + color_button_width, y + color_button_height), button_color, -1)
        cv2.rectangle(display, (x, y), (x + color_button_width, y + color_button_height), border_color, border_thickness)
        
        # Draw color name
        text_size = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = x + (color_button_width - text_size[0]) // 2
        text_y = y + (color_button_height + text_size[1]) // 2
        text_color = (255, 255, 255) if sum(color_value) < 200 else (0, 0, 0)
        cv2.putText(display, color_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Store button info for click detection
        settings_buttons.append({
            "rect": (x, y, color_button_width, color_button_height),
            "action": f"COLOR_{color_name}"
        })
    
    # Settings content
    settings_info = [
        "",
        "Current Settings:",
        "",
        f"Default Tempo: {tempo} BPM",
        f"Metronome: {'Enabled' if metronome_enabled else 'Disabled'}",
        f"Detection Confidence: 70%",
        f"Max Hands: 2",
        "",
        "Available Instruments:",
        "  - Piano (C4 octave)",
        "  - Guitar (E2-D#3 range)", 
        "  - Violin (G2-F#3 range)",
        "  - Cello (C2-B2 octave)",
        "  - Flute (C5 octave)",
        "",
        "Note: Settings can be adjusted during play"
    ]
    
    y_start = 240
    for i, setting in enumerate(settings_info):
        if setting == "":
            continue
        color = (255, 255, 255) if not setting.startswith("  ") else (180, 180, 180)
        cv2.putText(display, setting, (50, y_start + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Back button
    cv2.putText(display, "Press ESC to return to menu", (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    cv2.rectangle(display, (50, 300), (250, 350), (0, 255, 255), -1)
    cv2.putText(display, "Toggle Here", (60, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    # After toggling in mouse_callback, add a global flag:
    loop_toggle_feedback_time = time.time()

    # In draw_settings, add near bottom:
    if time.time() - loop_toggle_feedback_time < 1.0:  # show for 1 second
        feedback_text = f"Looping {'Enabled' if loop_enabled else 'Disabled'}"
        cv2.putText(display, feedback_text, ((w // 2) - 100, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)



def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for tempo slider and menu buttons"""
    global tempo, mouse_dragging, slider_rect, APP_STATE, mouse_x, mouse_y, current_background
    
    mouse_x, mouse_y = x, y
    
    if APP_STATE == "MENU" and event == cv2.EVENT_LBUTTONDOWN:
        # Check menu button clicks
        for button in menu_buttons:
            bx, by, bw, bh = button["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                APP_STATE = button["action"]
                print(f"Switched to {APP_STATE}")
                break
    
    elif APP_STATE == "SETTINGS" and event == cv2.EVENT_LBUTTONDOWN:
        clicked_button = False
        for button in settings_buttons:
            bx, by, bw, bh = button["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if button["action"].startswith("COLOR_"):
                    color_name = button["action"].replace("COLOR_", "")
                    current_background = color_name
                    print(f"Background color changed to {color_name}")
                elif button["action"] == "TOGGLE_LOOPING":
                    loop_enabled = not loop_enabled
                    print(f"Looping {'enabled' if loop_enabled else 'disabled'}")
                clicked_button = True
                break
        # If clicked elsewhere in settings, return to menu
        # if not clicked_button:
        #     APP_STATE = "MENU"
    
    elif APP_STATE == "TUTORIAL" and event == cv2.EVENT_LBUTTONDOWN:
        APP_STATE = "MENU"
    
    # Original slider logic for theremin mode
    elif APP_STATE == "THEREMIN":
        if slider_rect is None:
            return
            
        slider_x, slider_y, slider_w, slider_h = slider_rect
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if slider_y <= y <= slider_y + slider_h and slider_x <= x <= slider_x + slider_w:
                mouse_dragging = True
                relative_x = x - slider_x
                tempo = int(55 + (relative_x / slider_w) * (125 - 55))
                tempo = max(55, min(125, tempo))
        
        elif event == cv2.EVENT_MOUSEMOVE and mouse_dragging:
            if slider_x <= x <= slider_x + slider_w:
                relative_x = x - slider_x
                tempo = int(55 + (relative_x / slider_w) * (125 - 55))
                tempo = max(55, min(125, tempo))
        
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_dragging = False

cv2.namedWindow("Maestro")
cv2.setMouseCallback("Maestro", mouse_callback)

print("Theremin Gesture Player Ready!")
print("Navigate through the menu to start playing!")

# Load initial sounds
load_instrument_sounds(current_instrument)

metronome_sound = generate_metronome_sound()

mouse_dragging = False
slider_rect = None

last_note = -1  # Track only one note since only right hand plays
last_play_time = 0  # Track only one play time
current_volume = 0.5  # Single volume controlled by left hand
current_pitch_index = 0  # Single pitch controlled by right hand
hand_muted = [False, False]  # Individual muting for each hand

current_phase = 0.0
last_freq = 440.0

def play_realtime_audio(freq, instrument_name, volume, duration=0.1):
    """Play audio in real-time with minimal latency"""
    if volume <= 0:
        return
    
    try:
        # Generate short audio chunk for immediate playback
        audio_chunk = generate_continuous_audio_chunk(freq, instrument_name, duration)
        audio_int = (audio_chunk * 32767).astype(np.int16)
        stereo_wave = np.zeros((len(audio_int), 2), dtype=np.int16)
        stereo_wave[:, 0] = stereo_wave[:, 1] = audio_int
        
        sound = pygame.sndarray.make_sound(stereo_wave)
        sound.play()
    except Exception as e:
        print(f"Audio playback error: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Create clean display with selected background color
    display = np.full((h, w, 3), BACKGROUND_COLORS[current_background], dtype=np.uint8)
    
    if APP_STATE == "MENU":
        draw_menu(display, w, h)
    elif APP_STATE == "TUTORIAL":
        draw_tutorial(display, w, h)
    elif APP_STATE == "SETTINGS":
        draw_settings(display, w, h)
    elif APP_STATE == "THEREMIN":
        # Original theremin code with background color
        current_time = time.time()
        beat_interval = 60.0 / tempo
        
        if metronome_enabled and current_time - last_metronome_time >= beat_interval:
            metronome_sound.set_volume(current_volume * 0.3)
            metronome_sound.play()
            last_metronome_time = current_time
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Draw instrument selector (top strip)
        button_width = w // len(INSTRUMENTS)
        for i, (key, instrument) in enumerate(INSTRUMENTS.items()):
            x1, x2 = i * button_width, (i + 1) * button_width
            color = instrument["color"] if i == current_instrument else (40, 40, 40)
            cv2.rectangle(display, (x1, 0), (x2, 40), color, -1)
            cv2.rectangle(display, (x1, 0), (x2, 40), (255, 255, 255), 1)
            
            text_size = cv2.getTextSize(instrument["name"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x1 + (button_width - text_size[0]) // 2
            cv2.putText(display, instrument["name"], (text_x, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        left_hand_detected = False
        right_hand_detected = False
        left_hand_fist = False
        left_thumbs_up = False
        left_thumbs_down = False
        left_peace_sign = False
        finger_count = 0
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_idx, (hand_lms, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                if not handedness.classification:
                    continue
                    
                is_right_hand = handedness.classification[0].label == "Right"
                hand_color = (255, 0, 255) if is_right_hand else (0, 255, 255)  # Magenta for right, cyan for left
                hand_index = 1 if is_right_hand else 0
                
                index_tip = hand_lms.landmark[8]
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                
                if is_right_hand:
                    right_hand_detected = True
                    if y > 50:  # Below instrument selector
                        current_pitch_index = map_right_hand_to_pitch(y - 50, h - 50)
                    
                    if y < 40 and time.time() - last_switch_time > 1.0:
                        new_instrument = min(len(INSTRUMENTS)-1, x // button_width)
                        if new_instrument != current_instrument:
                            current_instrument = new_instrument
                            last_switch_time = time.time()
                            threading.Thread(target=load_instrument_sounds, 
                                           args=(current_instrument,), daemon=True).start()
                            print(f"Switched to {INSTRUMENTS[current_instrument]['name']}")
                else:
                    left_hand_detected = True
                    
                    if detect_fist(hand_lms):
                        left_hand_fist = True
                        current_volume = 0.0
                    elif detect_thumbs_up(hand_lms):
                        left_thumbs_up = True
                    elif detect_thumbs_down(hand_lms):
                        left_thumbs_down = True
                    elif detect_peace_sign(hand_lms):
                        left_peace_sign = True
                    else:
                        if y > 50:
                            current_volume = map_left_hand_to_volume(y - 50, h - 50)
                        finger_count = count_fingers(hand_lms)
                
                cv2.circle(display, (x, y), 12, hand_color, -1)
                cv2.circle(display, (x, y), 15, (255, 255, 255), 2)
                
                if hand_muted[hand_index]:
                    cv2.putText(display, "MUTED", (x-25, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    cv2.putText(display, "R" if is_right_hand else "L", 
                               (x-5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)
                
                mp_draw.draw_landmarks(display, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        if left_hand_detected and not hand_muted[0]:
            cv2.putText(display, f"Volume: {int(current_volume * 100)}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if right_hand_detected and not hand_muted[1]:
            note_info = INSTRUMENTS[current_instrument]["notes"][current_pitch_index]
            note_name = f"{note_info[0]}{note_info[1]}"
            cv2.putText(display, f"Note: {note_name}", 
                       (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Metronome status
        metronome_status = "ON" if metronome_enabled else "OFF"
        cv2.putText(display, f"Metronome: {metronome_status}", (10, h-100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if metronome_enabled else (128, 128, 128), 1)
        
        # Draw slider background
        slider_x, slider_y = w - 250, h - 50
        slider_w, slider_h = 200, 20
        slider_rect = (slider_x, slider_y, slider_w, slider_h)
        cv2.rectangle(display, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), (60, 60, 60), -1)
        cv2.rectangle(display, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), (255, 255, 255), 1)
        
        # Draw slider handle
        handle_pos = int(slider_x + ((tempo - 55) / (125 - 55)) * slider_w)
        cv2.circle(display, (handle_pos, slider_y + slider_h // 2), 8, (0, 255, 0), -1)
        cv2.circle(display, (handle_pos, slider_y + slider_h // 2), 8, (255, 255, 255), 2)
        
        # Draw tempo labels
        cv2.putText(display, f"Tempo: {tempo} BPM", (slider_x, slider_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "55", (slider_x - 20, slider_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        cv2.putText(display, "125", (slider_x + slider_w + 5, slider_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # Draw pitch/volume guides
        cv2.line(display, (0, 50), (w, 50), (100, 100, 100), 1)
        cv2.putText(display, "Highest Note", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(display, "Lowest Note", (w-120, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(display, "Loud", (10, h-120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(display, "Soft", (10, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        zone_height = (h - 50) // 12
        for i in range(12):
            zone_y = 50 + (i * zone_height)
            cv2.line(display, (w-30, zone_y), (w-20, zone_y), (80, 80, 80), 1)
            if i % 3 == 0:  # Mark every 3rd note more prominently
                cv2.line(display, (w-35, zone_y), (w-15, zone_y), (120, 120, 120), 2)
        
        if right_hand_detected and not hand_muted[1] and left_hand_detected and not hand_muted[0]:
            # Only play if both hands are present and not muted
            if current_pitch_index != last_stable_note:
                last_stable_note = current_pitch_index
                note_stability_time = current_time
            elif current_time - note_stability_time >= stability_threshold:
                if current_pitch_index in sounds:
                    sounds[current_pitch_index].set_volume(current_volume)
                    sounds[current_pitch_index].play()
                    last_note = current_pitch_index
                    last_play_time = current_time

        if LOOP_STATE.state == "IDLE":
            loop_enabled=False ####EDIT THIS IF YOU WANT LOOPING SKY__
            if left_thumbs_up and loop_enabled:
                if not loop_audio_data:  # No existing loop
                    LOOP_STATE.state = "ASKING_MEASURES"
                    measure_ask_start_time = current_time
                    print("Starting new loop - How many measures?")
                else:  # Add to existing loop
                    LOOP_STATE.state = "COUNTDOWN"
                    countdown_start_time = current_time
                    print("Adding to existing loop")
        
        elif LOOP_STATE.state == "ASKING_MEASURES":
            time_elapsed = current_time - measure_ask_start_time
            if time_elapsed < 5.0:  # Extended time to 5 seconds for better input
                # Display "How many measures?" message
                cv2.putText(display, "How many measures?", (w//2-150, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(display, f"Show fingers: {finger_count}", (w//2-100, h//2+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display, f"Time: {5-int(time_elapsed)}", (w//2-50, h//2+80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                LOOP_STATE.max_fingers_seen = max(LOOP_STATE.max_fingers_seen, finger_count)
            else:
                loop_measures = max(1, LOOP_STATE.max_fingers_seen)
                LOOP_STATE.state = "COUNTDOWN"
                countdown_start_time = current_time
                print(f"Loop will be {loop_measures} measures")
                # Reset the max fingers tracking
                LOOP_STATE.reset_fingers()
        
        elif LOOP_STATE.state == "COUNTDOWN":
            time_elapsed = current_time - countdown_start_time
            if time_elapsed < 3.0:
                countdown_num = 3 - int(time_elapsed)
                cv2.putText(display, str(countdown_num), (w//2-20, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            else:
                LOOP_STATE.state = "RECORDING"
                loop_start_time = current_time
                recording_start_time = current_time
                last_audio_time = current_time
                current_loop_measure = 0
                recorded_audio_buffer = []
                current_phase = 0.0  # Reset phase for new recording
                print("Recording loop...")
        
        elif LOOP_STATE.state == "RECORDING":
            time_elapsed = current_time - loop_start_time
            beats_per_minute = tempo
            seconds_per_beat = 60.0 / beats_per_minute
            measure_duration = seconds_per_beat * 4  # 4 beats per measure in 4/4 time
            total_duration = measure_duration * loop_measures
            
            # Display recording status
            cv2.putText(display, "RECORDING LOOP", (w//2-100, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            progress = min(100, (time_elapsed / total_duration) * 100)
            cv2.putText(display, f"Progress: {int(progress)}%", (w//2-80, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            current_measure = min(loop_measures, int(time_elapsed / measure_duration) + 1)
            cv2.putText(display, f"Measure: {current_measure}/{loop_measures}", (w//2-80, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            remaining_time = max(0, total_duration - time_elapsed)
            cv2.putText(display, f"Time left: {remaining_time:.1f}s", (w//2-80, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Generate continuous audio chunks and play in real-time
            chunk_duration = current_time - last_audio_time
            if chunk_duration > 0.02:  # Process chunks every 20ms for smoother audio
                if left_hand_detected and right_hand_detected and not left_hand_fist and current_volume > 0:
                    # Get the frequency for current note
                    note_info = INSTRUMENTS[current_instrument]["notes"][current_pitch_index]
                    freq = note_to_freq(note_info[0], note_info[1])
                    
                    instrument_name = INSTRUMENTS[current_instrument]["name"]
                    play_realtime_audio(freq, instrument_name, current_volume, chunk_duration)
                    
                    # Generate continuous audio chunk with phase continuity for recording
                    audio_chunk = generate_continuous_audio_chunk(
                        freq, instrument_name, chunk_duration, current_phase
                    )
                    
                    # Update phase for continuity
                    current_phase += 2 * np.pi * freq * chunk_duration
                    current_phase = current_phase % (2 * np.pi)  # Keep phase in reasonable range
                    last_freq = freq
                    
                    recorded_audio_buffer.extend(audio_chunk)
                else:
                    # Add silence for the chunk duration
                    silence_samples = int(sample_rate * chunk_duration)
                    recorded_audio_buffer.extend(np.zeros(silence_samples))
                    current_phase = 0.0  # Reset phase when silent
                
                last_audio_time = current_time
            
            print(f"[v0] Recording: {time_elapsed:.1f}s / {total_duration:.1f}s (Measure {int(time_elapsed/measure_duration)+1}/{loop_measures})")
            
            if time_elapsed >= total_duration:
                LOOP_STATE.state = "IDLE"
                if recorded_audio_buffer:
                    loop_audio_data.append(recorded_audio_buffer.copy())
                    recorded_audio_buffer = []
                    print(f"Loop recording complete! Total loops: {len(loop_audio_data)}")
                    print(f"[v0] Recorded {len(loop_audio_data[-1])} audio samples")

        # Handle loop control gestures with debouncing
        current_time = time.time()
        
        if left_thumbs_up and current_time - last_gesture_time["thumbs_up"] > gesture_cooldown:
            last_gesture_time["thumbs_up"] = current_time
            if LOOP_STATE.state == "IDLE" and loop_enabled:
                if not loop_audio_data:  # No existing loop
                    LOOP_STATE.state = "ASKING_MEASURES"
                    measure_ask_start_time = current_time
                    print("Starting new loop - How many measures?")
                else:  # Add to existing loop
                    LOOP_STATE.state = "COUNTDOWN"
                    countdown_start_time = current_time
                    print("Adding to existing loop")
        
        if left_thumbs_down and current_time - last_gesture_time["thumbs_down"] > gesture_cooldown:
            last_gesture_time["thumbs_down"] = current_time
            print(f"[v0] Thumbs down detected! LOOP_STATE: {LOOP_STATE.state}, loop_audio_data length: {len(loop_audio_data)}")
            
            if LOOP_STATE.state == "IDLE" and loop_audio_data:
                print("[v0] Attempting to export loop...")
                try:
                    export_loop_audio()
                    clear_loop()
                    print("[v0] Export successful!")
                except Exception as e:
                    print(f"[v0] Export failed: {e}")
            elif LOOP_STATE.state == "RECORDING":
                print("[v0] Stopping recording early...")
                LOOP_STATE.state = "IDLE"
                if recorded_audio_buffer:
                    loop_audio_data.append(recorded_audio_buffer.copy())
                    recorded_audio_buffer = []
                    print(f"[v0] Early recording saved! Total loops: {len(loop_audio_data)}")
        
        if left_peace_sign and current_time - last_gesture_time["peace_sign"] > gesture_cooldown:
            last_gesture_time["peace_sign"] = current_time
            if LOOP_STATE.state == "IDLE" and loop_audio_data:
                print("Peace sign detected - clearing loop without export")
                clear_loop()
            elif LOOP_STATE.state == "RECORDING":
                print("Peace sign detected - canceling recording")
                LOOP_STATE.state = "IDLE"
                recorded_audio_buffer = []
                print("Recording canceled")

        # Display loop status
        if loop_audio_data:
            cv2.putText(display, f"Loops: {len(loop_audio_data)}", (10, h-120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, "Thumbs down: Export & Clear", (10, h-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.putText(display, "Peace sign: Clear only", (10, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        cv2.putText(display, f"Loop State: {LOOP_STATE.state}", (10, h-90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Theremin Gesture Player", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        if APP_STATE == "THEREMIN" or APP_STATE == "SETTINGS":
            APP_STATE = "MENU"
            print("Returned to menu")
        else:
            break
    elif key == ord('m') or key == ord('M'):
        if APP_STATE == "THEREMIN":
            metronome_enabled = not metronome_enabled
            print(f"Metronome {'enabled' if metronome_enabled else 'disabled'}")

# Cleanup
pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows() 