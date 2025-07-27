"""
Configuration settings for Blackjack Analyzer
"""

# Camera settings
CAMERA_INDEX = 0  # Default camera (0 for built-in, 1+ for external)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Card detection parameters
MIN_CARD_AREA = 3000  # Minimum card size in pixels
MAX_CARD_AREA = 50000  # Maximum card size in pixels
CARD_ASPECT_RATIO = 1.4  # Standard playing card ratio (63mm x 88mm)
ASPECT_RATIO_TOLERANCE = 0.2

# Detection confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.9  # Green indicator
MEDIUM_CONFIDENCE_THRESHOLD = 0.7  # Yellow indicator
LOW_CONFIDENCE_THRESHOLD = 0.5  # Red indicator (below this, ignore)

# Image processing parameters
GAUSSIAN_KERNEL_SIZE = (5, 5)
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11
ADAPTIVE_THRESHOLD_CONSTANT = 2

# Card counting settings
INITIAL_DECK_COUNT = 6  # Standard shoe size
HI_LO_VALUES = {
    '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
    '7': 0, '8': 0, '9': 0,
    '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
}

# Betting spread (true count: bet units)
BETTING_SPREAD = {
    -2: 1,   # True count -2 or lower
    -1: 1,   # True count -1
    0: 1,    # True count 0
    1: 2,    # True count +1
    2: 4,    # True count +2
    3: 6,    # True count +3
    4: 8,    # True count +4
    5: 10    # True count +5 or higher
}

# UI settings
UI_FONT_SCALE = 0.7
UI_FONT_THICKNESS = 2
UI_LINE_THICKNESS = 2
UI_OVERLAY_ALPHA = 0.7  # Transparency (0=invisible, 1=opaque)
UI_MARGIN = 20
UI_PANEL_HEIGHT = 180
UI_STATS_WIDTH = 300

# Color scheme (BGR format for OpenCV)
COLORS = {
    'primary': (0, 255, 0),      # Green
    'secondary': (255, 255, 0),   # Yellow
    'danger': (0, 0, 255),        # Red
    'info': (255, 0, 0),          # Blue
    'warning': (0, 165, 255),     # Orange
    'success': (0, 255, 0),       # Green
    'background': (0, 0, 0),      # Black
    'text': (255, 255, 255)       # White
}

# Performance settings
DETECTION_THREAD_QUEUE_SIZE = 2
FPS_CALCULATION_WINDOW = 30  # Number of frames to average for FPS
MAX_CARD_HISTORY = 100  # Maximum detected cards to keep in history

# Debug settings
DEBUG_MODE = False  # Show additional debug information
SAVE_DETECTED_CARDS = False  # Save detected card images
DETECTED_CARDS_DIR = "detected_cards/"

# Calibration settings
AUTO_CALIBRATE = True  # Automatically calibrate on startup
CALIBRATION_FRAMES = 30  # Number of frames to use for calibration
BRIGHTNESS_ADJUSTMENT_RANGE = (0.5, 1.5)
CONTRAST_ADJUSTMENT_RANGE = (0.5, 1.5)

# Strategy settings
ENABLE_INDEX_PLAYS = True  # Enable count-based strategy deviations
SHOW_EXPECTED_VALUE = True  # Display EV calculations
SHOW_REASONING = True  # Display strategy reasoning

# Audio feedback (requires additional setup)
ENABLE_AUDIO = False  # Enable audio feedback for actions
AUDIO_VOLUME = 0.7

# Logging
ENABLE_LOGGING = False
LOG_FILE = "blackjack_analyzer.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR