import cv2
import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading
from queue import Queue

# Import our modules
from card_detector import CardDetector
from strategy_engine import StrategyEngine, GameState
from ui_overlay import UIOverlay
import config

@dataclass
class DetectedCard:
    """Represents a detected card with its properties"""
    rank: str
    suit: str
    confidence: float
    position: Tuple[int, int, int, int]  # x, y, w, h
    timestamp: float

class BlackjackAnalyzer:
    """Main application class for real-time blackjack analysis"""
    
    def __init__(self):
        self.card_detector = CardDetector()
        self.strategy_engine = StrategyEngine()
        self.ui_overlay = UIOverlay()
        
        # Game state tracking
        self.player_cards = []
        self.dealer_cards = []
        self.detected_cards_history = deque(maxlen=config.MAX_CARD_HISTORY)
        self.running_count = 0
        self.true_count = 0
        self.decks_remaining = config.INITIAL_DECK_COUNT
        
        # Performance tracking
        self.fps = 0
        self.detection_accuracy = 0.95
        self.frame_times = deque(maxlen=config.FPS_CALCULATION_WINDOW)
        
        # Threading for performance
        self.detection_queue = Queue(maxsize=config.DETECTION_THREAD_QUEUE_SIZE)
        self.result_queue = Queue()
        self.detection_thread = None
        
    def start(self):
        """Start the blackjack analyzer"""
        # Initialize camera
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if config.CAMERA_FPS:
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self._detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        print("Blackjack Analyzer Started!")
        print("Press 'q' to quit")
        print("Press 'r' to reset game state")
        print("Press 'c' to calibrate")
        print("Press 's' to show/hide statistics")
        
        self._main_loop()
        
    def _detection_worker(self):
        """Worker thread for card detection"""
        while True:
            if not self.detection_queue.empty():
                frame = self.detection_queue.get()
                detected_cards = self.card_detector.detect_cards(frame)
                self.result_queue.put(detected_cards)
    
    def _main_loop(self):
        """Main application loop"""
        show_stats = True
        last_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Calculate FPS
            current_time = time.time()
            self.frame_times.append(current_time - last_time)
            self.fps = 1 / np.mean(self.frame_times) if self.frame_times else 0
            last_time = current_time
            
            # Send frame to detection thread
            if self.detection_queue.empty():
                self.detection_queue.put(frame.copy())
            
            # Get detection results
            detected_cards = []
            if not self.result_queue.empty():
                detected_cards = self.result_queue.get()
                self._update_game_state(detected_cards)
            
            # Get strategy recommendation
            game_state = GameState(
                player_cards=self.player_cards,
                dealer_cards=self.dealer_cards,
                true_count=self.true_count
            )
            recommendation = self.strategy_engine.get_recommendation(game_state)
            
            # Draw UI overlay
            frame = self.ui_overlay.draw_overlay(
                frame,
                detected_cards,
                self.player_cards,
                self.dealer_cards,
                recommendation,
                self.running_count,
                self.true_count,
                self.fps,
                show_stats
            )
            
            # Display frame
            cv2.imshow('Blackjack Analyzer', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self._reset_game_state()
            elif key == ord('c'):
                self._calibrate()
            elif key == ord('s'):
                show_stats = not show_stats
                
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _update_game_state(self, detected_cards: List[DetectedCard]):
        """Update game state based on detected cards"""
        # Filter high confidence detections
        high_conf_cards = [c for c in detected_cards if c.confidence > config.MEDIUM_CONFIDENCE_THRESHOLD]
        
        # Update card history
        for card in high_conf_cards:
            self.detected_cards_history.append(card)
            
        # Simple heuristic: cards in upper half are dealer's, lower half are player's
        frame_height = config.CAMERA_HEIGHT
        
        new_player_cards = []
        new_dealer_cards = []
        
        for card in high_conf_cards:
            y_center = card.position[1] + card.position[3] // 2
            if y_center < frame_height // 2:
                new_dealer_cards.append((card.rank, card.suit))
            else:
                new_player_cards.append((card.rank, card.suit))
        
        # Update if cards changed
        if new_player_cards != self.player_cards or new_dealer_cards != self.dealer_cards:
            # Update running count for new cards
            all_new_cards = new_player_cards + new_dealer_cards
            all_old_cards = self.player_cards + self.dealer_cards
            
            for card in all_new_cards:
                if card not in all_old_cards:
                    self.running_count += self._get_card_count_value(card[0])
            
            self.player_cards = new_player_cards
            self.dealer_cards = new_dealer_cards
            
            # Update true count
            self.true_count = self.running_count / max(1, self.decks_remaining)
    
    def _get_card_count_value(self, rank: str) -> int:
        """Get Hi-Lo count value for a card"""
        return config.HI_LO_VALUES.get(rank, 0)
    
    def _reset_game_state(self):
        """Reset the game state"""
        self.player_cards = []
        self.dealer_cards = []
        self.running_count = 0
        self.true_count = 0
        print("Game state reset")
    
    def _calibrate(self):
        """Calibrate the card detector"""
        print("Calibration started...")
        self.card_detector.calibrate()
        print("Calibration complete")

def main():
    """Entry point"""
    analyzer = BlackjackAnalyzer()
    analyzer.start()

if __name__ == "__main__":
    main()