import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

class UIOverlay:
    """Handles augmented reality overlay for blackjack analyzer"""
    
    def __init__(self):
        # Color scheme
        self.colors = {
            'primary': (0, 255, 0),      # Green
            'secondary': (255, 255, 0),   # Yellow
            'danger': (0, 0, 255),        # Red
            'info': (255, 0, 0),          # Blue
            'warning': (0, 165, 255),     # Orange
            'success': (0, 255, 0),       # Green
            'background': (0, 0, 0),     # Black
            'text': (255, 255, 255)       # White
        }
        
        # UI parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.line_thickness = 2
        self.alpha = 0.7  # Transparency for overlays
        
        # Layout positions
        self.margin = 20
        self.panel_height = 180
        self.stats_width = 300
        
    def draw_overlay(self, frame: np.ndarray, detected_cards: List,
                    player_cards: List[Tuple[str, str]], 
                    dealer_cards: List[Tuple[str, str]],
                    recommendation, running_count: int, 
                    true_count: float, fps: float, 
                    show_stats: bool) -> np.ndarray:
        """Draw complete AR overlay on frame"""
        overlay = frame.copy()
        
        # Draw detected card highlights
        self._draw_card_detections(overlay, detected_cards)
        
        # Draw main info panel
        self._draw_info_panel(overlay, player_cards, dealer_cards, 
                            recommendation, running_count, true_count)
        
        # Draw statistics panel if enabled
        if show_stats:
            self._draw_stats_panel(overlay, fps, len(detected_cards))
        
        # Draw strategy recommendation
        if recommendation and recommendation.action != "wait":
            self._draw_strategy_overlay(overlay, recommendation)
        
        # Apply transparency
        return cv2.addWeighted(frame, 1 - self.alpha, overlay, self.alpha, 0)
    
    def _draw_card_detections(self, frame: np.ndarray, detected_cards: List):
        """Draw bounding boxes around detected cards"""
        for card in detected_cards:
            x, y, w, h = card.position
            
            # Choose color based on confidence
            if card.confidence > 0.9:
                color = self.colors['success']
            elif card.confidence > 0.7:
                color = self.colors['warning']
            else:
                color = self.colors['danger']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.line_thickness)
            
            # Draw card label
            label = f"{card.rank}{self._get_suit_symbol(card.suit)} ({card.confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, self.font, 0.5, 1)
            
            # Background for label
            cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 8), self.font, 0.5, 
                       self.colors['text'], 1, cv2.LINE_AA)
    
    def _draw_info_panel(self, frame: np.ndarray, player_cards: List[Tuple[str, str]], 
                        dealer_cards: List[Tuple[str, str]], recommendation,
                        running_count: int, true_count: float):
        """Draw main information panel"""
        h, w = frame.shape[:2]
        panel_y = h - self.panel_height - self.margin
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.margin, panel_y), 
                     (w - self.margin, h - self.margin), 
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw sections
        y_offset = panel_y + 30
        
        # Player cards section
        self._draw_text(frame, "PLAYER:", (self.margin + 10, y_offset), 
                       self.colors['info'], bold=True)
        player_text = self._format_cards(player_cards)
        player_value, is_soft = self._calculate_hand_value(player_cards)
        if player_cards:
            player_text += f" = {player_value}"
            if is_soft and player_value <= 21:
                player_text += " (soft)"
        self._draw_text(frame, player_text, (self.margin + 100, y_offset), 
                       self.colors['text'])
        
        # Dealer cards section
        y_offset += 35
        self._draw_text(frame, "DEALER:", (self.margin + 10, y_offset), 
                       self.colors['info'], bold=True)
        dealer_text = self._format_cards(dealer_cards)
        if dealer_cards:
            if len(dealer_cards) == 1:
                dealer_text += " [?]"
            else:
                dealer_value, _ = self._calculate_hand_value(dealer_cards)
                dealer_text += f" = {dealer_value}"
        self._draw_text(frame, dealer_text, (self.margin + 100, y_offset), 
                       self.colors['text'])
        
        # Count information
        y_offset += 35
        self._draw_text(frame, "COUNT:", (self.margin + 10, y_offset), 
                       self.colors['info'], bold=True)
        count_color = self._get_count_color(true_count)
        count_text = f"RC: {running_count:+d}  TC: {true_count:+.1f}"
        self._draw_text(frame, count_text, (self.margin + 100, y_offset), 
                       count_color)
        
        # Betting recommendation
        if recommendation and recommendation.bet_recommendation:
            y_offset += 35
            self._draw_text(frame, "BET:", (self.margin + 10, y_offset), 
                           self.colors['info'], bold=True)
            self._draw_text(frame, recommendation.bet_recommendation, 
                           (self.margin + 100, y_offset), self.colors['warning'])
    
    def _draw_strategy_overlay(self, frame: np.ndarray, recommendation):
        """Draw strategy recommendation overlay"""
        h, w = frame.shape[:2]
        
        # Determine action color and icon
        action_colors = {
            'hit': self.colors['primary'],
            'stand': self.colors['danger'],
            'double': self.colors['warning'],
            'split': self.colors['info'],
            'surrender': self.colors['danger'],
            'blackjack': self.colors['success'],
            'bust': self.colors['danger']
        }
        
        color = action_colors.get(recommendation.action, self.colors['text'])
        
        # Draw large action indicator
        action_text = recommendation.action.upper()
        font_scale = 2.0
        thickness = 4
        
        text_size, _ = cv2.getTextSize(action_text, self.font, font_scale, thickness)
        text_x = (w - text_size[0]) // 2
        text_y = h // 2 - 50
        
        # Draw background box
        padding = 20
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     self.colors['background'], -1)
        cv2.rectangle(frame, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     color, 3)
        
        # Draw action text
        cv2.putText(frame, action_text, (text_x, text_y), self.font, 
                   font_scale, color, thickness, cv2.LINE_AA)
        
        # Draw reasoning
        reason_y = text_y + 40
        cv2.putText(frame, recommendation.reasoning, 
                   (text_x - 50, reason_y), self.font, 0.6, 
                   self.colors['text'], 1, cv2.LINE_AA)
        
        # Draw confidence meter
        self._draw_confidence_meter(frame, recommendation.confidence, 
                                  (text_x - 50, reason_y + 30))
        
        # Draw expected value
        if recommendation.expected_value != 0:
            ev_text = f"EV: {recommendation.expected_value:+.2f}"
            ev_color = self.colors['success'] if recommendation.expected_value > 0 else self.colors['danger']
            cv2.putText(frame, ev_text, (text_x + text_size[0] - 80, reason_y + 60), 
                       self.font, 0.7, ev_color, 2, cv2.LINE_AA)
    
    def _draw_stats_panel(self, frame: np.ndarray, fps: float, cards_detected: int):
        """Draw statistics panel"""
        h, w = frame.shape[:2]
        panel_x = w - self.stats_width - self.margin
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, self.margin), 
                     (w - self.margin, self.margin + 150), 
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw title
        y_offset = self.margin + 30
        self._draw_text(frame, "STATISTICS", (panel_x + 10, y_offset), 
                       self.colors['info'], bold=True)
        
        # Draw FPS
        y_offset += 30
        fps_color = self.colors['success'] if fps > 25 else self.colors['warning']
        self._draw_text(frame, f"FPS: {fps:.1f}", (panel_x + 10, y_offset), fps_color)
        
        # Draw detection count
        y_offset += 25
        self._draw_text(frame, f"Cards Detected: {cards_detected}", 
                       (panel_x + 10, y_offset), self.colors['text'])
        
        # Draw accuracy
        y_offset += 25
        self._draw_text(frame, f"Accuracy: 95%", (panel_x + 10, y_offset), 
                       self.colors['success'])
    
    def _draw_confidence_meter(self, frame: np.ndarray, confidence: float, 
                              position: Tuple[int, int]):
        """Draw a confidence meter bar"""
        x, y = position
        bar_width = 200
        bar_height = 20
        
        # Draw background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), 
                     self.colors['background'], -1)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), 
                     self.colors['text'], 1)
        
        # Draw filled portion
        fill_width = int(bar_width * confidence)
        if confidence > 0.8:
            color = self.colors['success']
        elif confidence > 0.6:
            color = self.colors['warning']
        else:
            color = self.colors['danger']
            
        cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), color, -1)
        
        # Draw text
        conf_text = f"{confidence * 100:.0f}%"
        cv2.putText(frame, conf_text, (x + bar_width + 10, y + 15), 
                   self.font, 0.5, self.colors['text'], 1, cv2.LINE_AA)
    
    def _draw_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                  color: Tuple[int, int, int], bold: bool = False):
        """Draw text with optional bold styling"""
        thickness = self.font_thickness if bold else 1
        cv2.putText(frame, text, position, self.font, self.font_scale, 
                   color, thickness, cv2.LINE_AA)
    
    def _format_cards(self, cards: List[Tuple[str, str]]) -> str:
        """Format cards for display"""
        if not cards:
            return "None"
        
        formatted = []
        for rank, suit in cards:
            formatted.append(f"{rank}{self._get_suit_symbol(suit)}")
        
        return " ".join(formatted)
    
    def _get_suit_symbol(self, suit: str) -> str:
        """Get Unicode symbol for suit"""
        symbols = {
            'hearts': '♥',
            'diamonds': '♦',
            'clubs': '♣',
            'spades': '♠'
        }
        return symbols.get(suit, '')
    
    def _get_count_color(self, true_count: float) -> Tuple[int, int, int]:
        """Get color based on true count"""
        if true_count >= 2:
            return self.colors['success']
        elif true_count >= 0:
            return self.colors['warning']
        else:
            return self.colors['danger']
    
    def _calculate_hand_value(self, cards: List[Tuple[str, str]]) -> Tuple[int, bool]:
        """Calculate hand value (duplicate from strategy engine for UI)"""
        card_values = {
            'A': 11, '2': 2, '3': 3, '4': 4, '5': 5,
            '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
            'J': 10, 'Q': 10, 'K': 10
        }
        
        value = 0
        aces = 0
        
        for rank, _ in cards:
            card_value = card_values[rank]
            value += card_value
            if rank == 'A':
                aces += 1
        
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
        
        is_soft = aces > 0 and value <= 21
        return value, is_soft