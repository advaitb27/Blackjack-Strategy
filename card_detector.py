import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
from dataclasses import dataclass

@dataclass
class DetectedCard:
    """Represents a detected card with its properties"""
    rank: str
    suit: str
    confidence: float
    position: Tuple[int, int, int, int]  # x, y, w, h
    timestamp: float

class CardDetector:
    """Handles card detection and recognition using OpenCV"""
    
    def __init__(self):
        self.ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        self.suits = ['hearts', 'diamonds', 'clubs', 'spades']
        
        # Card detection parameters
        self.min_card_area = 3000
        self.max_card_area = 50000
        self.card_aspect_ratio = 1.4  # Standard playing card ratio
        self.aspect_ratio_tolerance = 0.2
        
        # Image processing parameters
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.gaussian_kernel = (5, 5)
        
        # Template matching (simplified - in real app, load actual templates)
        self.rank_templates = self._create_rank_templates()
        self.suit_templates = self._create_suit_templates()
        
        # Calibration data
        self.calibration_data = {
            'brightness': 1.0,
            'contrast': 1.0,
            'perspective_matrix': None
        }
        
    def detect_cards(self, frame: np.ndarray) -> List[DetectedCard]:
        """Detect and recognize cards in the frame"""
        detected_cards = []
        
        # Preprocess frame
        processed = self._preprocess_frame(frame)
        
        # Find card contours
        card_contours = self._find_card_contours(processed)
        
        # Process each potential card
        for contour in card_contours:
            card = self._process_card_contour(frame, contour)
            if card:
                detected_cards.append(card)
                
        return detected_cards
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better card detection"""
        # Apply calibration adjustments
        adjusted = self._apply_calibration(frame)
        
        # Convert to grayscale
        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, self.gaussian_kernel, 0)
        
        # Apply adaptive threshold for varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def _apply_calibration(self, frame: np.ndarray) -> np.ndarray:
        """Apply calibration settings to frame"""
        # Adjust brightness and contrast
        adjusted = cv2.convertScaleAbs(
            frame,
            alpha=self.calibration_data['contrast'],
            beta=self.calibration_data['brightness'] * 30
        )
        
        # Apply perspective correction if available
        if self.calibration_data['perspective_matrix'] is not None:
            h, w = frame.shape[:2]
            adjusted = cv2.warpPerspective(
                adjusted,
                self.calibration_data['perspective_matrix'],
                (w, h)
            )
            
        return adjusted
    
    def _find_card_contours(self, processed: np.ndarray) -> List[np.ndarray]:
        """Find contours that could be playing cards"""
        # Find edges
        edges = cv2.Canny(processed, self.canny_threshold1, self.canny_threshold2)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        card_contours = []
        
        for contour in contours:
            # Check if contour could be a card
            if self._is_card_contour(contour):
                card_contours.append(contour)
                
        return card_contours
    
    def _is_card_contour(self, contour: np.ndarray) -> bool:
        """Check if contour matches card characteristics"""
        area = cv2.contourArea(contour)
        
        # Check area
        if area < self.min_card_area or area > self.max_card_area:
            return False
        
        # Check if roughly rectangular
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) != 4:
            return False
        
        # Check aspect ratio
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        
        if width == 0 or height == 0:
            return False
            
        aspect_ratio = max(width, height) / min(width, height)
        
        if abs(aspect_ratio - self.card_aspect_ratio) > self.aspect_ratio_tolerance:
            return False
            
        return True
    
    def _process_card_contour(self, frame: np.ndarray, contour: np.ndarray) -> Optional[DetectedCard]:
        """Process a card contour to recognize rank and suit"""
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract card region
        card_img = frame[y:y+h, x:x+w]
        
        # Get perspective-corrected card image
        warped = self._get_card_perspective(card_img, contour)
        
        if warped is None:
            return None
        
        # Recognize rank and suit
        rank, suit, confidence = self._recognize_card(warped)
        
        if confidence < 0.5:  # Low confidence threshold
            return None
            
        import time
        return DetectedCard(
            rank=rank,
            suit=suit,
            confidence=confidence,
            position=(x, y, w, h),
            timestamp=time.time()
        )
    
    def _get_card_perspective(self, card_img: np.ndarray, contour: np.ndarray) -> Optional[np.ndarray]:
        """Get perspective-corrected view of card"""
        # Find corners
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(corners) != 4:
            return None
        
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners.reshape(4, 2))
        
        # Define destination points for standard card size
        card_width = 200
        card_height = 300
        dst_points = np.array([
            [0, 0],
            [card_width, 0],
            [card_width, card_height],
            [0, card_height]
        ], dtype=np.float32)
        
        # Get perspective transform
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Apply transform
        warped = cv2.warpPerspective(card_img, matrix, (card_width, card_height))
        
        return warped
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners in standard format"""
        # Sort by sum (top-left has smallest sum)
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)
        
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = corners[np.argmin(s)]  # Top-left
        ordered[2] = corners[np.argmax(s)]  # Bottom-right
        ordered[1] = corners[np.argmin(diff)]  # Top-right
        ordered[3] = corners[np.argmax(diff)]  # Bottom-left
        
        return ordered
    
    def _recognize_card(self, card_img: np.ndarray) -> Tuple[str, str, float]:
        """Recognize rank and suit from card image"""
        # Convert to grayscale
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        
        # Extract rank and suit regions (top-left corner)
        rank_region = gray[10:60, 10:40]
        suit_region = gray[60:110, 10:40]
        
        # Match against templates
        best_rank, rank_conf = self._match_rank(rank_region)
        best_suit, suit_conf = self._match_suit(suit_region)
        
        # Combined confidence
        confidence = (rank_conf + suit_conf) / 2
        
        return best_rank, best_suit, confidence
    
    def _match_rank(self, rank_region: np.ndarray) -> Tuple[str, float]:
        """Match rank region against templates"""
        best_rank = 'A'
        best_score = 0.0
        
        for rank in self.ranks:
            template = self.rank_templates.get(rank)
            if template is None:
                continue
                
            # Resize template to match region
            template_resized = cv2.resize(template, rank_region.shape[::-1])
            
            # Template matching
            result = cv2.matchTemplate(rank_region, template_resized, cv2.TM_CCOEFF_NORMED)
            score = np.max(result)
            
            if score > best_score:
                best_score = score
                best_rank = rank
                
        return best_rank, best_score
    
    def _match_suit(self, suit_region: np.ndarray) -> Tuple[str, float]:
        """Match suit region against templates"""
        best_suit = 'hearts'
        best_score = 0.0
        
        for suit in self.suits:
            template = self.suit_templates.get(suit)
            if template is None:
                continue
                
            # Resize template to match region
            template_resized = cv2.resize(template, suit_region.shape[::-1])
            
            # Template matching
            result = cv2.matchTemplate(suit_region, template_resized, cv2.TM_CCOEFF_NORMED)
            score = np.max(result)
            
            if score > best_score:
                best_score = score
                best_suit = suit
                
        return best_suit, best_score
    
    def _create_rank_templates(self) -> dict:
        """Create simplified rank templates (in real app, load from files)"""
        templates = {}
        
        # Create simple templates for demonstration
        for rank in self.ranks:
            template = np.ones((50, 30), dtype=np.uint8) * 255
            cv2.putText(template, rank, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
            templates[rank] = template
            
        return templates
    
    def _create_suit_templates(self) -> dict:
        """Create simplified suit templates (in real app, load from files)"""
        templates = {}
        
        # Create simple templates for demonstration
        for suit in self.suits:
            template = np.ones((50, 30), dtype=np.uint8) * 255
            
            # Draw simple suit symbols
            if suit == 'hearts':
                cv2.circle(template, (15, 20), 8, 0, -1)
                cv2.circle(template, (25, 20), 8, 0, -1)
                pts = np.array([[20, 30], [10, 20], [30, 20]], np.int32)
                cv2.fillPoly(template, [pts], 0)
            elif suit == 'diamonds':
                pts = np.array([[20, 10], [30, 25], [20, 40], [10, 25]], np.int32)
                cv2.fillPoly(template, [pts], 0)
            elif suit == 'clubs':
                cv2.circle(template, (20, 20), 8, 0, -1)
                cv2.circle(template, (15, 30), 6, 0, -1)
                cv2.circle(template, (25, 30), 6, 0, -1)
            elif suit == 'spades':
                pts = np.array([[20, 10], [10, 25], [30, 25]], np.int32)
                cv2.fillPoly(template, [pts], 0)
                cv2.circle(template, (15, 25), 6, 0, -1)
                cv2.circle(template, (25, 25), 6, 0, -1)
                
            templates[suit] = template
            
        return templates
    
    def calibrate(self):
        """Calibrate detector for current lighting conditions"""
        print("Starting calibration...")
        # In a real implementation, this would:
        # 1. Capture multiple frames
        # 2. Detect a calibration pattern
        # 3. Adjust parameters based on conditions
        # 4. Save calibration data
        
        # Simplified calibration
        self.calibration_data['brightness'] = 1.1
        self.calibration_data['contrast'] = 1.05
        print("Calibration complete!")