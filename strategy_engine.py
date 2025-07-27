from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class GameState:
    """Current state of the blackjack game"""
    player_cards: List[Tuple[str, str]]  # List of (rank, suit) tuples
    dealer_cards: List[Tuple[str, str]]
    true_count: float
    
@dataclass
class StrategyRecommendation:
    """Strategy recommendation with confidence and reasoning"""
    action: str  # 'hit', 'stand', 'double', 'split', 'surrender'
    confidence: float
    reasoning: str
    bet_recommendation: str
    expected_value: float

class StrategyEngine:
    """Blackjack strategy engine with basic strategy and card counting"""
    
    def __init__(self):
        # Card values
        self.card_values = {
            'A': 11, '2': 2, '3': 3, '4': 4, '5': 5,
            '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
            'J': 10, 'Q': 10, 'K': 10
        }
        
        # Basic strategy tables
        self.hard_totals_strategy = self._init_hard_totals_strategy()
        self.soft_totals_strategy = self._init_soft_totals_strategy()
        self.pair_strategy = self._init_pair_strategy()
        
        # Betting strategy parameters
        self.min_bet_units = 1
        self.max_bet_units = 10
        self.betting_spread = {
            -2: 1,   # True count -2 or lower
            -1: 1,   # True count -1
            0: 1,    # True count 0
            1: 2,    # True count +1
            2: 4,    # True count +2
            3: 6,    # True count +3
            4: 8,    # True count +4
            5: 10    # True count +5 or higher
        }
        
    def get_recommendation(self, game_state: GameState) -> StrategyRecommendation:
        """Get strategy recommendation based on current game state"""
        if not game_state.player_cards:
            return StrategyRecommendation(
                action="wait",
                confidence=1.0,
                reasoning="Waiting for cards to be dealt",
                bet_recommendation=self._get_bet_recommendation(game_state.true_count),
                expected_value=0.0
            )
        
        # Calculate hand values
        player_value, is_soft = self._calculate_hand_value(game_state.player_cards)
        dealer_up_card = game_state.dealer_cards[0][0] if game_state.dealer_cards else None
        
        # Check for blackjack
        if player_value == 21 and len(game_state.player_cards) == 2:
            return StrategyRecommendation(
                action="blackjack",
                confidence=1.0,
                reasoning="Natural blackjack!",
                bet_recommendation=self._get_bet_recommendation(game_state.true_count),
                expected_value=1.5
            )
        
        # Check for bust
        if player_value > 21:
            return StrategyRecommendation(
                action="bust",
                confidence=1.0,
                reasoning=f"Bust with {player_value}",
                bet_recommendation=self._get_bet_recommendation(game_state.true_count),
                expected_value=-1.0
            )
        
        # Get basic strategy recommendation
        if dealer_up_card:
            action, confidence, reasoning = self._get_basic_strategy(
                game_state.player_cards, dealer_up_card, is_soft
            )
            
            # Adjust for count
            action, reasoning = self._adjust_for_count(
                action, game_state.true_count, player_value, dealer_up_card, reasoning
            )
            
            # Calculate expected value
            ev = self._calculate_expected_value(player_value, dealer_up_card, action, game_state.true_count)
            
            return StrategyRecommendation(
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                bet_recommendation=self._get_bet_recommendation(game_state.true_count),
                expected_value=ev
            )
        
        return StrategyRecommendation(
            action="wait",
            confidence=0.5,
            reasoning="Waiting for dealer's up card",
            bet_recommendation=self._get_bet_recommendation(game_state.true_count),
            expected_value=0.0
        )
    
    def _calculate_hand_value(self, cards: List[Tuple[str, str]]) -> Tuple[int, bool]:
        """Calculate the value of a hand, returning (value, is_soft)"""
        value = 0
        aces = 0
        
        for rank, _ in cards:
            card_value = self.card_values[rank]
            value += card_value
            if rank == 'A':
                aces += 1
        
        # Adjust for aces
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
        
        is_soft = aces > 0 and value <= 21
        return value, is_soft
    
    def _get_basic_strategy(self, player_cards: List[Tuple[str, str]], 
                          dealer_up: str, is_soft: bool) -> Tuple[str, float, str]:
        """Get basic strategy recommendation"""
        # Check for pairs
        if len(player_cards) == 2 and player_cards[0][0] == player_cards[1][0]:
            return self._get_pair_strategy(player_cards[0][0], dealer_up)
        
        player_value, _ = self._calculate_hand_value(player_cards)
        
        # Use appropriate strategy table
        if is_soft:
            return self._get_soft_strategy(player_value, dealer_up)
        else:
            return self._get_hard_strategy(player_value, dealer_up)
    
    def _get_hard_strategy(self, player_value: int, dealer_up: str) -> Tuple[str, float, str]:
        """Get strategy for hard totals"""
        dealer_value = self.card_values[dealer_up]
        
        # Simplified basic strategy
        if player_value >= 17:
            return "stand", 0.95, f"Stand on hard {player_value}"
        elif player_value >= 13 and dealer_value <= 6:
            return "stand", 0.90, f"Stand on {player_value} vs dealer {dealer_up}"
        elif player_value == 12 and dealer_value in [4, 5, 6]:
            return "stand", 0.85, f"Stand on 12 vs dealer {dealer_up}"
        elif player_value == 11:
            return "double", 0.95, "Double on 11"
        elif player_value == 10 and dealer_value <= 9:
            return "double", 0.90, f"Double on 10 vs dealer {dealer_up}"
        elif player_value == 9 and dealer_value in [3, 4, 5, 6]:
            return "double", 0.85, f"Double on 9 vs dealer {dealer_up}"
        else:
            return "hit", 0.90, f"Hit {player_value} vs dealer {dealer_up}"
    
    def _get_soft_strategy(self, player_value: int, dealer_up: str) -> Tuple[str, float, str]:
        """Get strategy for soft totals"""
        dealer_value = self.card_values[dealer_up]
        
        # Simplified soft strategy
        if player_value >= 19:
            return "stand", 0.95, f"Stand on soft {player_value}"
        elif player_value == 18:
            if dealer_value in [9, 10] or dealer_up in ['J', 'Q', 'K', 'A']:
                return "hit", 0.85, f"Hit soft 18 vs dealer {dealer_up}"
            elif dealer_value in [3, 4, 5, 6]:
                return "double", 0.90, f"Double soft 18 vs dealer {dealer_up}"
            else:
                return "stand", 0.90, f"Stand on soft 18 vs dealer {dealer_up}"
        elif player_value in [13, 14, 15, 16, 17]:
            if dealer_value in [5, 6]:
                return "double", 0.85, f"Double soft {player_value} vs dealer {dealer_up}"
            else:
                return "hit", 0.85, f"Hit soft {player_value}"
        else:
            return "hit", 0.90, f"Hit soft {player_value}"
    
    def _get_pair_strategy(self, pair_rank: str, dealer_up: str) -> Tuple[str, float, str]:
        """Get strategy for pairs"""
        dealer_value = self.card_values[dealer_up]
        
        # Simplified pair strategy
        if pair_rank == 'A':
            return "split", 0.95, "Always split aces"
        elif pair_rank == '8':
            return "split", 0.95, "Always split 8s"
        elif pair_rank in ['2', '3', '7'] and dealer_value <= 7:
            return "split", 0.85, f"Split {pair_rank}s vs dealer {dealer_up}"
        elif pair_rank == '6' and dealer_value <= 6:
            return "split", 0.85, f"Split 6s vs dealer {dealer_up}"
        elif pair_rank == '9' and dealer_value not in [7, 10] and dealer_up not in ['J', 'Q', 'K', 'A']:
            return "split", 0.85, f"Split 9s vs dealer {dealer_up}"
        elif pair_rank in ['10', 'J', 'Q', 'K']:
            return "stand", 0.95, "Never split 10s"
        elif pair_rank == '5':
            return "double", 0.90, "Never split 5s, double instead"
        elif pair_rank == '4' and dealer_value in [5, 6]:
            return "split", 0.80, f"Split 4s vs dealer {dealer_up}"
        else:
            # Use hard total strategy
            total = self.card_values[pair_rank] * 2
            return self._get_hard_strategy(total, dealer_up)
    
    def _adjust_for_count(self, action: str, true_count: float, player_value: int, 
                         dealer_up: str, reasoning: str) -> Tuple[str, str]:
        """Adjust strategy based on true count"""
        # Index plays based on true count
        new_action = action
        new_reasoning = reasoning
        
        # Insurance
        if dealer_up == 'A' and true_count >= 3:
            new_reasoning += f" | Insurance recommended (TC: {true_count:.1f})"
        
        # 16 vs 10
        if player_value == 16 and dealer_up in ['10', 'J', 'Q', 'K'] and true_count >= 0:
            if action == "hit":
                new_action = "stand"
                new_reasoning = f"Stand 16 vs 10 (TC: {true_count:.1f})"
        
        # 15 vs 10
        if player_value == 15 and dealer_up in ['10', 'J', 'Q', 'K'] and true_count >= 4:
            if action == "hit":
                new_action = "stand"
                new_reasoning = f"Stand 15 vs 10 (TC: {true_count:.1f})"
        
        # 12 vs 3
        if player_value == 12 and dealer_up == '3' and true_count >= 2:
            if action == "hit":
                new_action = "stand"
                new_reasoning = f"Stand 12 vs 3 (TC: {true_count:.1f})"
        
        # 12 vs 2
        if player_value == 12 and dealer_up == '2' and true_count >= 3:
            if action == "hit":
                new_action = "stand"
                new_reasoning = f"Stand 12 vs 2 (TC: {true_count:.1f})"
        
        # 11 vs A
        if player_value == 11 and dealer_up == 'A' and true_count >= 1:
            if action == "hit":
                new_action = "double"
                new_reasoning = f"Double 11 vs A (TC: {true_count:.1f})"
        
        # 9 vs 2
        if player_value == 9 and dealer_up == '2' and true_count >= 1:
            if action == "hit":
                new_action = "double"
                new_reasoning = f"Double 9 vs 2 (TC: {true_count:.1f})"
        
        # 10 vs 10
        if player_value == 10 and dealer_up in ['10', 'J', 'Q', 'K'] and true_count >= 4:
            if action == "hit":
                new_action = "double"
                new_reasoning = f"Double 10 vs 10 (TC: {true_count:.1f})"
        
        # 10 vs A
        if player_value == 10 and dealer_up == 'A' and true_count >= 3:
            if action == "hit":
                new_action = "double"
                new_reasoning = f"Double 10 vs A (TC: {true_count:.1f})"
        
        return new_action, new_reasoning
    
    def _get_bet_recommendation(self, true_count: float) -> str:
        """Get betting recommendation based on true count"""
        # Find appropriate bet units
        bet_units = self.min_bet_units
        
        for tc_threshold in sorted(self.betting_spread.keys(), reverse=True):
            if true_count >= tc_threshold:
                bet_units = self.betting_spread[tc_threshold]
                break
        
        if true_count < -1:
            return f"Minimum bet ({bet_units} unit) - Unfavorable count"
        elif true_count >= 2:
            return f"Bet {bet_units} units - Favorable count!"
        else:
            return f"Bet {bet_units} unit(s) - Neutral count"
    
    def _calculate_expected_value(self, player_value: int, dealer_up: str, 
                                action: str, true_count: float) -> float:
        """Calculate approximate expected value of the hand"""
        # Simplified EV calculation
        base_ev = 0.0
        
        # Adjust for player hand strength
        if player_value == 21:
            base_ev = 0.9
        elif player_value == 20:
            base_ev = 0.7
        elif player_value == 19:
            base_ev = 0.5
        elif player_value == 18:
            base_ev = 0.2
        elif player_value == 17:
            base_ev = -0.1
        elif player_value <= 16:
            base_ev = -0.3
        
        # Adjust for dealer up card
        dealer_value = self.card_values[dealer_up]
        if dealer_value >= 7:
            base_ev -= 0.2
        elif dealer_value <= 6:
            base_ev += 0.2
        
        # Adjust for count
        count_adjustment = true_count * 0.005
        
        # Adjust for action
        if action == "double":
            base_ev *= 2
        elif action == "split":
            base_ev *= 0.9
        
        return max(-1.0, min(1.0, base_ev + count_adjustment))
    
    def _init_hard_totals_strategy(self) -> dict:
        """Initialize hard totals strategy table"""
        # Simplified - in real app would be complete basic strategy
        return {}
    
    def _init_soft_totals_strategy(self) -> dict:
        """Initialize soft totals strategy table"""
        # Simplified - in real app would be complete basic strategy
        return {}
    
    def _init_pair_strategy(self) -> dict:
        """Initialize pair splitting strategy table"""
        # Simplified - in real app would be complete basic strategy
        return {}