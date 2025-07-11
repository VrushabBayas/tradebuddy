"""
Risk management utilities for leveraged crypto trading.

Provides functions for calculating position sizes, stop losses, and take profits
optimized for Delta Exchange 10x leverage trading.
"""

from decimal import Decimal
from typing import Tuple, Optional, List
import structlog

from src.core.models import SessionConfig, SignalAction

logger = structlog.get_logger(__name__)


def calculate_leveraged_position_size(
    account_balance: float,
    position_size_pct: float,
    leverage: int,
    current_price: float,
    min_lot_size: float = 0.001
) -> Tuple[float, float, float]:
    """
    Calculate position size for leveraged trading.
    
    Args:
        account_balance: Total account balance in USD
        position_size_pct: Position size as percentage of account
        leverage: Leverage multiplier (e.g., 10 for 10x)
        current_price: Current asset price
        min_lot_size: Minimum lot size in base asset
        
    Returns:
        Tuple of (position_value_usd, position_amount_btc, margin_required)
    """
    # Calculate position value
    margin_required = account_balance * (position_size_pct / 100)
    position_value_usd = margin_required * leverage
    position_amount_btc = position_value_usd / current_price
    
    # Adjust to minimum lot size
    lots_needed = position_amount_btc / min_lot_size
    adjusted_lots = max(1, round(lots_needed, 3))  # Round to nearest 0.001
    adjusted_position_btc = adjusted_lots * min_lot_size
    adjusted_position_usd = adjusted_position_btc * current_price
    adjusted_margin = adjusted_position_usd / leverage
    
    logger.debug(
        "Position size calculated",
        account_balance=account_balance,
        position_size_pct=position_size_pct,
        leverage=leverage,
        margin_required=adjusted_margin,
        position_value_usd=adjusted_position_usd,
        position_amount_btc=adjusted_position_btc,
        lots=adjusted_lots
    )
    
    return adjusted_position_usd, adjusted_position_btc, adjusted_margin


def calculate_stop_loss_take_profit(
    entry_price: float,
    signal_action: SignalAction,
    stop_loss_pct: float,
    take_profit_pct: float,
    leverage: int = 10
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate stop loss and take profit levels for leveraged trading.
    
    Args:
        entry_price: Entry price for the trade
        signal_action: BUY or SELL signal
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        leverage: Leverage multiplier
        
    Returns:
        Tuple of (stop_loss_price, take_profit_price, risk_reward_ratio)
    """
    if signal_action not in [SignalAction.BUY, SignalAction.SELL]:
        return None, None, None
    
    # Calculate stop loss and take profit
    stop_loss_amount = entry_price * (stop_loss_pct / 100)
    take_profit_amount = entry_price * (take_profit_pct / 100)
    
    if signal_action == SignalAction.BUY:
        stop_loss_price = entry_price - stop_loss_amount
        take_profit_price = entry_price + take_profit_amount
    else:  # SELL
        stop_loss_price = entry_price + stop_loss_amount
        take_profit_price = entry_price - take_profit_amount
    
    # Calculate risk-reward ratio
    risk = abs(entry_price - stop_loss_price)
    reward = abs(take_profit_price - entry_price)
    risk_reward_ratio = reward / risk if risk > 0 else None
    
    logger.debug(
        "Stop loss and take profit calculated",
        entry_price=entry_price,
        signal_action=signal_action.value,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        risk_reward_ratio=risk_reward_ratio,
        leverage=leverage
    )
    
    return stop_loss_price, take_profit_price, risk_reward_ratio


def calculate_position_risk(
    position_value_usd: float,
    account_balance: float,
    leverage: int,
    stop_loss_pct: float
) -> Tuple[float, float, float]:
    """
    Calculate position risk metrics.
    
    Args:
        position_value_usd: Total position value in USD
        account_balance: Account balance
        leverage: Leverage multiplier
        stop_loss_pct: Stop loss percentage
        
    Returns:
        Tuple of (max_loss_usd, max_loss_pct, effective_leverage)
    """
    # Calculate maximum loss
    max_loss_usd = position_value_usd * (stop_loss_pct / 100)
    max_loss_pct = (max_loss_usd / account_balance) * 100
    
    # Calculate effective leverage (position value / account balance)
    effective_leverage = position_value_usd / account_balance
    
    logger.debug(
        "Position risk calculated",
        position_value_usd=position_value_usd,
        account_balance=account_balance,
        leverage=leverage,
        max_loss_usd=max_loss_usd,
        max_loss_pct=max_loss_pct,
        effective_leverage=effective_leverage
    )
    
    return max_loss_usd, max_loss_pct, effective_leverage


def optimize_position_for_delta_exchange(
    session_config: SessionConfig,
    current_price: float,
    account_balance: float = 10000.0
) -> dict:
    """
    Optimize position parameters for Delta Exchange trading.
    
    Args:
        session_config: Session configuration
        current_price: Current asset price
        account_balance: Account balance in USD
        
    Returns:
        Dictionary with optimized position parameters
    """
    # Calculate position size
    position_value_usd, position_amount_btc, margin_required = calculate_leveraged_position_size(
        account_balance=account_balance,
        position_size_pct=float(session_config.position_size_pct),
        leverage=session_config.leverage,
        current_price=current_price,
        min_lot_size=float(session_config.min_lot_size)
    )
    
    # Calculate risk metrics
    max_loss_usd, max_loss_pct, effective_leverage = calculate_position_risk(
        position_value_usd=position_value_usd,
        account_balance=account_balance,
        leverage=session_config.leverage,
        stop_loss_pct=float(session_config.stop_loss_pct)
    )
    
    # Calculate lot size
    lots = position_amount_btc / float(session_config.min_lot_size)
    
    return {
        "position_value_usd": position_value_usd,
        "position_amount_btc": position_amount_btc,
        "margin_required": margin_required,
        "lots": lots,
        "max_loss_usd": max_loss_usd,
        "max_loss_pct": max_loss_pct,
        "effective_leverage": effective_leverage,
        "risk_reward_ratio": float(session_config.take_profit_pct) / float(session_config.stop_loss_pct)
    }


def validate_position_safety(
    session_config: SessionConfig,
    position_params: dict,
    account_balance: float = 10000.0
) -> Tuple[bool, List[str]]:
    """
    Validate position safety parameters.
    
    Args:
        session_config: Session configuration
        position_params: Position parameters from optimize_position_for_delta_exchange
        account_balance: Account balance
        
    Returns:
        Tuple of (is_safe, warnings)
    """
    warnings = []
    is_safe = True
    
    # Check maximum loss percentage
    max_loss_pct = position_params["max_loss_pct"]
    if max_loss_pct > float(session_config.max_position_risk):
        warnings.append(f"Position risk {max_loss_pct:.1f}% exceeds maximum {session_config.max_position_risk}%")
        is_safe = False
    
    # Check effective leverage
    effective_leverage = position_params["effective_leverage"]
    if effective_leverage > session_config.leverage * 1.5:
        warnings.append(f"Effective leverage {effective_leverage:.1f}x is too high")
        is_safe = False
    
    # Check minimum lot size
    lots = position_params["lots"]
    if lots < 1:
        warnings.append(f"Position size {lots:.3f} lots is below minimum (1 lot)")
        is_safe = False
    
    # Check margin requirement
    margin_pct = (position_params["margin_required"] / account_balance) * 100
    if margin_pct > float(session_config.position_size_pct) * 1.2:
        warnings.append(f"Margin requirement {margin_pct:.1f}% exceeds expected {session_config.position_size_pct}%")
    
    return is_safe, warnings