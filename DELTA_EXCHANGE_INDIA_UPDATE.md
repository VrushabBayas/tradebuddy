# Delta Exchange India Integration Update

## Overview

TradeBuddy has been updated to support **Delta Exchange India** with proper INR currency handling and Indian regulatory compliance. The system now uses the correct fee structure and displays all financial data in Indian Rupees.

## Key Changes

### 1. 🇮🇳 Currency System

- **Base Currency**: Changed from USD to INR
- **Exchange Rate**: 1 USD = ₹85 (configurable)
- **Currency Symbol**: ₹ (Indian Rupee symbol)
- **Number Formatting**: Indian style with Lakh/Crore notation

### 2. 💰 Fee Structure (Delta Exchange India)

| Fee Type | Previous (Generic) | Updated (Delta India) |
|----------|-------------------|----------------------|
| **Futures Taker** | 0.1% | **0.05%** |
| **Futures Maker** | 0.1% | **0.02%** |
| **Options Taker** | 0.1% | **0.03%** |
| **Options Maker** | 0.1% | **0.03%** |
| **GST on Fees** | Not included | **18%** |
| **Option Fee Cap** | Not implemented | **10% of premium** |

### 3. 🔧 Technical Updates

#### Fee Calculation
- **Before**: Commission calculated on position size
- **After**: Commission calculated on leveraged position value + 18% GST

#### P&L Calculation (Fixed)
- **Before**: `pnl_usd = (price_change / entry_price) * position_size_usd`
- **After**: `pnl_usd = (price_change / entry_price) * position_size_usd * leverage`

#### Currency Display
- **Before**: All values in USD ($)
- **After**: All values in INR (₹) with Lakh/Crore formatting

### 4. 📊 Report Generation

Reports now display:
- **₹8,50,000** instead of $10,000
- **₹1.27 L** for 1.27 Lakhs  
- **₹12.75 Cr** for 12.75 Crores
- GST breakdown in trade details
- Delta Exchange India fee structure

### 5. 🏛️ Default Configuration

| Parameter | Previous | Updated |
|-----------|----------|---------|
| **Initial Capital** | $10,000 | **₹8,50,000** |
| **Commission** | 0.1% | **0.05%** |
| **Default Leverage** | 10x | **5x** |
| **Currency** | USD | **INR** |

## Code Changes Summary

### Files Modified

1. **`src/core/constants.py`**
   - Added Delta Exchange India fee structure
   - Added currency constants (USD_TO_INR = 85.0)
   - Updated default commission to 0.05%

2. **`src/backtesting/models.py`**
   - Fixed P&L calculation with proper leverage
   - Added INR P&L field (`pnl_inr`)
   - Added GST tracking (`gst_paid`)
   - Updated default capital to ₹8,50,000

3. **`src/backtesting/portfolio.py`**
   - Added GST calculation (18% on fees)
   - Fixed commission calculation on leveraged positions
   - Updated cost tracking

4. **`src/backtesting/reports.py`**
   - Added INR currency support
   - Implemented Indian number formatting (Lakh/Crore)
   - Updated chart labels and hover text

5. **`src/cli/main.py`**
   - Updated UI to show INR values
   - Changed default capital prompt to INR
   - Updated report generation to use INR

6. **`src/core/config.py`**
   - Added currency configuration
   - Updated default values for Indian market

## Usage Examples

### Backtesting with INR

```python
config = BacktestConfig(
    strategy_type=StrategyType.EMA_CROSSOVER,
    symbol=Symbol.BTCUSDT,
    timeframe=TimeFrame.ONE_HOUR,
    initial_capital=850000.0,  # ₹8.5 Lakhs
    leverage=5,
    commission_pct=0.05,  # Delta Exchange futures taker
    currency="INR"
)
```

### Trade P&L Example

```
Entry: ₹42,50,000 (BTC price)
Exit:  ₹43,35,000 (2% increase)
Position: $10,000 with 5x leverage

Results:
- Raw Return: 2%
- Leveraged Return: 10% (2% × 5x)
- P&L: $1,000 (₹85,000)
- Commission: ₹2,150 (0.05% on ₹42.5L)
- GST: ₹387 (18% on commission)
- Net P&L: ₹82,463
```

### Currency Formatting

| USD Amount | INR Display |
|------------|-------------|
| $1,000 | ₹85,000 |
| $15,000 | ₹1.28 L |
| $150,000 | ₹1.28 Cr |
| $1,500,000 | ₹12.75 Cr |

## Verification

Run the test script to verify all changes:

```bash
python test_inr_update.py
```

Expected output:
```
🇮🇳 Testing Delta Exchange India Integration
✅ Base Currency: INR
✅ Futures Taker Fee: 0.05%
✅ GST Rate: 18.0%
✅ P&L calculations include leverage and GST
✅ Reports formatted in Indian currency format
```

## Compliance Notes

### Indian Regulations
- **GST**: 18% applied to all trading fees as per Indian tax law
- **Fee Structure**: Matches Delta Exchange India's published rates
- **Currency**: All reporting in INR for Indian traders
- **Leverage**: Conservative default of 5x (vs 10x previously)

### Delta Exchange India Features
- **Legal Status**: Registered with FIU, Government of India
- **Settlement**: INR settlements for Indian traders
- **Lot Sizes**: Small lot sizes to facilitate entry
- **Fee Capping**: Options fees capped at 10% of premium

## Migration Notes

### For Existing Users
- Previous USD-based reports will still work
- New backtests will default to INR
- Exchange rate is configurable via environment variables

### Environment Variables
```bash
# Optional configuration
export BASE_CURRENCY="INR"
export USD_TO_INR_RATE="85.0"
export DEFAULT_COMMISSION="0.05"
```

## Benefits

1. **🎯 Accurate Modeling**: Realistic Indian trading costs and regulations
2. **💰 Better UX**: Familiar INR currency display for Indian traders  
3. **🏛️ Compliance**: GST and regulatory compliance built-in
4. **📊 Proper P&L**: Fixed leverage calculation for accurate backtesting
5. **🔄 Flexibility**: Configurable exchange rates and fee structures

---

**Note**: This update makes TradeBuddy specifically optimized for Delta Exchange India while maintaining the core functionality and architecture.