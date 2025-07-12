"""
HTML report generator for TradeBuddy backtesting results.

Creates professional interactive HTML reports with Plotly charts
and comprehensive performance analysis.
"""

import json
import structlog
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from jinja2 import Environment, BaseLoader

from src.backtesting.models import BacktestResult, BacktestTrade
from src.utils.helpers import to_float
from src.core.constants import TradingConstants


logger = structlog.get_logger(__name__)


class BacktestReportGenerator:
    """
    Professional HTML report generator for backtesting results.
    
    Creates comprehensive reports with interactive charts,
    detailed metrics, and professional styling.
    """

    def __init__(self, currency: str = "INR"):
        """Initialize report generator.
        
        Args:
            currency: Currency for reporting (INR or USD)
        """
        self.template_env = Environment(loader=BaseLoader())
        self.currency = currency
        self.currency_symbol = TradingConstants.CURRENCY_SYMBOL if currency == "INR" else "$"
        self.exchange_rate = TradingConstants.USD_TO_INR if currency == "INR" else 1.0
    
    def _convert_currency(self, value: float) -> float:
        """Convert USD value to reporting currency."""
        return value * self.exchange_rate
    
    def _format_currency(self, value: float) -> str:
        """Format value with currency symbol."""
        converted = self._convert_currency(value)
        if self.currency == "INR":
            # Indian number formatting (lakhs/crores)
            if abs(converted) >= 10000000:  # 1 crore
                return f"{self.currency_symbol}{converted/10000000:.2f} Cr"
            elif abs(converted) >= 100000:  # 1 lakh
                return f"{self.currency_symbol}{converted/100000:.2f} L"
            else:
                return f"{self.currency_symbol}{converted:,.0f}"
        else:
            return f"{self.currency_symbol}{converted:,.2f}"
        
    def generate_report(
        self, 
        result: BacktestResult, 
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive HTML report.

        Args:
            result: Backtesting results
            output_path: Output file path (optional)

        Returns:
            Path to generated HTML report
        """
        logger.info("Generating backtesting HTML report")
        
        # Generate filename if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{result.config.symbol}_{result.config.strategy_type}_{timestamp}.html"
            output_path = f"backtest_reports/{filename}"
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate charts
        charts = self._generate_charts(result)
        
        # Generate HTML content
        html_content = self._generate_html(result, charts)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(
            "HTML report generated successfully",
            output_path=output_path,
            file_size_kb=len(html_content) // 1024
        )
        
        return output_path

    def _generate_charts(self, result: BacktestResult) -> Dict[str, str]:
        """Generate all Plotly charts as HTML."""
        charts = {}
        
        # 1. Equity Curve Chart
        charts['equity_curve'] = self._create_equity_curve_chart(result)
        
        # 2. Drawdown Chart
        charts['drawdown'] = self._create_drawdown_chart(result)
        
        # 3. Trade Distribution Chart
        charts['trade_distribution'] = self._create_trade_distribution_chart(result)
        
        # 4. Performance Metrics Chart
        charts['performance_metrics'] = self._create_performance_metrics_chart(result)
        
        # 5. Monthly Returns Heatmap
        charts['monthly_returns'] = self._create_monthly_returns_chart(result)
        
        return charts

    def _create_equity_curve_chart(self, result: BacktestResult) -> str:
        """Create equity curve vs benchmark chart."""
        if not result.equity_curve:
            return "<div>No equity data available</div>"
        
        fig = go.Figure()
        
        # Strategy equity curve
        dates = [point.timestamp for point in result.equity_curve]
        equity_values = [point.equity for point in result.equity_curve]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_values,
            mode='lines',
            name='Strategy',
            line=dict(color='#2E8B57', width=2),
            hovertemplate=f'Date: %{{x}}<br>Equity: {self.currency_symbol}%{{y:,.2f}}<extra></extra>'
        ))
        
        # Benchmark if available
        benchmark_values = [point.benchmark_value for point in result.equity_curve if point.benchmark_value]
        if benchmark_values and len(benchmark_values) == len(dates):
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_values,
                mode='lines',
                name='Buy & Hold',
                line=dict(color='#FF6347', width=2, dash='dash'),
                hovertemplate=f'Date: %{{x}}<br>Benchmark: {self.currency_symbol}%{{y:,.2f}}<extra></extra>'
            ))
        
        # Add trade markers
        for trade in result.trades[:50]:  # Limit to 50 trades for readability
            marker_color = 'green' if trade.trade_action == 'LONG' else 'red'
            marker_symbol = 'triangle-up' if trade.trade_action == 'LONG' else 'triangle-down'
            
            fig.add_trace(go.Scatter(
                x=[trade.entry_time],
                y=[trade.entry_price],
                mode='markers',
                name=f'{trade.trade_action} Entry',
                marker=dict(
                    symbol=marker_symbol,
                    size=8,
                    color=marker_color,
                    line=dict(color='white', width=1)
                ),
                showlegend=False,
                hovertemplate=f'Trade #{trade.trade_id}<br>Action: {trade.trade_action}<br>Price: {self.currency_symbol}%{{y:,.2f}}<br>Date: %{{x}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Portfolio Equity Curve - {result.config.symbol} {result.config.strategy_type.title()}",
            xaxis_title="Date",
            yaxis_title=f"Portfolio Value ({self.currency_symbol})",
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='inline')

    def _create_drawdown_chart(self, result: BacktestResult) -> str:
        """Create drawdown underwater chart."""
        if not result.equity_curve:
            return "<div>No drawdown data available</div>"
        
        fig = go.Figure()
        
        dates = [point.timestamp for point in result.equity_curve]
        drawdowns = [-point.drawdown_pct for point in result.equity_curve]  # Negative for underwater effect
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdowns,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=1),
            name='Drawdown',
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title="Portfolio Drawdown (Underwater Curve)",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_trade_distribution_chart(self, result: BacktestResult) -> str:
        """Create trade P&L distribution chart."""
        if not result.trades:
            return "<div>No trade data available</div>"
        
        # Prepare data
        pnl_values = [trade.pnl_pct or 0 for trade in result.trades]
        trade_actions = [trade.trade_action for trade in result.trades]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('P&L Distribution', 'Trade Returns Scatter'),
            vertical_spacing=0.15
        )
        
        # Histogram of P&L
        fig.add_trace(
            go.Histogram(
                x=pnl_values,
                nbinsx=30,
                name='P&L Distribution',
                marker_color='lightblue',
                opacity=0.7,
                hovertemplate='Return Range: %{x:.1f}%<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Scatter plot of trades over time
        trade_dates = [trade.entry_time for trade in result.trades]
        colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_values]
        
        fig.add_trace(
            go.Scatter(
                x=trade_dates,
                y=pnl_values,
                mode='markers',
                name='Individual Trades',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add zero line to scatter plot
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
        
        fig.update_layout(
            title="Trade Analysis",
            template="plotly_white",
            height=700,
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_xaxes(title_text="Return (%)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_performance_metrics_chart(self, result: BacktestResult) -> str:
        """Create performance metrics radar chart."""
        metrics = result.performance_metrics
        
        # Normalize metrics for radar chart (0-10 scale)
        categories = [
            'Total Return',
            'Sharpe Ratio', 
            'Win Rate',
            'Profit Factor',
            'Recovery Factor',
            'Low Drawdown'
        ]
        
        values = [
            min(10, max(0, metrics.total_return_pct / 10)),  # 10% = score of 10
            min(10, max(0, metrics.sharpe_ratio * 2)),  # Sharpe 5 = score of 10
            min(10, max(0, metrics.win_rate_pct / 10)),  # 100% = score of 10
            min(10, max(0, metrics.profit_factor * 2)),  # PF 5 = score of 10
            min(10, max(0, metrics.recovery_factor)),  # RF 10 = score of 10
            min(10, max(0, (20 - metrics.max_drawdown_pct) / 2))  # 0% DD = score of 10
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(46, 139, 87, 0.3)',
            line=dict(color='#2E8B57', width=2),
            name='Strategy Performance',
            hovertemplate='%{theta}: %{r:.1f}/10<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickmode='linear',
                    tick0=0,
                    dtick=2
                )
            ),
            title="Performance Metrics Radar",
            template="plotly_white",
            height=500
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_monthly_returns_chart(self, result: BacktestResult) -> str:
        """Create monthly returns heatmap."""
        if not result.equity_curve or len(result.equity_curve) < 30:
            return "<div>Insufficient data for monthly returns analysis</div>"
        
        # Calculate monthly returns (simplified)
        monthly_data = {}
        prev_month_equity = result.initial_capital
        
        for point in result.equity_curve[::24]:  # Sample every 24 points for monthly approximation
            month_key = point.timestamp.strftime("%Y-%m")
            if month_key not in monthly_data:
                monthly_return = ((point.equity - prev_month_equity) / prev_month_equity) * 100
                monthly_data[month_key] = monthly_return
                prev_month_equity = point.equity
        
        if not monthly_data:
            return "<div>No monthly data available</div>"
        
        months = list(monthly_data.keys())
        returns = list(monthly_data.values())
        
        fig = go.Figure()
        
        colors = ['green' if r >= 0 else 'red' for r in returns]
        
        fig.add_trace(go.Bar(
            x=months,
            y=returns,
            marker_color=colors,
            opacity=0.7,
            name='Monthly Returns',
            hovertemplate='Month: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title="Monthly Returns",
            xaxis_title="Month",
            yaxis_title="Return (%)",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _generate_html(self, result: BacktestResult, charts: Dict[str, str]) -> str:
        """Generate complete HTML report."""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradeBuddy Backtesting Report - {{ symbol }} {{ strategy }}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; 
            padding: 20px; 
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(135deg, #2E8B57, #3CB371);
            color: white; 
            padding: 30px;
            text-align: center;
        }
        .header h1 { margin: 0; font-size: 2.5em; font-weight: 300; }
        .header p { margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }
        .section { 
            padding: 30px;
            border-bottom: 1px solid #eee;
        }
        .section:last-child { border-bottom: none; }
        .section h2 { 
            color: #2E8B57;
            margin-top: 0;
            font-size: 1.8em;
            font-weight: 400;
        }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card { 
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2E8B57;
        }
        .metric-value { 
            font-size: 2em;
            font-weight: bold;
            color: #2E8B57;
            margin: 0;
        }
        .metric-label { 
            color: #666;
            font-size: 0.9em;
            margin: 5px 0 0 0;
        }
        .chart-container { 
            margin: 20px 0;
            background: white;
            border-radius: 8px;
        }
        .trade-table { 
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
        }
        .trade-table th,
        .trade-table td { 
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .trade-table th { 
            background-color: #f8f9fa;
            font-weight: 600;
            color: #555;
        }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .config-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
        }
        .config-label { font-weight: 600; color: #555; }
        .config-value { color: #333; margin-top: 5px; }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 TradeBuddy Backtesting Report</h1>
            <p>{{ symbol }} • {{ strategy }} Strategy • {{ start_date }} to {{ end_date }}</p>
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {{ 'positive' if total_return|float > 0 else 'negative' }}">{{ total_return }}%</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ sharpe_ratio }}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {{ 'negative' if max_drawdown|float > 0 else 'positive' }}">{{ max_drawdown }}%</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ win_rate }}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ total_trades }}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ profit_factor }}</div>
                    <div class="metric-label">Profit Factor</div>
                </div>
            </div>
        </div>

        <!-- Configuration -->
        <div class="section">
            <h2>⚙️ Configuration</h2>
            <div class="config-grid">
                <div class="config-item">
                    <div class="config-label">Strategy</div>
                    <div class="config-value">{{ strategy }}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Symbol</div>
                    <div class="config-value">{{ symbol }}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Timeframe</div>
                    <div class="config-value">{{ timeframe }}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Initial Capital</div>
                    <div class="config-value">${{ initial_capital }}</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Leverage</div>
                    <div class="config-value">{{ leverage }}x</div>
                </div>
                <div class="config-item">
                    <div class="config-label">Commission</div>
                    <div class="config-value">{{ commission }}%</div>
                </div>
            </div>
        </div>

        <!-- Charts -->
        <div class="section">
            <h2>📈 Portfolio Performance</h2>
            <div class="chart-container">
                {{ equity_curve|safe }}
            </div>
        </div>

        <div class="section">
            <h2>📉 Risk Analysis</h2>
            <div class="chart-container">
                {{ drawdown|safe }}
            </div>
        </div>

        <div class="section">
            <h2>🎯 Trade Analysis</h2>
            <div class="chart-container">
                {{ trade_distribution|safe }}
            </div>
        </div>

        <div class="section">
            <h2>⭐ Performance Radar</h2>
            <div class="chart-container">
                {{ performance_metrics|safe }}
            </div>
        </div>

        <!-- Trade Log -->
        <div class="section">
            <h2>📋 Trade Log (Last 20 Trades)</h2>
            <table class="trade-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Action</th>
                        <th>Entry Date</th>
                        <th>Entry Price</th>
                        <th>Exit Date</th>
                        <th>Exit Price</th>
                        <th>P&L %</th>
                        <th>P&L $</th>
                        <th>Duration</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in recent_trades %}
                    <tr>
                        <td>{{ trade.trade_id }}</td>
                        <td>{{ trade.trade_action }}</td>
                        <td>{{ trade.entry_time.strftime('%Y-%m-%d %H:%M') }}</td>
                        <td>${{ trade.entry_price }}</td>
                        <td>{{ trade.exit_time.strftime('%Y-%m-%d %H:%M') if trade.exit_time else 'Open' }}</td>
                        <td>${{ trade.exit_price }}</td>
                        <td class="{{ 'positive' if trade.pnl_positive else 'negative' }}">
                            {{ trade.pnl_pct }}{% if trade.pnl_pct != '-' %}%{% endif %}
                        </td>
                        <td class="{{ 'positive' if trade.pnl_usd_positive else 'negative' }}">
                            ${{ trade.pnl_usd }}
                        </td>
                        <td>{{ trade.duration_hours }}{% if trade.duration_hours != '-' %}h{% endif %}</td>
                        <td>{{ trade.confidence_score }}/10</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Generated by TradeBuddy Backtesting Engine • {{ generation_time }}</p>
            <p>⚠️ Past performance does not guarantee future results</p>
        </div>
    </div>
</body>
</html>
        """
        
        template = self.template_env.from_string(html_template)
        
        # Prepare template variables with safe formatting
        strategy_name = str(result.config.strategy_type).replace('_', ' ').title()
        
        template_vars = {
            'symbol': str(result.config.symbol),
            'strategy': strategy_name,
            'start_date': result.start_date.strftime('%Y-%m-%d'),
            'end_date': result.end_date.strftime('%Y-%m-%d'),
            'total_return': f"{result.performance_metrics.total_return_pct:.2f}",
            'sharpe_ratio': f"{result.performance_metrics.sharpe_ratio:.2f}",
            'max_drawdown': f"{result.performance_metrics.max_drawdown_pct:.2f}",
            'win_rate': f"{result.performance_metrics.win_rate_pct:.1f}",
            'total_trades': result.performance_metrics.total_trades,
            'profit_factor': f"{result.performance_metrics.profit_factor:.2f}",
            'timeframe': str(result.config.timeframe),
            'initial_capital': f"{result.initial_capital:,.2f}",
            'leverage': result.config.leverage,
            'commission': f"{result.config.commission_pct:.1f}",
            'equity_curve': charts.get('equity_curve', ''),
            'drawdown': charts.get('drawdown', ''),
            'trade_distribution': charts.get('trade_distribution', ''),
            'performance_metrics': charts.get('performance_metrics', ''),
            'recent_trades': self._format_trades_for_template(result.trades[-20:] if result.trades else []),
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        return template.render(**template_vars)
    
    def _format_trades_for_template(self, trades):
        """Format trades for template display with safe string formatting."""
        formatted_trades = []
        for trade in trades:
            formatted_trade = {
                'trade_id': trade.trade_id,
                'trade_action': trade.trade_action,
                'entry_time': trade.entry_time,
                'entry_price': f"{trade.entry_price:,.2f}",
                'exit_time': trade.exit_time,
                'exit_price': f"{trade.exit_price:,.2f}" if trade.exit_price else '-',
                'pnl_pct': f"{trade.pnl_pct:.2f}" if trade.pnl_pct is not None else '-',
                'pnl_usd': f"{trade.pnl_usd:,.2f}" if trade.pnl_usd is not None else '-',
                'duration_hours': f"{trade.duration_minutes/60:.0f}" if trade.duration_minutes else '-',
                'confidence_score': trade.confidence_score,
                'pnl_positive': trade.pnl_pct is not None and trade.pnl_pct > 0,
                'pnl_usd_positive': trade.pnl_usd is not None and trade.pnl_usd > 0,
            }
            formatted_trades.append(formatted_trade)
        return formatted_trades