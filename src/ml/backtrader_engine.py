"""
backtrader_engine.py
Backtesting-Engine für TA- und ML-Strategien mit Parametrisierung und Reporting.
"""
import backtrader as bt
import pandas as pd
import os

class SignalStrategy(bt.Strategy):
    params = (
        ('signals', None),
        ('commission', 0.001),
        ('slippage', 0.001),
        ('size', 1),
    )
    def __init__(self):
        self.order = None
        self.signal_idx = 0
    def next(self):
        if self.p.signals is not None and self.signal_idx < len(self.p.signals):
            signal = self.p.signals[self.signal_idx]
            if signal == 1 and not self.position:
                self.buy(size=self.p.size)
            elif signal == 0 and self.position:
                self.close()
            self.signal_idx += 1

class EquityObserver(bt.Observer):
    lines = ('equity',)
    plotinfo = dict(plot=True, subplot=True)
    def next(self):
        self.lines.equity[0] = self._owner.broker.getvalue()

def run_backtest(df: pd.DataFrame, signals, strategy_name: str, commission: float = 0.001, slippage: float = 0.001, size: int = 1, report_dir: str = "backtest/reports") -> pd.DataFrame:
    """Führt einen Backtest mit backtrader durch und exportiert die Ergebnisse als CSV und HTML."""
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(SignalStrategy, signals=signals, commission=commission, slippage=slippage, size=size)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(slippage)
    cerebro.broker.setcash(10000)
    cerebro.addobserver(EquityObserver)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    result = cerebro.run()
    strat = result[0]
    # Suche nach dem EquityObserver im Observer-Array
    equity = None
    for obs in strat.observers:
        if isinstance(obs, EquityObserver):
            equity = list(obs.lines.equity.get(size=len(df)))
            break
    if equity is None:
        # Fallback: Leere Liste, falls Observer nicht gefunden
        equity = [None] * len(df)
    dates = df.index[:len(equity)]
    report = pd.DataFrame({"date": dates, "equity": equity})
    # Export
    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, f"{strategy_name}_backtest.csv")
    report.to_csv(csv_path, index=False)
    # HTML-Export (simple)
    html_path = os.path.join(report_dir, f"{strategy_name}_backtest.html")
    report.to_html(html_path, index=False)
    return report 