"""Check Ruiheng at an explicit historical cutoff, without later price data."""
import argparse
import numpy as np
import pandas as pd
import config as c
from download_data import download_stock_returns
from fit_hardware import select_by_historical_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--as-of', default='2026-08-31')
    args = parser.parse_args()
    asof = pd.Timestamp(args.as_of)
    funds = pd.DataFrame({'fund_code': ['001832'], 'fund_name': ['易方达瑞恒灵活配置混合']})
    aggregate = pd.read_csv(c.OUTPUT_DIR / 'aggregate_signal.csv', parse_dates=['date'])
    aggregate = aggregate[aggregate.date <= asof].copy()
    if aggregate.empty:
        raise ValueError('No benchmark data before cutoff')
    c.END_DATE = aggregate.date.max().strftime('%Y-%m-%d')
    # Transcribed from the published 2026 Q2 report; provenance retained per row.
    # Separate from the shared download cache to avoid implying a full history.
    holdings = pd.read_csv(c.DATA_DIR / 'ruiheng_verified_holdings.csv', dtype={'fund_code': str, 'stock_code': str}, parse_dates=['report_date'])
    # The source page identifies the quarterly report release as 2026-07-21.
    published = pd.Timestamp('2026-07-21')
    if asof <= published:
        raise ValueError('This verified report is not usable before its publication')
    holdings = holdings[holdings.report_date == holdings.report_date.max()].copy()
    pool = pd.read_csv(c.DATA_DIR / 'ai_hardware.csv', dtype=str)
    classification = pd.read_csv(c.DATA_DIR / 'ai_hardware_classification.csv', dtype=str)
    stocks = download_stock_returns(holdings, pool)
    stocks = stocks[stocks.date <= asof].copy()
    holdings['in_ai_classification'] = holdings.stock_code.isin(classification.stock_code)
    exposure = holdings.loc[holdings.in_ai_classification, 'weight'].sum()
    latest_date = aggregate.date.max()
    signals = pd.DataFrame(dict(date=[latest_date], fund_code=['001832'], fund_name=[funds.fund_name.iloc[0]],
                                reported_ai_exposure=[exposure], group=['AI Heavy' if exposure > c.AI_HEAVY_THRESHOLD else 'Other'], corr_20d=[np.nan]))
    result, _ = select_by_historical_fit(holdings, stocks, aggregate, signals)
    window = aggregate.loc[aggregate.date < latest_date, 'date'].tail(c.FIT_WINDOW)
    result['report_published_date'] = published
    result['fit_window_start'] = window.iloc[0]
    result['training_end'] = window.iloc[c.FIT_TRAIN_DAYS - 1]
    result['validation_start'] = window.iloc[c.FIT_TRAIN_DAYS]
    assert result.fit_window_end.max() < asof
    assert stocks.date.max() <= asof
    result.to_csv(c.OUTPUT_DIR / 'ruiheng_fit.csv', index=False, encoding='utf-8-sig')
    holdings.to_csv(c.OUTPUT_DIR / 'ruiheng_holdings.csv', index=False, encoding='utf-8-sig')
    print(holdings.to_string(index=False))
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
