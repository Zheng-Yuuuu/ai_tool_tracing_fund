import argparse
import sys
import pandas as pd
import config as c
from download_data import read_inputs, download_fund_nav, download_fund_holdings, download_stock_returns, data_check
from calculate import calculate
from plot_result import plot_result
from fit_hardware import select_by_historical_fit


def main():
    parser = argparse.ArgumentParser(description='公募基金 AI 硬件隐含调仓监测 MVP')
    parser.add_argument('--limit', type=int, help='阶段验证基金数，例如 1 或 5')
    parser.add_argument('--refresh', action='store_true', help='刷新数据；失败保留旧缓存')
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit 必须大于零')
    funds, pool = read_inputs(args.limit)
    nav = download_fund_nav(funds, args.refresh)
    holdings = download_fund_holdings(funds, args.refresh)
    stocks = download_stock_returns(holdings, pool, args.refresh)
    data_check(nav, holdings, stocks)
    print('\nFund NAV:\n', nav.head().to_string(index=False))
    print('\nFund Holdings:\n', holdings.head(10).to_string(index=False))
    print('\nStock Returns:\n', stocks.head(10).to_string(index=False))
    if args.download_only:
        return
    exposure_pool = pd.read_csv(c.DATA_DIR / 'ai_hardware_classification.csv', dtype=str)
    errors, signals, aggregate = calculate(nav, holdings, stocks, pool, exposure_pool=exposure_pool)
    signals, aggregate = select_by_historical_fit(holdings, stocks, aggregate, signals)
    if errors.estimated_return.notna().sum() == 0:
        raise ValueError('没有可计算的理论收益；STOP')
    c.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in [('fund_daily_error', errors), ('fund_signals', signals), ('aggregate_signal', aggregate)]:
        frame.to_csv(c.OUTPUT_DIR / f'{name}.csv', index=False, encoding='utf-8-sig')
    plot_result(aggregate, errors, signals)
    latest = signals[signals.date == signals.date.max()].copy()
    latest.to_csv(c.OUTPUT_DIR / 'latest_fund_groups.csv', index=False, encoding='utf-8-sig')
    fit = latest.copy()
    fit.to_csv(c.OUTPUT_DIR / 'latest_hardware_fit.csv', index=False, encoding='utf-8-sig')
    print('\n前十大持仓拟合检验：\n', fit[['fund_code', 'fund_name', 'group', 'validation_corr', 'oos_r2', 'fit_status']].to_string(index=False))
    print(f'\n最新基金分组（拟合通过且AI硬件仓位 > {c.AI_HEAVY_THRESHOLD:.0%}）：\n', latest[['fund_code', 'fund_name', 'reported_ai_exposure', 'group']].to_string(index=False))
    if aggregate.control_positioning.notna().sum() == 0 or aggregate.heavy_positioning.notna().sum() == 0:
        print('WARNING: Heavy / Control 缺少有效20日信号，Diff Signal 保持空值。请人工补充基金名单。')
    print(f'DONE: {c.OUTPUT_DIR}')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
