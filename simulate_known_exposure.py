"""Controlled synthetic fund: disclosed 40%, true 40->80->40% hardware weight."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import config as c


# Ten names drawn from the existing index, spanning optics, PCB, servers and chips.
TEN_CODES = ['300308', '300502', '002463', '002916', '300476',
             '601138', '000977', '688041', '688256', '688008']
WINDOW = 20
SHORT_WINDOW = 10
FAST_WINDOW = 5
SEED = 20261008
START, UP_START, PEAK, DOWN_END = map(pd.Timestamp,
    ['2026-01-01', '2026-05-01', '2026-07-01', '2026-09-30'])


def exposure_path(dates, final_weight=.40):
    dates = pd.DatetimeIndex(dates)
    weight = np.full(len(dates), .40)
    up = (dates >= UP_START) & (dates <= PEAK)
    down = (dates > PEAK) & (dates <= DOWN_END)
    weight[up] = .40 + .40 * (dates[up] - UP_START).days / (PEAK - UP_START).days
    weight[down] = .80 + (final_weight - .80) * (dates[down] - PEAK).days / (DOWN_END - PEAK).days
    return weight


def simulate(stocks, component_codes=TEN_CODES, noise_seed=SEED):
    chosen = stocks[stocks.stock_code.isin(component_codes)].copy()
    prices = chosen.pivot(index='date', columns='stock_code', values='daily_return')
    prices = prices.reindex(columns=component_codes).sort_index()
    prices = prices.loc[prices.index >= START]
    if prices.empty or prices.isna().any().any():
        raise ValueError('The ten benchmark stocks need complete daily returns')
    dates = prices.index
    sector = prices.mean(axis=1).to_numpy()
    actual_weight = exposure_path(dates)
    actual_weight_20 = exposure_path(dates, final_weight=.20)
    other = np.random.default_rng(noise_seed).normal(0, .006, len(dates))
    # Every one of the ten holdings has identical weight, and the benchmark
    # contains exactly those same ten stocks with identical index weights.
    # The last disclosed weights stay 4% each throughout; only latent true
    # holdings drift. The residual (1-w) earns an unrelated synthetic return.
    estimated = .40 * sector
    actual = actual_weight * sector + (1 - actual_weight) * other
    actual_20 = actual_weight_20 * sector + (1 - actual_weight_20) * other
    error = estimated - actual
    error_20 = estimated - actual_20
    data = pd.DataFrame(dict(date=dates, ai_return=sector,
                             benchmark_index=(1 + sector).cumprod(),
                             disclosed_top10_weight=.40,
                             actual_top10_weight=actual_weight,
                             actual_top10_weight_20_end=actual_weight_20,
                             actual_weight_per_stock=actual_weight / 10,
                             disclosed_weight_per_stock=.04,
                             non_hardware_return=other,
                             estimated_return=estimated,
                             actual_fund_return=actual,
                             actual_fund_return_20_end=actual_20,
                             error=error,
                             error_20_end=error_20))
    data['error_corr_20d'] = data.error.rolling(WINDOW, min_periods=WINDOW).corr(data.ai_return).clip(-1, 1)
    data['error_corr_10d'] = data.error.rolling(SHORT_WINDOW, min_periods=SHORT_WINDOW).corr(data.ai_return).clip(-1, 1)
    data['error_corr_5d'] = data.error.rolling(FAST_WINDOW, min_periods=FAST_WINDOW).corr(data.ai_return).clip(-1, 1)
    data['error_corr_20d_20_end'] = data.error_20_end.rolling(WINDOW, min_periods=WINDOW).corr(data.ai_return).clip(-1, 1)
    data['error_corr_5d_20_end'] = data.error_20_end.rolling(FAST_WINDOW, min_periods=FAST_WINDOW).corr(data.ai_return).clip(-1, 1)
    data['benchmark_index_rebased'] = data.benchmark_index / data.benchmark_index.iloc[0]
    return data


def draw(data, path):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (top, middle, bottom) = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True,
                                              gridspec_kw={'height_ratios': [2.1, 1, 1]})
    red_axis = top.twinx()
    blue, = top.plot(data.date, data.error_corr_20d, color='#092b70', linewidth=2.2,
                     label='20日滚动相关系数（左轴）')
    cyan, = top.plot(data.date, data.error_corr_10d, color='#138f9b', linewidth=1.5,
                     alpha=.9, label='40%终点：10日相关系数（左轴）')
    fast_a, = top.plot(data.date, data.error_corr_5d, color='#59a78a', linewidth=1,
                       linestyle=':', alpha=.72, label='40%终点：5日相关系数（左轴）')
    purple, = top.plot(data.date, data.error_corr_20d_20_end, color='#8b44a2',
                       linestyle='--', linewidth=2, label='20%终点：20日相关系数（左轴）')
    fast_b, = top.plot(data.date, data.error_corr_5d_20_end, color='#c27ac8',
                       linestyle=':', linewidth=1, alpha=.72, label='20%终点：5日相关系数（左轴）')
    red, = red_axis.plot(data.date, data.benchmark_index_rebased, color='#cb4c4c',
                         linewidth=1.9, label='同10股AI硬件指数（右轴）')
    top.set_ylim(-1, 1)
    top.set_ylabel('滚动相关系数')
    red_axis.set_ylabel('硬件指数，图示首日=1')
    blue.set_label('40%终点：20日相关系数（左轴）')
    top.legend(handles=[blue, cyan, fast_a, purple, fast_b, red], loc='upper left', fontsize=8, frameon=False, ncol=2)
    middle.plot(data.date, data.actual_top10_weight, color='#16785b', linewidth=2,
                label='情景A：9月底回到40%')
    middle.plot(data.date, data.actual_top10_weight_20_end, color='#8b44a2',
                linestyle='--', linewidth=1.8, label='情景B：9月底降到20%')
    middle.axhline(.4, color='#777777', linestyle='--', linewidth=1, label='用于估算的披露仓位：40%')
    middle.set_ylim(.15, .9)
    middle.yaxis.set_major_formatter(PercentFormatter(1))
    middle.set_ylabel('已知模拟仓位')
    middle.legend(loc='upper left', fontsize=9, frameon=False)
    bottom.plot(data.date, data.error, color='#53719a', linewidth=1, label='情景A每日误差')
    bottom.axhline(0, color='#888888', linewidth=.8)
    bottom.yaxis.set_major_formatter(PercentFormatter(1))
    bottom.set_ylabel('每日估算误差')
    bottom.legend(loc='upper left', fontsize=9, frameon=False)
    for ax in (top, middle, bottom):
        ax.grid(linestyle='--', alpha=.32)
        ax.set_axisbelow(True)
    bottom.xaxis.set_major_locator(mdates.MonthLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.suptitle('合成基金对照实验：真实仓位已知时，误差相关系数如何变化', fontsize=18, fontweight='bold', y=.97)
    fig.subplots_adjust(top=.90, bottom=.13, hspace=.13, left=.09, right=.91)
    fig.text(.09, .075, '真实收益 = 实际硬件仓位×10股指数日收益 + 其余仓位×独立模拟收益；估算收益固定为40%×指数日收益。', fontsize=10)
    fig.text(.09, .04, '两情景5月起加仓、7月初达80%；9月底分别降到40%和20%。细点线为5日滚动，波动更大。', fontsize=9, color='#555555')
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)


def main():
    cache = c.CACHE_DIR / 'stock_returns.csv'
    if cache.exists():
        stocks = pd.read_csv(cache, dtype={'stock_code': str}, parse_dates=['date'])
    else:
        from download_data import download_stock_returns
        pool = pd.read_csv(c.DATA_DIR / 'ai_hardware.csv', dtype=str)
        stocks = download_stock_returns(pd.DataFrame(columns=['stock_code', 'stock_name']),
                                         pool[pool.stock_code.isin(TEN_CODES)])
    data = simulate(stocks)
    assert set(TEN_CODES).issubset(set(stocks.stock_code))
    assert np.allclose(data.estimated_return, .4 * data.ai_return)
    assert np.isclose(data.actual_top10_weight.iloc[0], .4)
    assert np.isclose(data.actual_top10_weight.iloc[-1], .4)
    assert np.isclose(data.actual_top10_weight_20_end.iloc[-1], .2)
    assert np.allclose(data.actual_top10_weight_20_end[data.date < PEAK], data.actual_top10_weight[data.date < PEAK])
    assert np.isclose(data.actual_top10_weight.max(), .8)
    assert data.error_corr_20d.iloc[:WINDOW-1].isna().all()
    assert data.error_corr_10d.iloc[:SHORT_WINDOW-1].isna().all()
    assert data.error_corr_5d.iloc[:FAST_WINDOW-1].isna().all()
    assert data.error_corr_5d_20_end.iloc[:FAST_WINDOW-1].isna().all()
    assert np.isclose(data.error_corr_20d.iloc[-1], np.corrcoef(data.error.tail(WINDOW), data.ai_return.tail(WINDOW))[0, 1])
    assert np.isclose(data.error_corr_10d.iloc[-1], np.corrcoef(data.error.tail(SHORT_WINDOW), data.ai_return.tail(SHORT_WINDOW))[0, 1])
    assert np.isclose(data.error_corr_20d_20_end.iloc[-1], np.corrcoef(data.error_20_end.tail(WINDOW), data.ai_return.tail(WINDOW))[0, 1])
    assert np.isclose(data.error_corr_5d.iloc[-1], np.corrcoef(data.error.tail(FAST_WINDOW), data.ai_return.tail(FAST_WINDOW))[0, 1])
    assert np.isclose(data.error_corr_5d_20_end.iloc[-1], np.corrcoef(data.error_20_end.tail(FAST_WINDOW), data.ai_return.tail(FAST_WINDOW))[0, 1])
    c.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(c.DATA_DIR / 'ai_hardware.csv', dtype=str)
    components = pool.set_index('stock_code').loc[TEN_CODES].reset_index()
    components['index_weight'] = .10
    components['estimated_fund_weight'] = .04
    components.to_csv(c.OUTPUT_DIR / 'synthetic_ten_components.csv', index=False, encoding='utf-8-sig')
    data.to_csv(c.OUTPUT_DIR / 'synthetic_exposure_compare.csv', index=False, encoding='utf-8-sig')
    draw(data, c.OUTPUT_DIR / 'synthetic_exposure_compare.png')
    print('Rows:', len(data), 'range:', data.date.min(), data.date.max())
    print(data.groupby(data.date.dt.to_period('M')).agg(actual_weight=('actual_top10_weight', 'last'),
                mean_rolling_corr=('error_corr_20d', 'mean')).to_string())


if __name__ == '__main__':
    main()
