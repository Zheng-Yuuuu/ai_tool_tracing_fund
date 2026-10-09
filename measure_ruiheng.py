"""Single-fund case study, using the Q2 report only after publication."""
import numpy as np
import pandas as pd
from plot_result import plt, mdates
from matplotlib.ticker import PercentFormatter
import config as c


def main():
    published = pd.Timestamp('2026-07-21')
    cutoff = pd.Timestamp('2026-08-31')
    holdings = pd.read_csv(c.DATA_DIR / 'ruiheng_verified_holdings.csv', dtype={'stock_code': str})
    nav = pd.read_csv(c.CACHE_DIR / 'fund_nav.csv', dtype={'fund_code': str}, parse_dates=['date'])
    nav = nav[(nav.fund_code == '001832') & (nav.date > published) & (nav.date <= cutoff)].copy()
    stocks = pd.read_csv(c.CACHE_DIR / 'stock_returns.csv', dtype={'stock_code': str}, parse_dates=['date'])
    index = pd.read_csv(c.OUTPUT_DIR / 'aggregate_signal.csv', parse_dates=['date'])
    data = index[(index.date > published) & (index.date <= cutoff)][['date', 'ai_return', 'ai_index']].copy()
    data = data.merge(nav[['date', 'nav', 'daily_return']].rename(columns={'daily_return': 'actual_return'}), on='date', how='left', validate='one_to_one')
    returns = stocks.pivot(index='date', columns='stock_code', values='daily_return').reindex(index=data.date, columns=holdings.stock_code)
    weights = holdings.set_index('stock_code').weight
    data['estimated_return'] = returns.mul(weights, axis=1).sum(axis=1, min_count=1).to_numpy()
    data['covered_weight'] = returns.notna().mul(weights, axis=1).sum(axis=1).to_numpy()
    data['error'] = data.estimated_return - data.actual_return
    data['error_corr_20d'] = data.error.rolling(20, min_periods=20).corr(data.ai_return).clip(-1, 1)
    data['fund_code'] = '001832'
    data['holdings_report_date'] = '2026-06-30'
    data['holdings_published_date'] = published
    data['ai_index_rebased'] = data.ai_index / data.ai_index.iloc[0]
    assert len(data) >= 20 and data.actual_return.notna().all()
    assert data.date.min() > published and data.date.max() <= cutoff
    assert np.allclose(data.covered_weight, weights.sum())
    assert data.error_corr_20d.iloc[:19].isna().all()
    assert np.isclose(data.error_corr_20d.iloc[-1], np.corrcoef(data.error.tail(20), data.ai_return.tail(20))[0, 1])
    data.to_csv(c.OUTPUT_DIR / 'ruiheng_measurement.csv', index=False, encoding='utf-8-sig')

    fig, (ax, bars) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    right = ax.twinx()
    blue, = ax.plot(data.date, data.error_corr_20d, color='#092b70', linewidth=2.3, label='估算误差与硬件日收益的20日相关系数（左轴）')
    red, = right.plot(data.date, data.ai_index_rebased, color='#cb4c4c', linewidth=2, label='AI 硬件指数（右轴）')
    ax.set_ylim(-1, 1)
    ax.set_ylabel('20日相关系数')
    right.set_ylabel('指数（图示首日=1）')
    ax.axhline(0, color='#888888', linewidth=.8)
    ax.legend(handles=[blue, red], loc='upper left', fontsize=9, frameon=False)
    bars.bar(data.date, data.error, width=.8, color='#345d99')
    bars.axhline(0, color='#888888', linewidth=.8)
    bars.set_ylabel('估算收益－实际收益')
    bars.yaxis.set_major_formatter(PercentFormatter(1))
    bars.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    bars.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    for panel in (ax, bars):
        panel.grid(linestyle='--', alpha=.3)
        panel.set_axisbelow(True)
    fig.suptitle('易方达瑞恒（001832）：单基金估算误差实验', fontsize=18, fontweight='bold', y=.97)
    fig.text(.5, .915, '2026-07-22 至 2026-08-31  |  6月底持仓于7月21日披露后使用  |  不使用9月数据', ha='center', fontsize=10)
    fig.subplots_adjust(top=.85, bottom=.19, hspace=.12, left=.09, right=.90)
    fig.text(.09, .105, f'前十大原始权重合计 {weights.sum():.2%}；估算收益不归一化。前19个观测不足20日，蓝线留空。', fontsize=10)
    fig.text(.09, .055, '本图固定研究这一只基金，不按每日拟合结果切换样本；未覆盖的其余仓位会影响误差，不能直接判定调仓。', fontsize=9, color='#666666')
    fig.savefig(c.OUTPUT_DIR / 'ruiheng_measurement.png', dpi=180, facecolor='white')
    plt.close(fig)
    valid = data.dropna(subset=['error_corr_20d']).copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(valid))
    ax.plot(x, valid.error_corr_20d, color='#092b70', marker='o', linewidth=2)
    for i, value in enumerate(valid.error_corr_20d):
        ax.annotate(f'{value:.4f}', (i, value), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=11)
    ax.set_xticks(x, valid.date.dt.strftime('%m-%d'))
    ax.set_ylim(0, max(.65, valid.error_corr_20d.max() + .12))
    ax.set_ylabel('20日滚动 Pearson 相关系数')
    ax.set_xlabel('2026年交易日（按交易日等距排列）')
    ax.grid(linestyle='--', alpha=.3)
    ax.set_title('易方达瑞恒：全部有效20日相关系数', fontsize=17, pad=20)
    fig.subplots_adjust(bottom=.25, top=.84)
    fig.text(.125, .09, '误差 = 估算收益 − 实际收益；2026-07-22起计算，满20个观测后于08-18首次得到相关系数。', fontsize=10)
    fig.text(.125, .035, '07-22至08-17共19个交易日因窗口不足留空；截至08-31，共10个有效系数。纵轴局部放大。', fontsize=10)
    fig.savefig(c.OUTPUT_DIR / 'ruiheng_all_correlations.png', dpi=180, facecolor='white')
    plt.close(fig)
    print('Observations:', len(data), 'First correlation:', data.loc[data.error_corr_20d.notna(), 'date'].min())
    print('Top10 weight:', weights.sum(), 'Mean error:', data.error.mean())
    print(data.tail(1).to_string(index=False))


if __name__ == '__main__':
    main()
