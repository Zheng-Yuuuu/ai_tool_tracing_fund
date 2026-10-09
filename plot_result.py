"""Rolling correlation of mean error and sector return in blue, cumulative AI index in red."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import config as c

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def daily_error_comparison(aggregate, errors, signals):
    groups = signals[['date', 'fund_code', 'group']]
    merged = errors[['date', 'fund_code', 'error']].merge(groups, on=['date', 'fund_code'], how='left', validate='one_to_one')
    heavy = merged[merged.group == 'AI Heavy'].groupby('date')
    summary = heavy.agg(heavy_mean_error=('error', 'mean'), heavy_valid_error_count=('error', 'count')).reset_index()
    data = aggregate[['date', 'ai_return', 'ai_index', 'heavy_fund_count']].merge(summary, on='date', how='left', validate='one_to_one')
    data['heavy_valid_error_count'] = data.heavy_valid_error_count.fillna(0).astype(int)
    data = data.sort_values('date')
    data['heavy_error_corr_20d'] = data.heavy_mean_error.rolling(20, min_periods=20).corr(data.ai_return).clip(-1, 1)
    return data


def draw_chart(data, path):
    data = data.sort_values('date').copy()
    valid_index = data.ai_index.dropna()
    if valid_index.empty:
        raise ValueError('没有有效 AI 硬件指数可绘图')
    baseline = valid_index.iloc[0]
    fig, left = plt.subplots(figsize=(14, 6.8))
    right = left.twinx()
    fig.subplots_adjust(left=.075, right=.925, bottom=.23, top=.77)
    fig.suptitle('AI 硬件拟合通过基金：估算误差相关系数与 AI 硬件走势', fontsize=19, y=.96, fontweight='bold')
    period = f'{data.date.iloc[0]:%Y-%m-%d} — {data.date.iloc[-1]:%Y-%m-%d}'
    fig.text(.5, .885, f'{period}  |  先平均误差，再计算20日滚动相关系数  |  拟合通过且AI硬件仓位 > {c.AI_HEAVY_THRESHOLD:.0%}', ha='center', fontsize=11, color='#555555')
    blue, = left.plot(data.date, data.heavy_error_corr_20d, color='#092b70', linewidth=2.2, label='平均估算误差与硬件日收益的20日相关系数（左轴）', zorder=3)
    red, = right.plot(data.date, data.ai_index / baseline, color='#cb4c4c', linewidth=2, label='AI 硬件等权指数（右轴）', zorder=4)
    left.set_ylim(-1, 1)
    left.yaxis.set_major_locator(MultipleLocator(.2))
    left.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    left.set_ylabel('估算误差与硬件板块日收益的相关系数', fontsize=11, labelpad=12)
    right.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    right.set_ylabel('AI 硬件指数（图示首日 = 1）', fontsize=11, labelpad=12)
    left.axhline(0, color='#777777', linewidth=.9)
    left.grid(color='#d5d5d5', linestyle='--', linewidth=.8, alpha=.85)
    left.set_axisbelow(True)
    left.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    left.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    left.margins(x=.01)
    for ax in (left, right):
        ax.tick_params(labelsize=10, colors='#555555')
        ax.spines['top'].set_visible(False)
    fig.legend(handles=[blue, red], loc='upper center', bbox_to_anchor=(.5, .845), ncol=2, frameon=False, fontsize=10)
    fig.text(.075, .13, '误差 = 估算收益 − 实际收益；先对当天重仓组误差取算术平均，再与硬件日收益计算20日相关系数，保留原始正负号。', fontsize=10, color='#444444')
    fig.text(.075, .085, '蓝线更负：与较高实际硬件暴露的假设一致；更正：与较低暴露的假设一致。相关系数是研究代理指标。', fontsize=10, color='#444444')
    fig.text(.075, .04, '分组采用独立AI硬件分类名单；红线保留原20股指数。静态分类、报告日期前视偏差及未覆盖仓位会影响结果。', fontsize=9, color='#777777')
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)


def plot_result(data, errors=None, signals=None):
    if errors is None:
        errors = pd.read_csv(c.OUTPUT_DIR / 'fund_daily_error.csv', dtype={'fund_code': str})
    if signals is None:
        signals = pd.read_csv(c.OUTPUT_DIR / 'fund_signals.csv', dtype={'fund_code': str})
    frames = [data.copy(), errors.copy(), signals.copy()]
    for frame in frames:
        frame['date'] = pd.to_datetime(frame.date)
    data = daily_error_comparison(*frames)
    c.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(c.OUTPUT_DIR / 'heavy_daily_error_comparison.csv', index=False, encoding='utf-8-sig')
    data.to_csv(c.OUTPUT_DIR / 'heavy_correlation_comparison.csv', index=False, encoding='utf-8-sig')
    draw_chart(data, c.OUTPUT_DIR / 'result.png')
    for year, annual in data.groupby(data.date.dt.year):
        if len(annual) > 0:
            draw_chart(annual, c.OUTPUT_DIR / f'result_{year}.png')


if __name__ == '__main__':
    plot_result(pd.read_csv(c.OUTPUT_DIR / 'aggregate_signal.csv', parse_dates=['date']))
    print(f'图表已保存至 {c.OUTPUT_DIR}')
