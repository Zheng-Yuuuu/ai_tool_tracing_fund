import numpy as np
import pandas as pd
import config as c


def calculate(nav, holdings, stocks, pool, exposure_pool=None):
    nav = nav[nav.date.between(c.START_DATE, c.END_DATE or '2262-01-01')].copy()
    stocks = stocks[stocks.date.between(c.START_DATE, c.END_DATE or '2262-01-01')].copy()
    ai = stocks[stocks.stock_code.isin(pool.stock_code)].groupby('date').daily_return.mean().rename('ai_return').to_frame()
    if ai.empty or ai.ai_return.notna().sum() == 0:
        raise ValueError('AI 股票池没有有效收益数据；STOP')
    ai['ai_index'] = 100 * (1 + ai.ai_return).cumprod()
    exposure_pool = pool if exposure_pool is None else exposure_pool
    exposure = holdings.assign(ai_weight=holdings.weight.where(holdings.stock_code.isin(exposure_pool.stock_code), 0)).groupby(['fund_code', 'report_date']).ai_weight.sum().rename('reported_ai_exposure').reset_index()
    daily = []
    for code, fund in nav.groupby('fund_code'):
        reports = exposure[exposure.fund_code == code].drop(columns='fund_code').sort_values('report_date')
        # TODO:
        # Current MVP uses report_date as holdings availability date.
        # This creates look-ahead bias.
        # Production version must use actual fund report publication date.
        aligned = pd.merge_asof(fund.sort_values('date'), reports, left_on='date', right_on='report_date', direction='backward')
        positions = aligned[['date', 'fund_code', 'report_date']].merge(holdings, on=['fund_code', 'report_date'], how='left').merge(stocks[['date', 'stock_code', 'daily_return']], on=['date', 'stock_code'], how='left')
        positions['contribution'] = positions.weight * positions.daily_return
        positions['covered_weight'] = positions.weight.where(positions.daily_return.notna(), 0)
        estimated = positions.groupby('date').agg(estimated_return=('contribution', lambda x: x.sum(min_count=1)), covered_weight=('covered_weight', 'sum'))
        aligned = aligned.merge(estimated, on='date', how='left').rename(columns={'daily_return': 'actual_return'})
        aligned['error'] = aligned.estimated_return - aligned.actual_return
        daily.append(aligned)
    daily = pd.concat(daily, ignore_index=True).sort_values(['fund_code', 'date'])
    # Strictly greater than the configured threshold; tolerate floating-point summation at the boundary.
    heavy = daily.reported_ai_exposure.gt(c.AI_HEAVY_THRESHOLD) & ~np.isclose(daily.reported_ai_exposure, c.AI_HEAVY_THRESHOLD, rtol=0, atol=1e-12)
    daily['group'] = np.select([heavy, daily.reported_ai_exposure <= c.CONTROL_THRESHOLD], ['AI Heavy', 'Control'], default='Other')
    signals = daily.merge(ai[['ai_return']], on='date', how='left')
    for window in [10, 20, 30]:
        signals[f'corr_{window}d'] = np.nan
        for _, fund in signals.groupby('fund_code'):
            signals.loc[fund.index, f'corr_{window}d'] = fund.error.rolling(window, min_periods=window).corr(fund.ai_return).clip(-1, 1).to_numpy()
    aggregate = ai.copy()
    for group, prefix in [('AI Heavy', 'heavy'), ('Control', 'control')]:
        subset = signals[signals.group == group].groupby('date')
        aggregate[f'{prefix}_fund_count'] = subset.fund_code.nunique().reindex(aggregate.index, fill_value=0)
        aggregate[f'{prefix}_median_corr'] = subset.corr_20d.median().reindex(aggregate.index)
        aggregate[f'{prefix}_positioning'] = -aggregate[f'{prefix}_median_corr']
    aggregate['diff_signal'] = aggregate.heavy_positioning - aggregate.control_positioning
    errors = daily[['date', 'fund_code', 'fund_name', 'actual_return', 'estimated_return', 'error', 'report_date', 'covered_weight']]
    signals = signals[['date', 'fund_code', 'fund_name', 'reported_ai_exposure', 'group', 'corr_10d', 'corr_20d', 'corr_30d']]
    aggregate = aggregate.reset_index()[['date', 'ai_return', 'ai_index', 'heavy_fund_count', 'control_fund_count', 'heavy_median_corr', 'control_median_corr', 'heavy_positioning', 'control_positioning', 'diff_signal']]
    return errors, signals, aggregate
