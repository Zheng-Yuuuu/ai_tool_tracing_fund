"""Free Eastmoney data, with CSV caching and explicit failures."""
import json
import re
import time
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import requests
from bs4 import BeautifulSoup
import config as c


def end_date():
    return c.END_DATE or datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')


def read_inputs(limit=None):
    funds = pd.read_csv(c.DATA_DIR / 'fund_list.csv', dtype=str).head(limit)
    pool = pd.read_csv(c.DATA_DIR / 'ai_hardware.csv', dtype=str)
    for frame, key in [(funds, 'fund_code'), (pool, 'stock_code')]:
        if frame.empty or frame[key].isna().any() or not frame[key].str.fullmatch(r'\d{6}').all():
            raise ValueError(f'{key}: 必须提供非空、六位字符串代码')
        if frame[key].duplicated().any():
            raise ValueError(f'{key}: 名单包含重复代码')
    return funds, pool


def get(url, params=None, **kwargs):
    response = requests.get(url, params=params, timeout=c.REQUEST_TIMEOUT, **kwargs)
    response.raise_for_status()
    return response


def attempt(primary, fallback):
    for n in range(c.RETRIES):
        try:
            result = primary() if n == 0 else fallback()
            if result.empty:
                raise ValueError('接口返回空数据')
            return result
        except Exception as exc:
            print(f'  请求失败 ({n + 1}/{c.RETRIES}): {type(exc).__name__}: {str(exc)[:240]}', flush=True)
            if n + 1 < c.RETRIES:
                time.sleep(1)
    raise RuntimeError('AkShare / 东方财富接口均未成功；已有缓存保留')


def cached(filename, keys, dates, selected, column, fetch, refresh):
    path = c.CACHE_DIR / filename
    c.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path, dtype={'fund_code': str, 'stock_code': str}, parse_dates=dates) if path.exists() else pd.DataFrame()
    result = old.copy()
    available = set(old[column]) if not old.empty else set()
    hits = sum(code in available for code, _ in selected) if not refresh else 0
    print(f'{filename}: 缓存命中 {hits}/{len(selected)}', flush=True)
    failures = []
    for code, name in selected:
        if not refresh and code in available:
            continue
        print(f'下载: {code} {name}', flush=True)
        try:
            fresh = fetch(code, name)
            if fresh.empty:
                raise ValueError('没有有效数据')
            result = pd.concat([result, fresh], ignore_index=True).drop_duplicates(keys, keep='last').sort_values(keys)
            temporary = path.with_suffix('.tmp')
            result.to_csv(temporary, index=False, encoding='utf-8-sig')
            temporary.replace(path)
        except Exception as exc:
            print(f'ERROR {code}: {exc}', flush=True)
            if result.empty or code not in set(result[column]):
                failures.append(code)
            else:
                print(f'WARNING {code}: 本次下载失败，使用旧缓存')
    if failures:
        raise RuntimeError(f'{filename} 缺少数据: {failures}；STOP')
    return result[result[column].isin([x[0] for x in selected])].copy()


def nav_direct(code):
    rows = []
    page = 1
    while True:
        data = get('https://api.fund.eastmoney.com/f10/lsjz', {'fundCode': code, 'pageIndex': page, 'pageSize': 20, 'startDate': '', 'endDate': end_date()}, headers={'Referer': f'https://fundf10.eastmoney.com/jjjz_{code}.html'}).json()
        batch = data['Data']['LSJZList']
        rows.extend(batch)
        if not batch or min(x['FSRQ'] for x in batch) < c.START_DATE or len(rows) >= int(data['TotalCount']):
            break
        page += 1
    return pd.DataFrame(rows).rename(columns={'FSRQ': '净值日期', 'DWJZ': '单位净值', 'JZZZL': '日增长率'})


def download_fund_nav(funds, refresh=False):
    def fetch(code, name):
        raw = attempt(lambda: ak.fund_open_fund_info_em(symbol=code, indicator='单位净值走势'), lambda: nav_direct(code))
        df = raw.rename(columns={'净值日期': 'date', '单位净值': 'nav'}).copy()
        df['date'] = pd.to_datetime(df['date'])
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        df = df.sort_values('date').drop_duplicates('date')
        if '日增长率' in df:
            # Missing published returns stay missing, never silently replace dividend returns.
            df['daily_return'] = pd.to_numeric(df['日增长率'].astype(str).str.rstrip('%'), errors='coerce') / 100
        else:
            print('WARNING: 无日增长率，采用单位净值变化；分红/拆分可能影响结果')
            df['daily_return'] = df.nav.pct_change(fill_method=None)
        df['fund_code'], df['fund_name'] = code, name
        return df.loc[df.date.between(c.START_DATE, end_date()), ['date', 'fund_code', 'fund_name', 'nav', 'daily_return']]
    result = cached('fund_nav.csv', ['fund_code', 'date'], ['date'], list(funds[['fund_code', 'fund_name']].itertuples(index=False, name=None)), 'fund_code', fetch, refresh)
    result['fund_name'] = result.fund_code.map(funds.set_index('fund_code').fund_name)
    return result


def holdings_direct(code, year):
    text = get('https://fundf10.eastmoney.com/FundArchivesDatas.aspx', {'type': 'jjcc', 'code': code, 'topline': 10, 'year': year, 'month': ''}).text
    match = re.search(r'content:\s*("(?:[^"\\]|\\.)*")', text)
    if not match:
        raise ValueError('无法解析持仓 HTML')
    soup = BeautifulSoup(json.loads(match.group(1)), 'html.parser')
    frames = []
    for box in soup.select('.box'):
        table, title = box.find('table'), box.find('h4')
        if table is not None and title is not None:
            frame = pd.read_html(StringIO(str(table)), converters={'股票代码': str})[0]
            frame.columns = [str(x).replace(' ', '') for x in frame.columns]
            frame['季度'] = title.get_text()
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def download_fund_holdings(funds, refresh=False):
    def fetch(code, name):
        frames = []
        # Include preceding year to cover January before the first requested report.
        for year in range(pd.Timestamp(c.START_DATE).year - 1, pd.Timestamp(end_date()).year + 1):
            raw = attempt(lambda: ak.fund_portfolio_hold_em(symbol=code, date=str(year)), lambda: holdings_direct(code, year))
            quarter = raw['季度'].str.extract(r'(\d{4})年\s*([1-4])季度')
            raw['report_date'] = [pd.Period(f'{y}Q{q}', freq='Q').end_time.normalize() if pd.notna(y) else pd.NaT for y, q in quarter.itertuples(index=False, name=None)]
            raw['weight'] = pd.to_numeric(raw['占净值比例'].astype(str).str.rstrip('%'), errors='coerce') / 100
            raw = raw.rename(columns={'股票代码': 'stock_code', '股票名称': 'stock_name'})
            raw['stock_code'] = raw.stock_code.astype(str).str.strip()
            raw = raw.dropna(subset=['report_date', 'weight']).sort_values('weight', ascending=False).groupby('report_date').head(10)
            raw['fund_code'] = code
            if not raw.weight.between(0, 1).all() or (raw.groupby('report_date').weight.sum() > 1.001).any():
                raise ValueError('持仓权重异常')
            frames.append(raw[['fund_code', 'report_date', 'stock_code', 'stock_name', 'weight']])
        return pd.concat(frames, ignore_index=True)
    result = cached('fund_holdings.csv', ['fund_code', 'report_date', 'stock_code'], ['report_date'], list(funds[['fund_code', 'fund_name']].itertuples(index=False, name=None)), 'fund_code', fetch, refresh)
    prior = result[result.report_date < pd.Timestamp(c.START_DATE)].groupby('fund_code').report_date.max()
    relevant = result.report_date.ge(pd.Timestamp(c.START_DATE)) | result.report_date.eq(result.fund_code.map(prior))
    return result[relevant & result.report_date.le(pd.Timestamp(end_date()))].copy()


def stock_direct(code):
    market = '1' if code.startswith('6') else '0'
    data = get('https://79.push2his.eastmoney.com/api/qt/stock/kline/get', {'secid': f'{market}.{code}', 'klt': 101, 'fqt': 1, 'beg': c.START_DATE.replace('-', ''), 'end': '20500101', 'lmt': 10000, 'ut': '7eea3edcaed734bea9cbfc24409ed989', 'fields1': 'f1,f2,f3,f4,f5,f6', 'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'}, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'}).json()
    rows = data.get('data') or {}
    return pd.DataFrame([x.split(',') for x in rows.get('klines', [])], columns=['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率'])


def download_stock_returns(holdings, pool, refresh=False):
    stocks = pd.concat([holdings[['stock_code', 'stock_name']], pool[['stock_code', 'stock_name']]]).drop_duplicates('stock_code')
    unsupported = stocks[~stocks.stock_code.str.fullmatch(r'\d{6}') | ~stocks.stock_code.str.startswith(('0', '3', '6'))]
    if not unsupported.empty:
        print(f'WARNING: 暂不支持港股/北交所等行情，排除并降低 covered_weight: {unsupported.stock_code.tolist()}')
    if pool.stock_code.isin(unsupported.stock_code).any():
        raise ValueError('AI 股票池包含暂不支持的股票代码')
    stocks = stocks[~stocks.stock_code.isin(unsupported.stock_code)]
    primary_available = True
    def fetch(code, name):
        def primary():
            nonlocal primary_available
            if not primary_available:
                return stock_direct(code)
            try:
                return ak.stock_zh_a_hist(symbol=code, period='daily', start_date=c.START_DATE.replace('-', ''), end_date=end_date().replace('-', ''), adjust='qfq', timeout=c.REQUEST_TIMEOUT)
            except Exception:
                primary_available = False
                print('WARNING: 本轮 AkShare 股票接口失效，后续股票使用东方财富直连接口')
                raise
        raw = attempt(primary, lambda: stock_direct(code))
        df = pd.DataFrame({'date': pd.to_datetime(raw['日期']), 'daily_return': pd.to_numeric(raw['涨跌幅'], errors='coerce') / 100})
        df['stock_code'], df['stock_name'] = code, name
        return df.loc[df.date.between(c.START_DATE, end_date()), ['date', 'stock_code', 'stock_name', 'daily_return']]
    return cached('stock_returns.csv', ['stock_code', 'date'], ['date'], list(stocks.itertuples(index=False, name=None)), 'stock_code', fetch, refresh)


def data_check(nav, holdings, stocks):
    print('\n========== DATA CHECK ==========')
    for title, df, code in [('Fund NAV', nav, 'fund_code'), ('Fund Holdings', holdings, 'fund_code'), ('Stock Returns', stocks, 'stock_code')]:
        print(f'{title}:\n{"Funds" if code == "fund_code" else "Stocks"}: {df[code].nunique()}\nRows: {len(df)}')
        if 'date' in df:
            print(f'Date Range: {df.date.min()} -> {df.date.max()}')
        else:
            print(f'Reports: {df.report_date.nunique()}')
        if df.empty:
            raise RuntimeError(f'{title} 为空；STOP')
    print('===============================')
