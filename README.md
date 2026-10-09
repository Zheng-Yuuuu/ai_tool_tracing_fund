# 公募基金 AI 硬件隐含调仓监测 MVP

本仓库包含公募基金持仓估算实验，以及用已知仓位验证指标的合成基金实验。公开行情取自免费数据接口，首次运行需要联网；缓存和大批生成文件留在本机，不纳入版本控制。

Python 3.11+，AkShare/东方财富免费数据，CSV缓存，无UI。本机使用项目内Python 3.12虚拟环境。

## 运行

双击run.bat或执行 `.venv\Scripts\python.exe main.py`。默认缓存；`--refresh`刷新，`--limit 1`小样本验证，`--download-only`仅下载。首次安装使用 `pip install -r requirements.txt`。

运行合成实验：`.venv\Scripts\python.exe simulate_known_exposure.py`。首次运行会下载所需股票行情，输出 `output/synthetic_exposure_compare.png` 与对应 CSV。两种情景从40%升至80%，再分别降至40%和20%，绘制5日、10日、20日滚动相关系数。合成基金的其余仓位收益由固定随机种子产生，便于复现。仓位估算固定使用40%，这一设计用于检验相关系数能否反映已知的暴露变化。

## 当前筛选规则

以拟合通过为主，AI硬件相关股票占基金净资产的权重合计严格大于20%作为底线（只检查前十大，不归一化）。阈值在config.py。拟合通过要求beta>0、验证相关>=0.70、样本外R²>=0.50、每天行情覆盖前十大权重>=95%。这些是未经优化的研究阈值。

对每个历史日期，采用当时最近一期报告的前十大固定相对权重，使用该日期之前最近60个指数交易日：前40日拟合 `硬件日收益=alpha+beta*前十大组合日收益`，后20日验证。窗口可以早于持仓报告期，表示用当前已知持仓检验历史跟踪能力，不声称此前实际持有这些股票。只在拟合诊断中归一化前十大权重。拟合窗口截止前一交易日，数据不足或覆盖不足不入选。不使用最新结果回填历史名单。

样本外R²=1-验证预测误差平方和/相对于训练期指数收益均值的验证误差平方和。缺失的小比例股票收益贡献按0且不再次归一化，覆盖不足95%直接拒绝。拟合支持截距与尺度校准，不等于可原样复制指数；同向市场波动和持仓与指数重叠都可能提高拟合结果。

AI Heavy表示同时通过仓位底线与拟合检验的研究组；exposure_group记录仅仓位筛选结果。Control仍是仓位不超过5%的组，其他归Other，当前主图不使用对照组。

## 误差与蓝线

每日估算收益=披露前十大原始权重乘各股日收益后求和，权重不归一化；误差=估算收益-基金实际收益。港股等未支持行情会降低covered_weight，全部无行情时留空。

先计算当天入选研究组有效误差的算术平均，再与硬件板块日收益计算20日滚动Pearson相关，要求完整20对观测，空组不填0。组员变化也会影响蓝线。红线仍为原20股等权收益构成的累计指数，图示首日设为1。

## 股票名单

`data/fund_list.csv`是20只候选基金；`data/ai_hardware.csv`是原20只指数股票；`data/ai_hardware_classification.csv`是独立25只分类名单，原20只沿用项加东山精密、生益科技、立讯精密、三环集团、芯原股份，附新增依据和来源。`ai_hardware_classification_audit.csv`标记其他缓存证券未核实，不计入不等于确定不相关。多业务公司的全股票权重计入，不能解释为穿透AI营收比例。旧broad_semiconductor.csv仅作历史记录。

## 输出

- fund_daily_error.csv：原始估算误差与行情覆盖。
- fund_signals.csv：逐日分组、逐只相关系数、拟合结果和窗口截止日期。
- latest_fund_groups.csv、latest_hardware_fit.csv：最新分组和拟合结果，eligible_research_sample标记入选。
- heavy_correlation_comparison.csv：组内平均误差、有效基金数及heavy_error_corr_20d。
- result.png、result_2025.png、result_2026.png：全期及年度图。
- aggregate_signal.csv保留旧中位数Positioning/Diff供比较，不作为当前蓝线。

只重绘可运行plot_result.py；更改分类或阈值应运行main.py重新筛选。

## 局限与验证

TODO: Current MVP uses report_date as holdings availability date. This creates look-ahead bias. Production version must use actual fund report publication date.

当前按报告期末当作可用日期，且静态分类包括后来披露的信息，因此不是无前视偏差回测。筛选使用过去行情不会消除上述偏差。前十大以外仓位被忽略，相关系数只能作为研究代理指标，不能直接证明加减仓。缓存截至2026-09-30，不代表实时行情。

运行 `.venv\Scripts\python.exe -m unittest test_calculate test_experiment` 检查误差方向、阈值、先平均再相关、训练验证隔离、数据覆盖和筛选不使用当天或未来收益。
