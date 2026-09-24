# 序数题的 G′ 候选改进

这组脚本在初赛**有人工 bbox 的序数子集**上做 RGB 实验。输入是
`experiment_f_recursive_multi.py` 已保存的 `predictions.jsonl`：其中每题的
`refined_boxes` 是 G 用来按横坐标计数的候选。模型与 RGB 数据留在运行机器上，
不在仓库内；所有路径都可通过参数指定。推理脚本不读取人工答案，答案仅在
`evaluate_g_ordinal.py` 算分时使用。

流程有两步，彼此独立并保留原 F 日志：

1. `experiment_g_wide_refine.py` 对细分后仍占原图宽度至少 80% 的框，再用
   “每个实例单独一个框”的提示词与重叠横向裁图细分。只有至少两个有效子框、
   每个子框明显比父框窄小且能数到题目序号时才替换父框。
2. `experiment_g_candidate_recall.py` 处理同图同目标候选数小于最大序数的组。
   它依次尝试整图不同提示、目标附近的上下文带、已找到框左右的未覆盖区域，
   最后尝试轻微亮度和温度变化。新候选至少需被两次调用找到，且通过尺寸限制；
   不满足条件的提案只写入日志。过宽或面积过大的旧候选暂不进入此补搜步骤。

两步只修改新日志中的候选池，不覆写原 F 输出框。最后仍由
`evaluate_g_ordinal.py` 按位置执行 G；候选数不足时它回退 A。补搜的数量门槛
不等于类别验证，应检查逐题得失，不要只看总 ACC。

在仓库目录下运行；将这些路径换成当前机器的实际路径。三个输出目录必须不同，
重复执行相同命令可以从日志续跑；改变参数时请另设输出目录。

```bash
F_JOURNAL=/path/to/f/predictions.jsonl
DATA_ROOT=/path/to/reference_subset
MODEL=/path/to/LocateAnything-3B
REFERENCES=/path/to/ordinal_subset/annotations.json
A_PREDICTIONS=/path/to/a/queries_rgb.json
OUT=/path/to/experiment_outputs

python experiment_g_wide_refine.py \
  --journal "$F_JOURNAL" --data-root "$DATA_ROOT" --model-path "$MODEL" \
  --output-dir "$OUT/wide" --check-only

python experiment_g_wide_refine.py \
  --journal "$F_JOURNAL" --data-root "$DATA_ROOT" --model-path "$MODEL" \
  --output-dir "$OUT/wide"

python experiment_g_candidate_recall.py \
  --journal "$OUT/wide/predictions.jsonl" \
  --data-root "$DATA_ROOT" --model-path "$MODEL" \
  --output-dir "$OUT/recall" --check-only

python experiment_g_candidate_recall.py \
  --journal "$OUT/wide/predictions.jsonl" \
  --data-root "$DATA_ROOT" --model-path "$MODEL" \
  --output-dir "$OUT/recall"
```

评估须让三个阶段使用**完全相同的题目、GT、A 预测和 IoU 定义**：

```bash
python evaluate_g_ordinal.py \
  --journal "$F_JOURNAL" --candidate-stage refined \
  --references "$REFERENCES" --baseline-predictions "$A_PREDICTIONS" \
  --output-dir "$OUT/g_original"

python evaluate_g_ordinal.py \
  --journal "$OUT/wide/predictions.jsonl" --candidate-stage refined \
  --references "$REFERENCES" --baseline-predictions "$A_PREDICTIONS" \
  --output-dir "$OUT/g_wide"

python evaluate_g_ordinal.py \
  --journal "$OUT/recall/predictions.jsonl" --candidate-stage refined \
  --references "$REFERENCES" --baseline-predictions "$A_PREDICTIONS" \
  --output-dir "$OUT/g_prime"
```

各评估目录的 `summary.json` 有平均 IoU 与 Acc@0.5，`per_query.csv` 有逐题分数。
大框脚本的 `summary.json` 记录触发、接受、调用与错误数；补搜的
`gate_audit.json` 记录各组门控，`predictions.jsonl` 中的 `recall.attempts`
和 `recall.proposals` 保留每次模型输出与未接纳的框。两段脚本均支持
`--ids ...` 小范围试验；补搜脚本会更新指定 ID 所在的整个同图同目标组。
`--probe-complete` 可对候选数刚好够的组额外重查，默认关闭，因为假候选
会改变 G 的序数排序。

要浏览**原 G 与 G′ 输出框发生变化**的题目：

```bash
python visualize_g_comparison.py \
  --old-eval-dir "$OUT/g_original" \
  --middle-eval-dir "$OUT/g_wide" \
  --new-eval-dir "$OUT/g_prime" \
  --references "$REFERENCES" --baseline-predictions "$A_PREDICTIONS" \
  --data-root "$DATA_ROOT" --output-dir "$OUT/g_comparison"
```

打开 `g_comparison/index.html`。页面只列输出框变化的题，分别展示 A、原 G、
G′ 和独立 GT 面板；下排固定放大 GT 附近，便于核查小目标和标注边界。
脚本会完整解码 RGB，图像缺失或截断时直接报错，不会静默生成误导图片。
`--middle-eval-dir` 可省略，此时只标明原 G 与 G′ 的差别。

测试：`python -m unittest discover -p 'test_*.py'`。
