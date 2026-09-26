# LocateAnything 复赛 RGB bbox 预测

`main` 用于复赛全量题目生成 bbox。数据集和 LocateAnything-3B 基座模型不在
此仓库内，由使用者通过命令行提供路径。初赛有人工参考答案的预测、算分和多种子
统计保存在 `preliminary-reference-eval` 分支。

## 环境：RTX 6000D

在 Linux、Python 3.10 或 3.11 环境中，从本仓库目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; assert torch.cuda.is_available(); x=torch.ones((16,16),device='cuda',dtype=torch.bfloat16); assert float((x@x)[0,0])==16; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"
```

最后一条会实际在显卡上计算，避免旧 PyTorch 虽能导入却无法运行新显卡。
`nvidia-smi` 应能识别 RTX 6000D；驱动不兼容时需要更新驱动。
推理使用 SDPA，无需安装 FlashAttention 或 MagiAttention。

## 数据格式

传入包含图片目录的 `--data-root`，例如：

```text
/path/to/final_dataset/
  Images/visible/...
  Images/infrared/...
  Images/depth/...
  queries/queries.json
```

也接受根目录下的 `queries.json`；其他位置可用 `--queries` 指定。
JSON 根节点是按题目 ID 索引的对象。每条至少包含 `visible` 和 `query`；
输出会保留原有字段（包括 `infrared`、`depth`），增加归一化
`bbox: [x1, y1, x2, y2]`。模型只读取 RGB，红外和深度不参与推理。

## 运行

以下命令在仓库目录执行。**请把占位路径换成实际位置**；数据目录和模型目录
不必在仓库内。先检查输入，再用独立目录做少量试跑，最后跑全量：

```bash
python predict_bbox.py \
  --data-root /path/to/final_dataset \
  --model-path /path/to/LocateAnything-3B \
  --output-dir /path/to/check_only \
  --check-only

python predict_bbox.py \
  --data-root /path/to/final_dataset \
  --model-path /path/to/LocateAnything-3B \
  --output-dir /path/to/pilot_output \
  --limit 10

python predict_bbox.py \
  --data-root /path/to/final_dataset \
  --model-path /path/to/LocateAnything-3B \
  --output-dir /path/to/final_output
```

默认使用初赛恢复脚本的 RGB 生成参数：图像 patch 预算 25600、hybrid 解码、
temperature 0.7、top-p 0.9、最多一次 temperature 0.2 的重试。
随机种子默认 42，并按原始题目顺序设定逐题种子。
如果大图显存不足，程序会在该题重试时降低图像预算；可用
`--image-token-limit` 明确设置全局预算。

每条结果同步保存到 `predictions.jsonl`，重启同一命令时跳过已成功题目，
重新处理失败题目。`predictions_valid.json` 只含有效模型框；`failures.json`
记录失败原因；`summary.json` 给出成功、失败及待处理数量。
只有 `summary.json` 中 `submission_ready: true` 时，才会写
`submission_complete.json`。此时它包含当前运行的全部题目，格式可用于提交。
正常全量运行还应检查 `total_queries` 与复赛题目数一致、`complete: true`、
`fallback_boxes: 0`。试跑的 `--limit 10` 应使用单独输出目录，以免误提交。

少量题目持续无法解析出 bbox 时，先查看 `failures.json`。如果必须生成
包含所有题目的文件，可在原命令后加 `--fallback-full-image` 并使用**相同输出目录**
重试失败项：最终仍失败的题目会使用 `[0,0,1,1]`，
数量记录在 `fallback_boxes`。这种框通常质量很差，应优先重试或检查原始回答。
改变模型、数据或生成参数时必须换输出目录，以免混入旧结果。
输出目录有 `run.lock` 时表示有进程占用；强制终止后确认旧进程已停止再移除锁。

测试：`python -m unittest discover -p 'test_*.py'`。

**v1 与 v2 共用坐标解析器** `bbox_coordinates.py`。旧版 v1 曾把模型输出的
倒序角点（例如 `<box><288><388><272><444></box>`）直接记为错误；现在两版
都会先规范角点顺序，再校验和输出归一化 `xyxy`。v1 的推理策略不变。
已经完成的旧版运行可在原输出目录用原命令续跑失败题；已有成功结果不会重跑，
原始运行文件应另行保留以便比较。`<box>None</box>` 仍表示模型没有给出框。

## v2：复赛全量 RGB，序数题使用 G

`predict_bbox_v2.py` 是独立运行入口；上面的 `predict_bbox.py`（v1）执行 A 策略。
v2 对每一道题先运行原单框 A 提示词。仅当题目含**唯一明确的水平序数**
（如 `third ... from left to right`）时，再运行官方多框提示词，去父框、去重，
对异常大框最多递归细分 3 次，按候选框中心的左右顺序直接选第 N 个，作为 G。
G 没有足够候选框或多框调用失败时回退 A。其他题保持 A。
v2 不需要人工答案，也不调用拼图重定位模型；因此这里的“G”指直接选取
经整理后的第 N 个候选框，和初赛实验中的 G 决策一致。

**坐标修复：**修复前的 v1 在解析时直接要求 `x2>x1`、`y2>y1`，还只接受整数。
现在两版把模型返回的四个值先作为两个角，分别取横纵坐标的最小值和最大值，
然后检查是否为非零面积的合法框。被交换的角会计入 `reversed_corners`，
不会直接记为失败。官方 `<box><x1><y1><x2><y2></box>` 坐标按 0–1000
量化单位换算；普通括号中的 0–1 小数保持原样；大于 1000 的普通坐标仅在
能依据图像尺寸解释为像素时换算。最终提交仍使用 0–1 的 `xyxy` 顺序。
零面积、负值和无法确定单位的框仍会被拒绝，详情在逐题记录的 `*_audit` 中。

在仓库目录执行（替换实际路径）：

```bash
# 只有 queries.json、暂时没有 RGB 图片时，可先检查题目路由
python predict_bbox_v2.py --data-root /path/to/final_dataset \
  --queries /path/to/queries.json --model-path /path/to/LocateAnything-3B \
  --output-dir /path/to/v2_check --check-only --skip-image-check

# GPU 小测：先专门选 5 道明确的水平序数题，核对 G 过程
python predict_bbox_v2.py --data-root /path/to/final_dataset \
  --model-path /path/to/LocateAnything-3B \
  --output-dir /path/to/v2_ordinal_pilot --only-ordinals --limit 5

# 复赛全量；不要加 --limit 或 --only-ordinals
python predict_bbox_v2.py --data-root /path/to/final_dataset \
  --model-path /path/to/LocateAnything-3B \
  --output-dir /path/to/v2_full
```

如果查询文件不在数据根目录的 `queries/queries.json` 或 `queries.json`，
以上后两条也要加 `--queries /path/to/queries.json`。RTX 6000D 环境安装
仍按本页开头的步骤；本地 5060 可做小测。实际运行前应检查 RGB 文件存在。

`summary.json` 中的 `ordinal_queries` 是路由到 G 的题数，`sources` 分别统计
`a`、`g_initial`、`g_refined`、`a_ordinal_fallback`；`reversed_corner_boxes`
统计被规范化的 A/首轮多框原始框数。`predictions.jsonl` 保留原始回答、
解析审计、候选框和递归步骤，方便核查坐标问题。只有
`submission_ready: true` 且 `total_queries` 等于完整题量时，才使用
`submission_complete.json`。小测输出目录与全量目录必须分开。
中断后可用同一命令和目录续跑；修改参数或脚本后换新目录。

## G′：复赛全量 RGB 实验版

`predict_bbox_gprime.py` 是一个完整的复赛推理入口。它先按 v2 产生 A 与初始 G；
然后对仍横跨原图至少 80% 宽度的序数候选框执行“单个实例”重定位及重叠横向裁图。
只有得到至少两个明显小于父框的有效子框、且候选数足以选择题目序号时，才替换大框。
同图同目标的序数题共享补搜：候选数不足最大序号时，换提示词搜索整图、上下文带和
未覆盖的横向区域，最后尝试 ±8% 亮度与略高温度。新框必须在至少两次独立调用中
出现并通过尺寸门槛。最终按左右位置直接选第 N 个；不足时回退该题原 A 框。
其余题仍使用 A。默认不启用 `--probe-complete`，以免已有足够候选的题因补搜而改序。

这套门控、提示词和默认参数取自初赛 G′ 的 `experiment_g_wide_refine.py` 与
`experiment_g_candidate_recall.py`。脚本不读取人工标注，也不产生“复赛准确率”；
初赛参考集的得分不能当作复赛得分。与 v2 一样，仅识别唯一明确的水平序数格式。

在仓库目录运行，路径按机器实际情况替换。先检查，再用不同输出目录小测，最后全量：

```bash
python predict_bbox_gprime.py \
  --data-root /path/to/final_dataset --queries /path/to/queries.json \
  --model-path /path/to/LocateAnything-3B \
  --output-dir /path/to/gprime_check --check-only

python predict_bbox_gprime.py \
  --data-root /path/to/final_dataset --queries /path/to/queries.json \
  --model-path /path/to/LocateAnything-3B \
  --output-dir /path/to/gprime_ordinal_pilot --only-ordinals --limit 5

python predict_bbox_gprime.py \
  --data-root /path/to/final_dataset --queries /path/to/queries.json \
  --model-path /path/to/LocateAnything-3B \
  --output-dir /path/to/gprime_full
```

输出目录中的 `predictions.jsonl` 是最终逐题日志；`_initial_g/` 是中间 A＋G 日志，
其中的提交文件只是中间结果。最终提交只使用**输出目录根层**的
`submission_complete.json`。`summary.json` 须显示 `submission_ready: true`、
`total_queries` 等于当前复赛 query 总数、`fallback_boxes: 0`。
脚本逐题保存结果，可用相同命令和输出目录续跑；改变参数、模型、数据或脚本版本时
另设输出目录。`--fallback-full-image` 只用于确有无法生成框的题，优先检查
`failures.json`。更多门控与模型调用细节保存在每题的
`wide_refinement_steps`、`recall` 和 `gprime_boxes` 中。
