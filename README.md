# 初赛：人工参考集上的 RGB 预测与评估

本分支保留初赛 1827 道有人工 bbox 的题目所用流程。数据和 LocateAnything-3B
基座模型需自行准备，不提交到仓库。脚本支持显式传入路径。

## RTX 6000D 环境

建议 Linux、Python 3.10 或 3.11，在仓库目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; assert torch.cuda.is_available(); x=torch.ones((16,16),device='cuda',dtype=torch.bfloat16); assert float((x@x)[0,0])==16; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"
```

## 预测与算分

下面示例假定数据位于仓库相邻的 `datasets/reference_subset`，模型位于相邻的
`LocateAnything-3B`。`queries.json` 无 bbox，`annotations.json` 是人工参考答案。

```bash
python predict_rgb_all.py \
  --annotation ../datasets/reference_subset/queries.json \
  --data-root ../datasets/reference_subset \
  --model-path ../LocateAnything-3B \
  --output-dir ../outputs/rgb_reference \
  --image-token-limit 25600

python evaluate.py \
  --predictions ../outputs/rgb_reference/queries_rgb.json \
  --references ../datasets/reference_subset/annotations.json
```

预测逐条保存，可用相同命令续跑。`summary.json` 的 `complete: true` 才代表
本组题目全部生成有效框；`failures.json` 记录需要重试或人工检查的题目。
评估按 ID 对齐，输出平均 IoU、Acc@0.5 等指标和逐题明细。

复现最初恢复的初赛脚本参数时可运行：

```bash
python seed_sweep.py --original-rgb --seeds 42 43 \
  --annotation ../datasets/reference_subset/queries.json \
  --references ../datasets/reference_subset/annotations.json \
  --data-root ../datasets/reference_subset \
  --full-queries ../datasets/full/queries.json \
  --model-path ../LocateAnything-3B \
  --output-dir ../outputs/seed_sweep_original_rgb
```

该模式使用 25600 图像预算、temperature 0.7、top-p 0.9，并按全量初赛题目的
原始顺序计算随机种子。统计文件仅保留各轮和跨轮指标，不保留逐题预测。

测试：`python -m unittest discover -p 'test_*.py'`。

## 序数子集与官方多框拼图实验

此分支新增独立实验，不修改 `predict_rgb_all.py` 和 `evaluate.py` 的原有行为。
从仓库目录执行；以下路径对应本地 `D:\AICamp` 布局，Linux 上可换成自己的绝对路径。

先筛选**恰好一个英文序数词和一个明确的左右方向**的有答案题；
同时含上下排序方向的复合题排除。当前参考集筛出 192/1827 题，112 张 RGB 图。
输出只有 JSON，图片仍从 `../datasets/reference_subset` 读取。

```bash
python prepare_ordinal_subset.py
```

检查 `../datasets/ordinal_subset/parsed_ordinals.json` 中自动提取的 `target`。
该文件供人工检查；`queries.json` 不含答案，`annotations.json` 含人工 bbox。

复用原有 A 基线预测和算分脚本，先做仅检查输入，再预测并评估：

```bash
python predict_rgb_all.py --annotation ../datasets/ordinal_subset/queries.json --data-root ../datasets/reference_subset --model-path ../LocateAnything-3B --output-dir ../outputs/ordinal_a --check-only
python predict_rgb_all.py --annotation ../datasets/ordinal_subset/queries.json --data-root ../datasets/reference_subset --model-path ../LocateAnything-3B --output-dir ../outputs/ordinal_a --image-token-limit 4096
python evaluate.py --predictions ../outputs/ordinal_a/queries_rgb.json --references ../datasets/ordinal_subset/annotations.json --output-dir ../outputs/ordinal_a/evaluation
```

RTX 5060 上可先用独立目录 `--output-dir ../outputs/ordinal_a_pilot --limit 3`
并加 `--image-token-limit 1024` 做小规模逻辑测试。改变图像预算后应换输出目录；4096 与此前 25600 预算的实验
不可直接比较分数。8GB 显存可能仍不足以完整加载 BF16 基座和缓存。

F 使用官方多实例提示 `Locate all the instances that match the following description: ...`，
解析返回的**全部** bbox；不再枚举单框提示。其流程是：去除含至少两个独立子框
的大框、IoU 去重、按原图水平位置排序、裁剪并等高拼接、再用单实例提示在拼图
中选择。`--check-only` 不加载模型。先试跑，再在独立目录跑整个序数子集：

```bash
python experiment_f_official_multi.py --check-only
python experiment_f_official_multi.py --limit 3 --image-token-limit 1024 --output-dir ../outputs/ordinal_f_pilot
python experiment_f_official_multi.py --output-dir ../outputs/ordinal_f_official_multi --baseline-predictions ../outputs/ordinal_a/queries_rgb.json
```

本地试跑可加 `--image-token-limit 1024` 降低图像预算；正式对比 A/F 时务必让
两者使用相同预算。F 的 `--multi-max-new-tokens` 默认为 2048，可调高以避免多框
输出截断。其 `predictions.jsonl` 保留原始模型回答、全部候选框及各过滤阶段；
`puzzles/` 中绿色边框表示与 GT 匹配的候选，红色表示所选候选；
`per_query.csv` 和 `summary.json` 给出 F IoU、Acc@0.5、各阶段 GT 覆盖以及可选的
A/F 对错交叉统计。`queries_f.json` 仅收录成功预测的框，可再用 `evaluate.py`
独立核对分数。所有失败和缺失均按参考题总数计 0，不会偷用 GT 选择候选。
若只需对照已有的全量 A 预测，可把 `--baseline-predictions` 指向原来的
`../outputs/rgb_all/queries_rgb.json`；记录并比较两边实际使用的图像预算。

## 递归细分大框（独立实验）

`experiment_f_recursive_multi.py` 保留上面的 F 脚本不变。它先记录原始 F
选择；当多框结果出现疑似大区域时，在该区域内再次使用官方多实例提示，
最多细分 3 层。每层只接受面积至少缩小 20% 的子框，父框留作原始分支备用，
不会和子框一起放入计数拼图。首次无候选框时直接采用 A 框。
触发规则：首层候选框至少占整图 10%；多框时还须是第二大框的
至少 1.75 倍。例外是候选框至少占整图 2%、覆盖 A 框至少 90%，
且面积至少为 A 框 3 倍。递归后续层只继续细分仍占原图至少 10% 的框，
避免小框在裁剪图中显得巨大。阈值依据第一轮 192 题的逐题记录收紧：
4 道被细分改错的框均占整图约 3%–5%。新版在 192 道初赛有答案序数题上
实测 79/192 Acc@0.5，细分新增 2 道正确题、未改错原本正确的题。

```bash
python experiment_f_recursive_multi.py --baseline-predictions ../outputs/rgb_all/queries_rgb.json --check-only
python experiment_f_recursive_multi.py --baseline-predictions ../outputs/rgb_all/queries_rgb.json --ids 000023_001 002760_002 002683_001 --output-dir ../outputs/ordinal_f_recursive_pilot
python experiment_f_recursive_multi.py --baseline-predictions ../outputs/rgb_all/queries_rgb.json --output-dir ../outputs/ordinal_f_recursive_full
```

在其它机器上给 `--queries`、`--references`、`--data-root`、`--model-path`
传入实际路径。每次运行用独立 `--output-dir`；相同命令可以续跑。
输出 `summary.json` 同时报告原始 F、递归后最终预测与 A 的 IoU/Acc@0.5，
以及新找回/丢失的正确候选数；`predictions.jsonl` 保存每层提示、原始回答、
映射回原图的子框。`puzzles_initial/` 和 `puzzles_refined/` 分别保存选择拼图。

## 本地查看 A/F 同错题的各阶段图

先把云端递归实验的 `predictions.jsonl` 和对应的**完整 RGB 原图**下载到本地。
当前工作区原有的部分 RGB 文件被截断，因此本次使用
`../analysis/ordinal_recursive_gate_v2_20260924/rgb_complete` 中补齐的 30 张图。
以下命令只读取日志、人工答案和 RGB 图像，不加载模型或调用 GPU。
默认筛选 A 与最终 F 都错，
但 F 原始多框属于以下三类的 45 题：17 题已有准确候选、19 题大框覆盖答案、
9 题仅部分覆盖或偏移。路径可用参数替换。

```bash
python visualize_ordinal_stages.py --list-only
python visualize_ordinal_stages.py --journal ../analysis/ordinal_recursive_gate_v2_20260924/predictions.jsonl --output-dir ../outputs/ordinal_stage_review_45
python visualize_ordinal_stages.py --ids 003810_001 002760_004 --output-dir ../outputs/ordinal_stage_review_examples
```

如果完整图像保存在其它目录，传入 `--data-root`。脚本发现图片无法完整解码时
会报错，不会用截断图片制作核查图。

打开输出目录的 `index.html`，可按三类浏览。每题分别生成无标注 RGB 原图、
GT/A/最终 F 对照图、第一阶段原始框、去父框后的候选、去重后的候选、
重建的初始拼图；发生细分时还生成每层
裁剪图、映射回原图的子框、细分后候选和拼图。绿色表示人工答案或 IoU≥0.5
的拼图候选，红色表示模型选择，紫色表示最终框或拼图输出。页面还保留模型原文，
`manifest.csv` 汇总每题文件数和 IoU。人工答案只用于标记图片，不参与预测。

## 带上下文的 F 与按序号直选的 G

新版 `experiment_f_recursive_multi.py` 给拼图中的每个候选裁块增加默认每侧
2.5% 的边缘上下文（长宽总共扩大到原框的 1.05 倍），但选中后仍返回
**原始候选框**，不会把扩大的裁块当作答案框。
`--puzzle-context-padding` 可调整此比例。F 若在拼图上给出多个框，或单框明显
横跨多个拼图块，会用“只选一个黄色边框内的块”的提示重试一次；重试仍无法
确定单块时，按 G 规则直接选对应序号。拼图无候选或候选数不足以数到目标序号时
回退到 A。G 按候选在原图中的横坐标从左到右排序，右往左计数时从最后一块开始。

G 可直接使用已经保存的 F 日志评分，无需图像、模型或 GPU：

```bash
python evaluate_g_ordinal.py --candidate-stage initial --output-dir ../outputs/ordinal_g_replay_initial
python evaluate_g_ordinal.py --candidate-stage refined --output-dir ../outputs/ordinal_g_replay_refined
```

这两个命令使用旧版 192 题日志回放，分别得到 95/192 与 97/192 的 Acc@0.5；
这是**旧候选框的 G 分数**，不是新上下文拼图重新推理后的 F 分数。可通过
`--journal`、`--references`、`--baseline-predictions` 指定其他机器上的路径。
F 新版运行时请使用新的 `--output-dir`；完整 RGB 图像路径通过 `--data-root` 指定。
输出会记录首次拼图回答、重试回答、是否跨块，以及最终由 F、G 还是 A 决策。
本地小规模 GPU 试跑示例：

```bash
python experiment_f_recursive_multi.py --ids 003810_001 002760_004 --data-root ../analysis/ordinal_recursive_gate_v2_20260924/rgb_complete --model-path ../LocateAnything-3B --baseline-predictions ../outputs/rgb_all/queries_rgb.json --output-dir ../outputs/ordinal_fg_context_pilot
```

可用 `visualize_ordinal_stages.py --journal <新日志> --puzzle-context-padding 0.025`
重建新版拼图；这个参数须与推理时一致。
