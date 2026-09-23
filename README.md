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
