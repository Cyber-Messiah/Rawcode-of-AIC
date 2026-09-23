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
