# 当前核心代码

本目录仅保留 LocateAnything 基准模型的 RGB 推理流程，不含微调或多模态融合。

| 文件 | 作用 |
| --- | --- |
| predict_rgb_all.py | 唯一推理入口：模型调用、bbox 解析、全量循环、重试、续跑与结果导出 |
| requirements.txt | 固定运行依赖版本 |
| test_predict_rgb_all.py | 续跑、失败恢复、9555 条模拟任务及输入输出协议测试 |
| README.md | 安装、运行和输出说明 |

模型定义、处理器、分词器和权重位于项目根目录 LocateAnything-3B，仍是必需文件；图片及 queries 位于 LocateAnything_Competition。这里删除的是 code 中未使用的旧模型副本。

已移除：predict_locany_competition.py（两个必要辅助函数并入新入口）、modeling_locateanything.py（旧融合模型）、locany_finetune_magi_stream.py（训练）、prepare_locany_sft.py（训练数据转换）、evaluate_locany_fusion.py（有标注的融合评测）、run_locany_rgbt_cross_attention_stage.sh（旧训练启动器）。原有 Git 历史保留，今后需要恢复训练时可查阅。

提示词、生成参数、输出路径及进度格式保持兼容；同步新版到服务器后仍可用原命令续跑。此清理本身未在远程 GPU 重跑模型。成功输出有效 bbox 说明调用链已跑通；全量完成以 summary.json 为准，准确率还需要人工或真实标注评估。

# RGB 全量 bbox 生成

## Linux 服务器安装（Python 3.10 / 3.11）

日志中 PyTorch 2.1.2 被 Transformers 禁用属于版本不兼容，不是没有安装 torch。
将本目录的 requirements.txt 一起上传到服务器。建议使用独立环境，避免预装的
torchaudio、flash-attn 等与升级后的 torch 冲突：

```sh
cd /root/autodl-tmp/code/Rawcode-of-AIC
python -m venv /root/autodl-tmp/venvs/locate-rgb
source /root/autodl-tmp/venvs/locate-rgb/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, torchvision, transformers, peft, cv2, decord, lmdb; from transformers import AutoModel; print('torch:', torch.__version__, 'transformers:', transformers.__version__, 'CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python predict_rgb_all.py --limit 10
python predict_rgb_all.py
```

这里沿用报错环境的 cu118 路线；适用于兼容 CUDA 11.8 的服务器 GPU，
不适用于本地 RTX 50 系显卡。若服务器也是新架构 GPU，需要另选支持该架构的
PyTorch/CUDA 组合并同步调整 torch/torchvision 版本。每次重新登录服务器后，
先执行上面的 source 激活环境。依赖版本依据模型安装说明及 PyTorch 官方配对，
尚未在用户的远程 GPU 环境实际安装验证。

参考：https://huggingface.co/nvidia/LocateAnything-3B
及 https://pytorch.org/get-started/previous-versions/ 。

入口为独立的 `predict_rgb_all.py`，无需旧脚本。只读取 visible 图片；默认全部 queries（当前 9555 条），不需要融合权重或训练代码。

在具备 CUDA PyTorch 及模型依赖的 Python 环境中，从本目录执行：

```sh
python predict_rgb_all.py --check-only
python predict_rgb_all.py --limit 10
python predict_rgb_all.py
```

前两次推理与全量运行使用相同输出目录，因此已成功的 10 条会自动跳过。程序相对自身位置寻找项目内的数据集、模型和输出目录，不要求固定工作目录。服务器目录不同可显式指定：

```sh
python predict_rgb_all.py --annotation /path/to/queries/queries.json --data-root /path/to/dataset --model-path /path/to/LocateAnything-3B --output-dir /path/to/rgb_results
```

模型依赖参照模型目录 README（包括 CUDA PyTorch、transformers==4.57.1、torchvision、numpy、Pillow、peft、opencv-python-headless、decord、lmdb）。本入口使用 SDPA，不要求安装 MagiAttention；使用完整 BF16 模型，未实现量化或 CPU 卸载，8GB 显卡可能无法容纳模型和推理缓存。可直接在原 SSH GPU 服务器运行。当前机器尚未做真实模型运行验证。

默认图像 patch 预算 4096，降低显存消耗，但缩小图片可能影响小目标精度；可用 `--image-token-limit 25600` 恢复下载模型的预算，并使用新的输出目录。单条显存不足时降低预算重试，下一条恢复配置预算。首次使用 hybrid，解析失败时使用 slow 重试，默认最多 3 次。加载模型本身失败会立即终止，不会生成假的结果。

输出默认位于项目 `outputs/rgb_all/`：

- `queries_rgb.json`：仅包含成功生成 bbox 的样本，保留原字段，bbox 为 0～1 的 `[x1,y1,x2,y2]`。
- `predictions.jsonl`：每条处理后同步落盘，包含原始回答或错误。重启跳过成功项，重新处理失败项。
- `failures.json`：需要重试或人工检查的样本，不以整图框代替失败。
- `summary.json`：只有 `complete: true` 且 `successful: 9555` 才表示当前全量生成完成。成功解析不等于 bbox 语义正确。
- `run_config.json`：防止不同数据或推理参数混用输出。原地替换模型权重后请使用新的输出目录。

输出 JSON 每 50 条更新一次，正常结束及 Ctrl+C 时也更新。强制终止后重启会从逐条日志恢复，并修复末尾未写完整的一条。不得多个进程共用一个输出目录；强制结束留下 `run.lock` 时，确认旧进程已停止后删除该文件再运行。

有未解决失败时退出码为 2；重新运行同一命令重试。模型/环境异常需先解决。无需一次性将 9555 张输入加载到显存，按条执行，样本总数只影响总耗时和磁盘日志大小。

验证：`python -m unittest test_predict_rgb_all.py`。测试覆盖 9555 条模拟任务、失败后续跑、日志残行恢复、Ctrl+C 后保存和 bbox 解析；不替代真实 GPU 推理验证。

