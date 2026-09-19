# 根据原本项目中学长写的md中提到的关键代码

| # | 路径 | 作用 |
|---|------|------|
| 1 | `code/tools/predict_locany_competition.py` | 比赛推理入口：读取 queries，生成带 bbox 的提交 JSON |
| 2 | `code/Embodied/eaglevl/model/locany/modeling_locateanything.py` | 基础模型定义，含 RGB‑T 融合模块与训练/生成接口 |
| 3 | `code/tools/prepare_locany_sft.py` | 数据转换：比赛 queries → LocateAnything 训练 JSONL |
| 4 | `code/Embodied/eaglevl/train/locany_finetune_magi_stream.py` | 训练入口：流式打包、MTP、冻结策略与 loss 计算 |
| 5 | `code/tools/evaluate_locany_fusion.py` | 有 GT 评测：IoU、ACC@0.5、可视化与错误样本输出 |
| 6 | `code/tools/run_locany_rgbt_cross_attention_stage.sh` | 训练启动脚本：环境变量、融合参数、分布式训练命令 |