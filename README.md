# 基于 PANNs 的日常环境声音检测演示系统

这是一个轻量级演示项目，基于 `panns-inference` 中的 PANNs `Cnn14` 预训练模型，实现：

- 单文件命令行推理
- Gradio Web 演示界面
- ESC-50 测试集零样本批量评估
- 长音频自动切片聚合

## 目录结构

```text
.
├── model_loader.py
├── audio_processor.py
├── inference.py
├── web_demo.py
├── evaluator.py
├── predict.py
├── evaluate.py
├── download_esc50.py
├── requirements.txt
├── run_demo.bat
└── run_demo.sh
```

## 环境要求

- Python 3.8+
- 推荐首次运行时联网，以自动下载：
  - PANNs `Cnn14` 权重
  - AudioSet 类别映射
  - ESC-50 数据集

## 安装依赖

```bash
pip install -r requirements.txt
```

## 1. 命令行预测

```bash
python predict.py --audio path/to/example.wav
```

可选参数：

- `--checkpoint`: 指定本地 PANNs 权重文件
- `--device cpu|cuda`: 指定推理设备
- `--top_k 5`: 输出前 K 个类别

示例输出：

```text
Loaded PANNs model (CPU) in 0.80s.
Processing audio: dog_bark.wav (2.30s, 32000Hz mono, windows=1)
Top-5 predictions:
  1. Bark (0.9200)
  2. Whimper (0.0500)
  3. Howl (0.0200)
  4. Dog (0.0100)
  5. Animal (0.0000)
Inference time: 340.00 ms
```

## 2. Web 演示模式

```bash
python web_demo.py
```

启动后浏览器打开本地地址，例如：

```text
http://127.0.0.1:7860
```

界面支持：

- 上传 `.wav` / `.mp3` / `.flac` 音频
- 直接使用麦克风录音
- 展示 Top-5 结果表格和柱状图
- 显示推理耗时、切片窗口数和设备信息

Windows 下一键启动：

```bat
run_demo.bat
```

Linux / macOS 下一键启动：

```bash
bash run_demo.sh
```

## 3. ESC-50 下载与评估

下载数据集：

```bash
python download_esc50.py --output_dir .
```

评估 fold 1：

```bash
python evaluate.py --data_root ./ESC-50-master --test_fold 1
```

也支持通过 `predict.py --eval` 触发：

```bash
python predict.py --eval --data_root ./ESC-50-master --test_fold 1
```

可选参数：

- `--subset_ratio 0.5`: 只评估一半测试集样本，适合快速演示
- `--output confusion_matrix.png`: 指定混淆矩阵输出路径

## 设计说明

- 音频统一重采样到 `32000 Hz`、单声道
- 对于超过 `10 秒` 的音频，自动按 `5 秒` 窗口、`2.5 秒` 步长切片
- 多窗口结果默认取平均概率作为整段音频的最终输出
- 当音频短于 `0.5 秒` 时会直接报错提示
- 默认使用 GPU；若不可用则自动回退到 CPU
- ESC-50 官方数据集共有 `2000` 条音频，按 `5` 折划分时每个 fold 实际是 `400` 条样本；如果只想演示约 `200` 条样本，可使用 `--subset_ratio 0.5`

## 注意事项

- `panns-inference` 在不同平台上的底层依赖可能略有差异，若 `mp3` 读取失败，可先转成 `wav`
- ESC-50 评估使用的是零样本标签映射，结果会受到标签映射质量影响
- 首次下载模型和数据集时体积较大，请确保网络稳定
