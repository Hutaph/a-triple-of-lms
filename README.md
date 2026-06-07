# A-Triple-of-LMs

Benchmark 3 họ model LLM (Phi-3, Llama 4, Qwen3) trên bộ câu hỏi Big Data nhỏ, sau đó chấm điểm các câu trả lời thô bằng LLM judge. Mục tiêu là so sánh chất lượng output, chi phí, độ trễ, và hiệu quả của các model trên các tác vụ liên quan đến Big Data / Spark.

## Model được benchmark

| Alias | Model hiển thị | Model ID OpenRouter |
| :--- | :--- | :--- |
| `scout` | Llama 4 Scout | `meta-llama/llama-4-scout` |
| `maverick` | Llama 4 Maverick | `meta-llama/llama-4-maverick` |
| `qwen3-14b` | Qwen3-14B | `qwen/qwen3-14b` |
| `qwen3-32b` | Qwen3-32B | `qwen/qwen3-32b` |
| `qwen3-235b` | Qwen3-235B-A22B-Thinking | `qwen/qwen3-235b-a22b-thinking-2507` |

Mỗi alias có alias viết liền tương đương (ví dụ: `llama4-scout`, `llama4_scout` đều trỏ về `scout`).

## Yêu cầu hệ thống

- Python 3.10+
- Hệ điều hành: Windows, macOS, Linux (WSL)
- Kết nối internet để gọi OpenRouter API

## Cài đặt

Clone repo và tạo virtual environment (khuyến nghị):

```powershell
git clone https://github.com/Hutaph/a-triple-of-lms.git
cd a-triple-of-lms
```

Tạo venv (Linux / macOS / WSL):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Tạo venv (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài dependencies:

```bash
pip install -r requirements.txt
```

## Cấu hình

Tạo file `.env` ở root project từ `.env.example` và điền API key:

```bash
cp .env.example .env
```

Các biến môi trường cần thiết:

```env
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Các biến tùy chỉnh model ID:

```env
LLAMA4_SCOUT_MODEL=meta-llama/llama-4-scout
LLAMA4_MAVERICK_MODEL=meta-llama/llama-4-maverick
QWEN3_4B_MODEL=qwen/qwen3-4b
QWEN3_14B_MODEL=qwen/qwen3-14b
QWEN3_32B_MODEL=qwen/qwen3-32b
QWEN3_235B_A22B_THINKING_MODEL=qwen/qwen3-235b-a22b-thinking-2507
```

## Cấu trúc project

```text
a-triple-of-lms/
├── src/
│   ├── openrouter_benchmark.py   # Chạy benchmark các model qua OpenRouter
│   ├── llm_judge.py              # LLM judge chấm điểm output thô
│   └── phi3.py                   # Hỗ trợ chạy Phi-3 cục bộ (ONNX/CUDA)
├── data/
│   └── bigdata_10_questions.json # Bộ câu hỏi benchmark Big Data
├── outputs/
│   ├── llama4/                   # Output Llama 4 (scout, maverick)
│   ├── qwen3/                    # Output Qwen3 (14B, 32B, 235B)
│   ├── phi3/                     # Output Phi-3 (Mini, Medium)
│   └── llm_judge/                # Kết quả judge, bảng, biểu đồ
├── notebooks/
│   ├── llm_judge_evaluation.ipynb # Notebook chạy judge + dashboard
│   ├── Phi3_demo.ipynb
│   └── Phi3_colab_outputs.ipynb
├── requirements.txt
├── README.md
```

## Chạy benchmark

Chạy từng model:

```bash
# Linux / macOS / WSL
python3 src/openrouter_benchmark.py --model scout
python3 src/openrouter_benchmark.py --model maverick
python3 src/openrouter_benchmark.py --model qwen3-14b
python3 src/openrouter_benchmark.py --model qwen3-32b
python3 src/openrouter_benchmark.py --model qwen3-235b
```

```powershell
# Windows PowerShell
python src\openrouter_benchmark.py --model scout
python src\openrouter_benchmark.py --model maverick
python src\openrouter_benchmark.py --model qwen3-14b
python src\openrouter_benchmark.py --model qwen3-32b
python src\openrouter_benchmark.py --model qwen3-235b
```

Kiểm tra danh sách model khả dụng:

```bash
python src/openrouter_benchmark.py --list-models
```

Smoke test:

```bash
python src/openrouter_benchmark.py --model qwen3-14b --limit 1 --sleep 0
```

Giới hạn số câu hỏi:

```bash
python src/openrouter_benchmark.py --model scout --limit 5
```

Output được ghi vào:

```text
outputs/llama4/llama4_scout_outputs.json
outputs/llama4/llama4_maverick_outputs.json
outputs/qwen3/Qwen3-14B_outputs.json
outputs/qwen3/Qwen3-32B_outputs.json
outputs/qwen3/Qwen3-235B-A22B-Thinking_outputs.json
outputs/phi3/Phi-3_Mini_4K_outputs.json
outputs/phi3/Phi-3_Medium_4K_outputs.json
```

Mỗi record bao gồm output model, token usage, cost, generation stats OpenRouter, timing, và metric token/s.

## Chạy LLM Judge

Sau khi đã có output thô, chạy judge để chấm điểm:

```bash
python src/llm_judge.py --provider openrouter --judge-model deepseek/deepseek-chat-v3.1
```

Chỉ judge model cụ thể:

```bash
python src/llm_judge.py \
  --provider openrouter \
  --judge-model deepseek/deepseek-chat-v3.1 \
  --predictions outputs/llama4/llama4_scout_outputs.json \
              outputs/llama4/llama4_maverick_outputs.json \
              outputs/qwen3/Qwen3-14B_outputs.json \
              outputs/qwen3/Qwen3-32B_outputs.json
```

Các flag hữu ích:

| Flag | Mô tả |
| :--- | :--- |
| `--resume` | Tiếp tục từ vị trí bị dừng (không chấm lại những câu đã xong) |
| `--dry-run` | Chỉ kiểm tra số dòng matched, không gọi API |
| `--limit N` | Chỉ chấm N câu đầu tiên |
| `--sleep 21.0` | Thời gian chờ giữa các request (tăng nếu bị rate limit) |

Judge results bao gồm:

- Điểm rubric chất lượng (0-10) theo nhiều metric
- Metric hiệu quả cost/latency
- Tỷ lệ reliability: empty output, length stop, missing generation stats

Output được ghi vào:

```text
outputs/llm_judge/
├── results/
│   ├── llm_judge_results.json
│   └── llm_judge_summary.json
├── tables/
│   ├── llm_dashboard_table.csv
│   ├── llm_metric_long_table.csv
│   └── llm_model_leaderboard.csv
└── visuals/<model_family>/
    ├── 01_model_leaderboard.png
    ├── 02_score_distribution.png
    ├── 03_score_by_difficulty_category.png
    ├── 04_category_model_heatmap.png
    ├── 05_metric_model_heatmap.png
    ├── 06_metric_radar.png
    └── 07_efficiency_length_issues.png
```

## Chạy qua Notebook

Mở Jupyter notebook để chạy judge và xem dashboard trực quan:

```bash
jupyter notebook notebooks/llm_judge_evaluation.ipynb
```

Hoặc dùng JupyterLab:

```bash
jupyter lab notebooks/llm_judge_evaluation.ipynb
```

## Chạy Phi-3 cục bộ (tuỳ chọn)

`phi3.py` hỗ trợ chạy Phi-3 cục bộ qua ONNX/CUDA. Yêu cầu GPU NVIDIA và ONNX Runtime CUDA.

```bash
python src/phi3.py --model phi3-medium
```