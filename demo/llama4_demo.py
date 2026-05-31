import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from IPython import get_ipython
from IPython.display import Markdown, display
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_SCOUT_MODEL = "meta-llama/llama-4-scout"
DEFAULT_MAVERICK_MODEL = "meta-llama/llama-4-maverick"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1200

SYSTEM_PROMPT = """Bạn là một chuyên gia Big Data Engineer có kinh nghiệm triển khai hệ thống xử lý dữ liệu lớn trong môi trường production. Hãy trả lời chính xác, dễ hiểu, có cấu trúc rõ ràng và liên hệ với các use case thực tế. Khi so sánh công nghệ, cần nêu rõ kiến trúc, ưu điểm, hạn chế, trường hợp nên sử dụng và trade-off khi triển khai."""

USER_PROMPT = """Hãy trình bày sự khác nhau giữa Hadoop và Apache Spark trong xử lý dữ liệu lớn.

Yêu cầu câu trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo các phần sau:

1. Tổng quan ngắn gọn về Hadoop và Apache Spark.
2. So sánh hai công nghệ theo các tiêu chí:
   - Kiến trúc xử lý dữ liệu
   - Cơ chế lưu trữ và tính toán
   - Hiệu năng
   - Khả năng xử lý batch, streaming và interactive analytics
   - Mức độ phù hợp trong các hệ thống Big Data hiện đại
3. Trình bày ưu điểm và hạn chế của Hadoop.
4. Trình bày ưu điểm và hạn chế của Apache Spark.
5. Cho ví dụ thực tế: khi nào nên dùng Hadoop, khi nào nên dùng Spark.
6. Kết luận ngắn gọn: nếu xây dựng một pipeline phân tích dữ liệu lớn hiện nay, nên chọn công nghệ nào trong từng trường hợp.

Yêu cầu thêm:
- Trả lời dễ hiểu cho sinh viên mới học Big Data.
- Không chỉ liệt kê, hãy giải thích ý nghĩa của từng điểm khác biệt.
- Có thể dùng bảng so sánh nếu phù hợp.
- Không trả lời quá dài, ưu tiên rõ ràng và thực tế."""


def load_environment() -> tuple[OpenAI, dict[str, dict[str, str]]]:
    """Load OpenRouter configuration and return a ready OpenAI-compatible client."""
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing OPENROUTER_API_KEY. Add it to .env or export it before running this demo."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
    client = OpenAI(base_url=base_url, api_key=api_key)

    models = {
        "scout": {
            "display_name": "Llama 4 Scout",
            "model_id": os.getenv("LLAMA4_SCOUT_MODEL", DEFAULT_SCOUT_MODEL),
        },
        "maverick": {
            "display_name": "Llama 4 Maverick",
            "model_id": os.getenv("LLAMA4_MAVERICK_MODEL", DEFAULT_MAVERICK_MODEL),
        },
    }
    return client, models


def call_llama4_model(
    client: OpenAI,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call a Llama 4 model through OpenRouter and return assistant text."""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenRouter API call failed: {exc}") from exc

    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise RuntimeError("OpenRouter returned an empty response.")

    return content.strip()


def display_markdown_response(response: str) -> None:
    """Render Markdown in notebooks; print plain Markdown in terminals."""
    shell = get_ipython()
    if shell and shell.__class__.__name__ == "ZMQInteractiveShell":
        display(Markdown(response))
        return

    print(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo Llama 4 on OpenRouter for a Big Data prompt.")
    parser.add_argument(
        "--model",
        choices=["scout", "maverick"],
        default="scout",
        help="Llama 4 model to use. Use --model maverick for the Maverick demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        client, models = load_environment()
        model_config = models[args.model]
        model_id = model_config["model_id"]

        print(f"Model: {model_config['display_name']}")
        print(f"Model ID: {model_id}")
        print(f"Temperature: {DEFAULT_TEMPERATURE}")
        print(f"Max tokens: {DEFAULT_MAX_TOKENS}")
        print()

        response = call_llama4_model(
            client=client,
            model_id=model_id,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        display_markdown_response(response)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
