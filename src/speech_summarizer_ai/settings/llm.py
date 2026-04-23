"""LLM 要約（Foundry Local）の設定。"""

from __future__ import annotations

# 録音停止後にリアルタイム文字起こしが DB へ反映したタイミングで自動要約するか。
AUTO_LLM_SUMMARIZE_AFTER_STT: bool = True

# Foundry Local SDK の ``Configuration(app_name=...)`` に渡すアプリ名。
FOUNDRY_LLM_APP_NAME: str = "Speech-Summarizer-AI"

# 要約モデルのカタログ別名（短いファミリー名）。SDK が EP に合わせて完全 ID に解決する。
# load 時は ``.speech_summarizer_llm_probe_ok`` の記録 → この別名 → カタログの同族 ID の順で試す。
# 固定したいときだけ ``"...-instruct-generic-gpu:4"`` のような完全 ID を書く。
FOUNDRY_LLM_MODEL_ALIAS: str = "qwen2.5-0.5b"

# ``True`` なら ``models/llm`` をキャッシュにし起動時に ONNX を検査。``False`` は SDK 既定キャッシュ。
FOUNDRY_LLM_CACHE_IN_PROJECT: bool = True

# LLM が生成する会話タイトルの文字数上限。
FOUNDRY_LLM_TITLE_MAX_CHARS: int = 48

# タイトル生成時の ``max_tokens`` 上限。
FOUNDRY_LLM_TITLE_MAX_TOKENS: int = 160
