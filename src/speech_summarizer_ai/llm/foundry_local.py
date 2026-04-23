"""Foundry Local 経由の要約（``foundry-local-sdk-winml``）。"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speech_summarizer_ai import settings as config
from speech_summarizer_ai.platform_utils import paths


class FoundryLocalNotAvailableError(ImportError):
    """``foundry_local_sdk`` が未インストール、または読み込めない場合。"""


class FoundryLocalModelNotCachedError(RuntimeError):
    """``allow_download=False`` でキャッシュからの load が続けて失敗したとき。未配置以外（ロック等）も含む。"""


def _load_failure_user_hint(exc: BaseException) -> str:
    """モデルパス不存在エラー向けの補足ヒント文を返す。

    Args:
        exc: SDK が送出した例外。

    Returns:
        str: ヒント文字列。該当しない場合は空文字。
    """
    t = str(exc).replace("\r\n", "\n")
    if "Model path does not exist" not in t:
        return ""
    return (
        " ヒント: ログの「Failed to load model …」に出る完全な ID（例: "
        "qwen2.5-0.5b-instruct-generic-gpu:4）用のファイルが ``models/llm`` に無い状態です。"
        "短いエイリアスはマシンごとに別バリアントへ解決されます。"
        "``settings/llm.py`` の FOUNDRY_LLM_MODEL_ALIAS をその完全 ID に合わせるか、"
        "起動時の LLM 取得を再実行してください。"
    )


def _import_sdk() -> tuple[type, type]:
    """Foundry Local SDK を import し、設定型とマネージャ型を返す。

    Returns:
        tuple[type, type]: ``(Configuration, FoundryLocalManager)`` のクラスオブジェクト。

    Raises:
        FoundryLocalNotAvailableError: パッケージが未インストールまたは import できない場合。
    """
    try:
        from foundry_local_sdk import Configuration, FoundryLocalManager

        return Configuration, FoundryLocalManager
    except ImportError as e:
        raise FoundryLocalNotAvailableError(
            "Foundry Local SDK が見つかりません。\n"
            "pip install foundry-local-sdk-winml foundry-local-core-winml openai\n"
            "詳細: https://learn.microsoft.com/en-us/azure/foundry-local/get-started"
        ) from e


# インポート時に 1 回だけ束ねる（同一 str オブジェクトを共有する）。
DEFAULT_SUMMARY_SYSTEM_PROMPT: str = """あなたはビジネス文書の要約に熟練したアシスタントです。
ユーザーから「---文字起こし本文---」のあとに続く会話ログだけを根拠に、日本語でプロフェッショナルな要約を書いてください。

入力の特徴:
- 行頭に `[HH:MM:SS]` などの時刻が付いたリアルタイム文字起こしが多く含まれます。時刻は「いつ頃どの話題が出たか」の手がかりに使い、必要な箇所だけ簡潔に参照して構いません（時刻の列挙や全文の時系列コピーは不要です）。

文体・品質:
- 丁寧で簡潔なビジネス調（です・ます調）に統一する。
- 箇条書きまたは短い段落で、意思決定者が一目で把握できる構造にする。

含める内容（該当がなければ省略）:
- 会話の目的または概要
- 合意・決定事項
- 未決の論点・リスク
- 次のアクション（分かる範囲で担当・期限を明記）

厳守:
- 文字起こしの繰り返し・コピペは禁止。要約は原文より短いことが原則。
- 入力にない事実の補完・推測は禁止。
- 出力は要約本文のみ（挨拶、前置き、「以下に要約します」等は禁止）。
"""

DEFAULT_TITLE_SYSTEM_PROMPT: str = """あなたはビジネス文書・会議記録のタイトル付けに熟練したアシスタントです。
ユーザーから「---文字起こし本文---」のあとに続く会話ログだけを根拠に、一覧・議事・CRM に載せるのにふさわしい、プロフェッショナルで具体的な日本語タイトルを1つだけ出力してください。

入力の特徴:
- `[HH:MM:SS]` 等の時刻付き行が含まれることがあります。内容の主題を把握するための参考にし、タイトルに時刻を無理に入れる必要はありません（時刻だらけのタイトルは避ける）。

厳守:
- 出力はタイトル文字列そのもののみ（1行。改行・箇条書き・番号・引用符は付けない）。
- 件名のように簡潔かつ内容が伝わる語を選ぶ。「打ち合わせ」「会議」単独のような泛用語だけのタイトルは避ける。
- 入力にない固有名詞や案件名の捏造は禁止。
- 前置き（「タイトル:」「【」等）は禁止。
"""


def _wrap_transcript_for_summary_user_message(transcript_body: str) -> str:
    """要約用 user メッセージ本文を組み立てる（文字起こし区画を明示する）。

    Args:
        transcript_body: 文字起こしプレーンテキスト。

    Returns:
        str: LLM に渡す user ロール本文。
    """
    body = transcript_body.strip()
    return (
        "次のセクションは会話の文字起こしです。あなたの応答ではこの文字起こしを"
        "一言一句写さず、内容だけを要約してください。\n\n"
        "---文字起こし本文---\n"
        f"{body}"
    )


def _looks_like_transcript_echo(summary: str, raw_transcript: str) -> bool:
    """要約出力が文字起こしのコピーに見えるかどうかを返す。

    Args:
        summary: モデル出力。
        raw_transcript: 元の文字起こし。

    Returns:
        bool: コピーとみなせる場合 True。
    """
    s, r = summary.strip(), raw_transcript.strip()
    if not s or not r:
        return False
    if s == r:
        return True
    if len(r) < 40:
        return False
    if len(s) >= int(len(r) * 0.92) and s[:120] == r[:120]:
        return True
    return False


def _extract_assistant_text_from_complete_chat(
    client: Any, messages: list[dict[str, str]]
) -> str:
    """``complete_chat`` の応答から assistant の本文を取り出す。

    Args:
        client: チャットクライアント。
        messages: 送受信メッセージ列。

    Returns:
        str: assistant のテキスト。取得できなければ空文字。
    """
    try:
        result = client.complete_chat(messages)
    except Exception:
        return ""
    choices = getattr(result, "choices", None) or []
    for choice in choices:
        msg = getattr(choice, "message", None)
        if msg is None:
            continue
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if role == "assistant" and isinstance(content, str) and content.strip():
            return content.strip()
    if choices:
        try:
            msg = choices[0].message
            c = getattr(msg, "content", None)
            if isinstance(c, str) and c.strip():
                return c.strip()
        except (IndexError, AttributeError):
            pass
    return ""


@dataclass(frozen=True)
class SummarizeResult:
    """要約 1 件分の結果。

    Attributes:
        text: 要約本文。
        model_alias: 使用したモデル別名。
    """

    text: str
    model_alias: str


def format_transcript_lines(lines: Sequence[tuple[str, str]]) -> str:
    """``(時刻, 本文)`` を ``[時刻] 本文`` 形式のテキストに連結する。

    Args:
        lines: 段落の列。

    Returns:
        str: 改行区切り。本文が空の要素は除く。
    """
    parts: list[str] = []
    for ts, body in lines:
        t = str(ts).strip()
        b = str(body).strip()
        if not b:
            continue
        if t:
            parts.append(f"[{t}] {b}")
        else:
            parts.append(b)
    return "\n".join(parts)


def _wrap_transcript_for_title_user_message(transcript_body: str) -> str:
    """タイトル生成用 user メッセージ本文を組み立てる。

    Args:
        transcript_body: 文字起こしプレーンテキスト。

    Returns:
        str: LLM に渡す user ロール本文。
    """
    body = transcript_body.strip()
    return (
        "次のセクションは会話の文字起こしです。内容にふさわしい短いタイトルを"
        "1行だけ出力してください（説明文や「タイトル」という語は不要）。\n\n"
        "---文字起こし本文---\n"
        f"{body}"
    )


def sanitize_meeting_title(raw: str, *, max_chars: int | None = None) -> str:
    """LLM のタイトル出力を 1 行・長さ上限に揃える。

    Args:
        raw: 生文字列。
        max_chars: 上限。``None`` なら設定値。

    Returns:
        str: 整形結果。空入力は ``""``。
    """
    lim = max_chars if max_chars is not None else config.FOUNDRY_LLM_TITLE_MAX_CHARS
    s = raw.strip()
    if not s:
        return ""
    line = s.splitlines()[0].strip()
    for prefix in ("タイトル:", "タイトル：", "【", "■", "- ", "・"):
        if line.startswith(prefix):
            line = line[len(prefix) :].strip()
    line = line.strip("「」『』\"'")
    if len(line) > lim:
        line = line[: max(1, lim - 1)] + "…"
    return line


class FoundryLocalSummarizer:
    """Foundry Local チャットで要約する。

    マネージャはプロセス単位で 1 回初期化し、モデルハンドルはインスタンスごとに load / unload する。
    """

    _manager_lock = threading.Lock()
    _manager_initialized = False

    def __init__(
        self,
        *,
        model_alias: str = config.FOUNDRY_LLM_MODEL_ALIAS,
        app_name: str = config.FOUNDRY_LLM_APP_NAME,
        system_prompt: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        project_root: Path | None = None,
    ) -> None:
        """インスタンスを初期化する。

        Args:
            model_alias: カタログ上のモデル別名。
            app_name: Foundry Local アプリ名。
            system_prompt: 要約用 system プロンプト。``None`` のとき既定の要約プロンプトを使う。
            temperature: チャット生成温度。
            max_tokens: チャット最大トークン。
            project_root: モデルキャッシュ等の基準パス。``None`` で自動解決。

        Returns:
            None
        """
        self._model_alias = model_alias
        self._app_name = app_name
        self._system_prompt = (
            system_prompt
            if system_prompt is not None
            else DEFAULT_SUMMARY_SYSTEM_PROMPT
        )
        self._temperature = temperature
        self._max_tokens = max_tokens
        root = project_root if project_root is not None else paths.project_root()
        if config.FOUNDRY_LLM_CACHE_IN_PROJECT:
            self._model_cache_dir = str(
                paths.foundry_llm_cache_directory(root).resolve()
            )
        else:
            self._model_cache_dir = None
        self._configuration: Any = None
        self._manager: Any = None
        self._model: Any = None
        self._loaded = False

    def _ensure_manager(self) -> None:
        """プロセス単位の Foundry Local マネージャを初期化し、インスタンスに紐付ける。

        Returns:
            None
        """
        Configuration, FoundryLocalManager = _import_sdk()
        with FoundryLocalSummarizer._manager_lock:
            if not FoundryLocalSummarizer._manager_initialized:
                if self._model_cache_dir is not None:
                    Path(self._model_cache_dir).mkdir(parents=True, exist_ok=True)
                cfg_kw: dict[str, Any] = {"app_name": self._app_name}
                if self._model_cache_dir is not None:
                    cfg_kw["model_cache_dir"] = self._model_cache_dir
                try:
                    self._configuration = Configuration(**cfg_kw)
                except TypeError:
                    # 古い SDK で ``model_cache_dir`` 未対応のときは従来どおり
                    print(
                        "[summarize] Foundry Local: model_cache_dir 非対応の SDK のため "
                        "既定キャッシュのみ使用します。",
                        flush=True,
                    )
                    self._configuration = Configuration(app_name=self._app_name)
                FoundryLocalManager.initialize(self._configuration)
                if self._model_cache_dir is not None:
                    print(
                        f"[summarize] Foundry Local model_cache_dir={self._model_cache_dir!r}",
                        flush=True,
                    )
                FoundryLocalSummarizer._manager_initialized = True
            self._manager = FoundryLocalManager.instance

    def _iter_family_variant_ids(self, catalog: Any) -> list[str]:
        """``list_models()`` から ``self._model_alias`` と同族のモデル ID を列挙する。

        Args:
            catalog: ``FoundryLocalManager.catalog``。

        Returns:
            list[str]: 重複のないモデル ID 一覧。取得に失敗した場合は空リスト。
        """
        family = self._model_alias.strip()
        try:
            entries = list(catalog.list_models())
        except Exception as e:
            print(
                f"[summarize] catalog.list_models() 失敗: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            return []

        out: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            mid = getattr(entry, "id", None)
            malias = getattr(entry, "alias", None)
            if not isinstance(mid, str) or not mid:
                continue
            matches = (
                malias == family
                or mid == family
                or mid.startswith(f"{family}-")
                or mid.startswith(f"{family}:")
            )
            if not matches:
                continue
            if mid in seen:
                continue
            seen.add(mid)
            out.append(mid)
        return out

    def load_model(
        self,
        *,
        download_eps: bool = True,
        allow_download: bool = True,
        ep_progress: Callable[[str, float], None] | None = None,
        model_download_progress: Callable[[float], None] | None = None,
    ) -> None:
        """モデルをロードする（キャッシュ候補の試行、必要ならダウンロード）。

        Args:
            download_eps: True のとき EP を取得・登録する。
            allow_download: False のときキャッシュのみ。候補がことごとく失敗すると
                :class:`FoundryLocalModelNotCachedError`。
            ep_progress: 任意。EP 名と進捗（0〜100）を渡すコールバック。
            model_download_progress: 任意。モデル DL 進捗（0〜100）を渡すコールバック。

        Returns:
            None
        """
        self._ensure_manager()
        assert self._manager is not None

        if download_eps:
            current_ep = ""

            def _ep_cb(ep_name: str, percent: float) -> None:
                nonlocal current_ep
                if ep_progress is not None:
                    ep_progress(ep_name, percent)
                else:
                    if ep_name != current_ep:
                        if current_ep:
                            print()
                        current_ep = ep_name
                    print(f"\r  {ep_name:<30}  {percent:5.1f}%", end="", flush=True)

            self._manager.download_and_register_eps(progress_callback=_ep_cb)
            if ep_progress is None and current_ep:
                print()

        catalog = self._manager.catalog

        def _mdl(progress: float) -> None:
            if model_download_progress is not None:
                model_download_progress(progress)
            else:
                print(
                    f"\rDownloading model: {progress:.2f}%",
                    end="",
                    flush=True,
                )

        def _apply_chat_client_settings() -> None:
            client = self._model.get_chat_client()
            client.settings.temperature = self._temperature
            client.settings.max_tokens = self._max_tokens

        def _resolved_id_of(model: Any, fallback: str) -> str:
            rid = getattr(model, "id", None)
            return rid if isinstance(rid, str) and rid else fallback

        def _try_load_variant(
            variant: str, *, retries: int = 0
        ) -> tuple[Any, BaseException | None]:
            """指定バリアントで ``get_model`` と ``load`` を試す。

            Args:
                variant: エイリアスまたは完全 ID。
                retries: 失敗時の追加リトライ回数。

            Returns:
                tuple[Any, BaseException | None]: 成功時は ``(ハンドル, None)``。失敗時は ``(None, 最後の例外)``。
            """
            last: BaseException | None = None
            for attempt in range(retries + 1):
                if attempt == 1:
                    time.sleep(0.25)
                elif attempt >= 2:
                    time.sleep(0.75)
                try:
                    handle = catalog.get_model(variant)
                    if handle is None:
                        raise RuntimeError("catalog.get_model が None を返した")
                    handle.load()
                    return handle, None
                except Exception as e:
                    last = e
                    print(
                        f"[summarize] LLM load() 失敗 "
                        f"(候補={variant!r}, 試行 {attempt + 1}/{retries + 1}): "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
            return None, last

        root_for_marker = paths.project_root()

        candidates: list[tuple[str, int]] = []  # (variant, extra retries)
        seen_candidates: set[str] = set()

        def _add_candidate(variant: str | None, retries: int) -> None:
            if not variant:
                return
            key = variant.strip()
            if not key or key in seen_candidates:
                return
            seen_candidates.add(key)
            candidates.append((key, retries))

        prev_resolved = paths.read_llm_resolved_id(root_for_marker, self._model_alias)

        # 1) 短いエイリアス（SDK が環境に合うバリアントを解決）。キャッシュ利用時はこれが最も安定。
        _add_candidate(self._model_alias, retries=2 if not allow_download else 0)

        # 2) マーカー記載の解決済み完全 ID は、models/llm に実体があるときだけ試す（カタログとパスがずれた ID の無駄打ちを避ける）
        if prev_resolved and paths.foundry_llm_resolved_weights_present(
            root_for_marker, prev_resolved
        ):
            _add_candidate(prev_resolved, retries=1)

        # 3) カタログから列挙したファミリー一致の他バリアント（環境差・コピー移送対策）
        for cid in self._iter_family_variant_ids(catalog):
            _add_candidate(cid, retries=0)

        last_err: BaseException | None = None
        loaded_variant: str | None = None
        for variant, retries in candidates:
            handle, err = _try_load_variant(variant, retries=retries)
            if err is None and handle is not None:
                self._model = handle
                self._loaded = True
                loaded_variant = variant
                last_err = None
                resolved = _resolved_id_of(handle, variant)
                print(
                    f"[summarize] LLM load 成功: 要求={variant!r} 解決 ID={resolved!r}",
                    flush=True,
                )
                try:
                    paths.write_llm_probe_marker(
                        root_for_marker, self._model_alias, resolved
                    )
                except OSError:
                    pass
                break
            last_err = err

        if last_err is None and loaded_variant is not None:
            _apply_chat_client_settings()
            return

        if not allow_download:
            cache_hint = self._model_cache_dir or "（Foundry SDK 既定キャッシュ）"
            hint = _load_failure_user_hint(last_err) if last_err is not None else ""
            tried = ", ".join(v for v, _ in candidates) or "(なし)"
            raise FoundryLocalModelNotCachedError(
                f"LLM モデル {self._model_alias!r} をキャッシュから load() できませんでした。"
                f"未ダウンロードの場合は起動時の要約モデル取得を完了してください。"
                f"取得済みでもファイルロック・ディスク・GPU/EP の一時エラーで失敗することがあります。"
                f" model_cache_dir={cache_hint!r}。"
                f" 試行候補: [{tried}]。"
                f" 直近の原因: {type(last_err).__name__ if last_err else 'NoError'}: {last_err}"
                f"{hint}"
            ) from last_err

        # ダウンロード経路: 短いエイリアスで取り直し、SDK に環境適合バリアントを選ばせる。
        download_handle = catalog.get_model(self._model_alias)
        download_handle.download(_mdl)
        if model_download_progress is None:
            print()
        download_handle.load()
        self._model = download_handle
        self._loaded = True

        resolved = _resolved_id_of(download_handle, self._model_alias)
        print(
            f"[summarize] LLM download 成功: alias={self._model_alias!r} 解決 ID={resolved!r}",
            flush=True,
        )
        try:
            paths.write_llm_probe_marker(root_for_marker, self._model_alias, resolved)
        except OSError:
            pass

        _apply_chat_client_settings()

    def unload(self) -> None:
        """ロード済みモデルハンドルをアンロードする。

        Returns:
            None
        """
        if self._model is not None and self._loaded:
            try:
                self._model.unload()
            except Exception:
                pass
            self._loaded = False

    def generate_conversation_title(self, transcript: str) -> str:
        """文字起こしから一覧表示用の短いタイトルを生成する。

        Args:
            transcript: 文字起こし全文。

        Returns:
            str: 整形済みタイトル。入力が空なら空文字。

        Raises:
            RuntimeError: モデル未ロードの場合。
        """
        if not self._loaded or self._model is None:
            raise RuntimeError(
                "モデルが未ロードです。先に load_model() を呼び出してください。"
            )
        raw = transcript.strip()
        if not raw:
            print("[summarize] LLM title: skip (empty input)", flush=True)
            return ""

        system = DEFAULT_TITLE_SYSTEM_PROMPT
        user_message = _wrap_transcript_for_title_user_message(raw)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        client = self._model.get_chat_client()
        prev_temp = getattr(client.settings, "temperature", self._temperature)
        prev_max = getattr(client.settings, "max_tokens", self._max_tokens)
        try:
            client.settings.temperature = 0.25
            client.settings.max_tokens = config.FOUNDRY_LLM_TITLE_MAX_TOKENS
            print(
                f"[summarize] LLM title: generating (model={self._model_alias}, "
                f"chars={len(raw)})…",
                flush=True,
            )

            def _delta_content(chunk: Any) -> str | None:
                try:
                    delta = chunk.choices[0].delta
                    return (
                        getattr(delta, "content", None) if delta is not None else None
                    )
                except (IndexError, AttributeError):
                    return None

            parts: list[str] = []
            for chunk in client.complete_streaming_chat(messages):
                c = _delta_content(chunk)
                if c:
                    parts.append(c)
            text = "".join(parts).strip()
            if not text:
                text = _extract_assistant_text_from_complete_chat(client, messages)
            out = sanitize_meeting_title(text)
            print(
                f"[summarize] LLM title: done (raw_len={len(text)}, title_len={len(out)})",
                flush=True,
            )
            return out
        finally:
            try:
                client.settings.temperature = prev_temp
                client.settings.max_tokens = prev_max
            except Exception:
                client.settings.temperature = self._temperature
                client.settings.max_tokens = self._max_tokens

    def summarize_transcript(
        self,
        transcript: str,
        *,
        extra_instructions: str | None = None,
        stream: bool = False,
        on_stream_chunk: Callable[[str], None] | None = None,
    ) -> SummarizeResult:
        """文字起こし全文を要約する。

        Args:
            transcript: 文字起こしテキスト（プレーン）。
            extra_instructions: システム指示に追加する一文（任意）。
            stream: True のときストリーミング生成。``on_stream_chunk`` に断片を渡す。
            on_stream_chunk: ストリーミング時のコールバック（文字列断片）。

        Returns:
            SummarizeResult: 要約テキストとモデル別名。ストリーミング時も結合結果を ``text`` に入れる。

        Raises:
            RuntimeError: モデル未ロードの場合。
        """
        if not self._loaded or self._model is None:
            raise RuntimeError(
                "モデルが未ロードです。先に load_model() を呼び出してください。"
            )

        raw_transcript = transcript.strip()
        if not raw_transcript:
            print("[summarize] LLM: skip (empty input after strip)", flush=True)
            return SummarizeResult(text="", model_alias=self._model_alias)

        if extra_instructions:
            system = (
                self._system_prompt.rstrip()
                + "\n\n追加の指示:\n"
                + extra_instructions.strip()
            )
        else:
            system = self._system_prompt

        user_message = _wrap_transcript_for_summary_user_message(raw_transcript)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

        client = self._model.get_chat_client()
        client.settings.temperature = self._temperature
        client.settings.max_tokens = self._max_tokens

        print(
            f"[summarize] LLM: generating (model={self._model_alias}, "
            f"transcript_chars={len(raw_transcript)}, stream={stream})…",
            flush=True,
        )

        def _delta_content(chunk: Any) -> str | None:
            try:
                delta = chunk.choices[0].delta
                return getattr(delta, "content", None) if delta is not None else None
            except (IndexError, AttributeError):
                return None

        # Foundry Local（WinML）では complete_chat がユーザー入力をそのまま返すことがあるため、
        # 非ストリーム時も complete_streaming_chat で集約する（fundry_test と同じ経路）。
        parts: list[str] = []
        for chunk in client.complete_streaming_chat(messages):
            content = _delta_content(chunk)
            if content:
                parts.append(content)
                if stream and on_stream_chunk is not None:
                    on_stream_chunk(content)
        text = "".join(parts).strip()

        # 万一ストリームが空ならcomplete_chatをフォールバック（抽出は assistant 優先）
        if not text:
            text = _extract_assistant_text_from_complete_chat(client, messages)

        # 文字起こし／ラップ済み user 全文のコピーとみなす場合はフォールバック再試行
        if _looks_like_transcript_echo(
            text, raw_transcript
        ) or _looks_like_transcript_echo(text, user_message):
            print(
                "[summarize] LLM: 出力が文字起こし全文に近いため、"
                "user単一メッセージでストリーム再試行します。",
                flush=True,
            )
            alt_messages: list[dict[str, str]] = [
                {
                    "role": "user",
                    "content": (
                        f"{system}\n\n{_wrap_transcript_for_summary_user_message(raw_transcript)}"
                    ),
                }
            ]
            parts2: list[str] = []
            for chunk in client.complete_streaming_chat(alt_messages):
                c = _delta_content(chunk)
                if c:
                    parts2.append(c)
            text = "".join(parts2).strip()
            if not text:
                text = _extract_assistant_text_from_complete_chat(client, alt_messages)

        print(
            f"[summarize] LLM: done (output_chars={len(text)}, "
            f"stream_mode={'callback' if stream else 'aggregate'})",
            flush=True,
        )
        return SummarizeResult(text=text, model_alias=self._model_alias)

    def summarize_transcript_lines(
        self,
        lines: Sequence[tuple[str, str]],
        **kwargs: Any,
    ) -> SummarizeResult:
        """時刻付き行を 1 テキストに整形してから要約する。

        Args:
            lines: ``(時刻, 本文)`` の列。
            **kwargs: :meth:`summarize_transcript` にそのまま渡す。

        Returns:
            SummarizeResult: :meth:`summarize_transcript` と同じ。
        """
        n = len(lines)
        body = format_transcript_lines(lines)
        print(
            f"[summarize] Formatted transcript: {n} raw line(s) -> "
            f"{len(body)} char(s) for LLM input",
            flush=True,
        )
        return self.summarize_transcript(body, **kwargs)


def summarize_transcript_with_foundry_local(
    transcript: str,
    *,
    model_alias: str = config.FOUNDRY_LLM_MODEL_ALIAS,
    project_root: Path | None = None,
    load_model_kwargs: dict[str, Any] | None = None,
    **summarize_kwargs: Any,
) -> SummarizeResult:
    """1 回限りでモデルを load して要約し、unload するヘルパー。

    Args:
        transcript: 文字起こしテキスト。
        model_alias: モデル別名。
        project_root: プロジェクトルート。``None`` で自動。
        load_model_kwargs: :meth:`FoundryLocalSummarizer.load_model` に渡す引数。
        **summarize_kwargs: :meth:`FoundryLocalSummarizer.summarize_transcript` に渡す引数。

    Returns:
        SummarizeResult: 要約結果。
    """
    load_model_kwargs = load_model_kwargs or {}
    s = FoundryLocalSummarizer(model_alias=model_alias, project_root=project_root)
    try:
        s.load_model(**load_model_kwargs)
        return s.summarize_transcript(transcript, **summarize_kwargs)
    finally:
        s.unload()


def foundry_sdk_importable() -> bool:
    """Foundry Local SDK が import 可能かどうかを返す。

    Returns:
        bool: import に成功すれば True。
    """
    try:
        _import_sdk()
    except ImportError:
        return False
    return True


def probe_foundry_llm_ready(
    project_root: Path | None = None,
    *,
    model_alias: str | None = None,
) -> bool:
    """キャッシュのみ（追加 DL なし）でモデル load できるかを試す。

    SDK 未導入時は True を返す（起動ダイアログで LLM 必須にしないため）。

    Args:
        project_root: プロジェクトルート。``None`` で自動。
        model_alias: モデル別名。``None`` で設定値。

    Returns:
        bool: load に成功すれば True。未取得や失敗は False。
    """
    try:
        if model_alias is not None:
            s = FoundryLocalSummarizer(
                project_root=project_root, model_alias=model_alias
            )
        else:
            s = FoundryLocalSummarizer(project_root=project_root)
        s.load_model(download_eps=True, allow_download=False)
        s.unload()
        return True
    except FoundryLocalNotAvailableError:
        return True
    except FoundryLocalModelNotCachedError:
        return False
    except Exception:
        return False
