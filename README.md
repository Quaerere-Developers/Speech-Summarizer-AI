# Speech Summarizer AI

Speech Summarizer AI アプリケーションです。

## レイアウト

- `src/speech_summarizer_ai/` — Python パッケージ本体（`settings` / `platform_utils` / `domain` / `data` / `audio` / `stt` / `llm` / `controllers` / `ui`）
- `resources/icons/` — アプリアイコン（`app.svg` / Windows 用 `app.ico` / `app.png`）。PyInstaller ビルド時に同梱され、EXE にも埋め込まれる
- `scripts/` — 開発用エントリーポイント（`run_dev.py` など）
- 実行時データ（SQLite・録音セッション・ローカルモデル）は **`platform_utils.paths.project_root()`** の基準で配置されます。
  - **Windows:** 通常は `%LOCALAPPDATA%\WEEL\SpeechSummarizerAI\` 直下に `database/`・`sessions/`・`models/` が作成されます（MSIX 等で WinRT の `ApplicationData.local_folder` が使える場合はそちらを優先。未パッケージの EXE では上記にフォールバックします）。
  - **Windows 以外:** リポジトリルート（`src` の親）直下に同じ名前のフォルダが使われます。
- `tests/` — テスト
- `speech_summarizer_ai.spec` — Windows 向け PyInstaller 定義

## セットアップと起動

仮想環境を作成して依存関係を入れます。

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

または、パッケージを editable install すると `speech-summarizer-ai` コマンドが使えます。

```bash
pip install -e .
```

開発時はリポジトリルートから次のいずれかで起動できます。

```bash
python scripts/run_dev.py
```

editable install 後は次のコマンドでも起動できます。

```bash
speech-summarizer-ai
python -m speech_summarizer_ai
```

録音は `sessions/<セッションID>/audio.wav` に保存されます（基準ディレクトリは上記の `project_root()`。Windows の配布版では **ユーザーデータ領域** 側の `sessions/` です）。

## Windows 実行ファイルの作成（PyInstaller）

前提:

- Windows PC、**このリポジトリ用**の仮想環境で `requirements.txt`（または `pip install -e .`）を満たしていること（別プロジェクトの venv だと `webrtcvad` 不足などで PyInstaller の hook が失敗することがあります）
- リポジトリルートで作業する

手順:

```bash
venv\Scripts\activate
pyinstaller --noconfirm speech_summarizer_ai.spec
```

クリーンビルドする場合:

```bash
pyinstaller --noconfirm --clean speech_summarizer_ai.spec
```

成果物は **`dist/SpeechSummarizerAI.exe` の 1 ファイル**（onefile）です。初回起動時に展開のため、onedir 版より起動が遅くなることがあります。
