# 開発記録

## 目的
- 動画の手ブレ補正（光学フロー）と元音声の合成を行うツールを整理・公開する。

## 変更履歴（抜粋）
- CLI版の音声合成対応（moviepy を使用）。
- CLI版の調整パラメータ（radius, scaling_factor）追加。
- CLI版のズーム&クロップで黒縁を軽減。
- CLI版の進捗表示（tqdm）追加。
- GUI版の進捗バー追加。
- GUI版のドラッグ&ドロップ対応（tkinterdnd2）。
- README を日本語で整備。

## 依存関係
- opencv-python
- numpy
- moviepy
- tqdm
- tkinter（Python 同梱）
- tkinterdnd2

## メモ
- 処理中に `temp_stabilized.mp4` を生成。
- 出力は `libx264` / `aac` を使用。

