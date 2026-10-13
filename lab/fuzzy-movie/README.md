# 模糊记忆

实验二题目三。在这个目录里运行，不改动仓库根目录的 EstateAgents 项目。

用一段模糊的情节找回电影。对 Wikipedia Movie Plots 的英文剧情做句向量检索，并与 BM25 关键词检索并排比较。

## 数据

完整片库 34,886 部，年份 1901–2017。检索只用 `Plot`，不用额外的 `PlotSummary`。

```bash
python download_data.py
python build_index.py
```

Kaggle 原版可放在 `data/wiki_movie_plots_deduped.csv`。没有原版时，`download_data.py` 会下载 Hugging Face 备用 CSV。

`build_index.py` 会把每部电影的剧情切成短句并编码，首次大约需要半小时，并下载 `paraphrase-multilingual-MiniLM-L12-v2`。

## 运行

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。

演示前修改 `static/config.js` 里的学号和姓名，页面左上角会显示。

## 检索

- 语义检索：多语言 MiniLM 句向量。问句按逗号拆开，每个分句在该片的短句里取最大余弦，再相加。中文里的「陀螺」「冰山」「变成了猪」等会补一条英文短语。
- 关键词检索：BM25。英文按单词，中文按二字词。中文问句和英文剧情通常没有共同词。
- 电影宇宙：每部电影的短句向量取平均，再做 PCA，画在页面下方。
