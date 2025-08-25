简体中文： 基于sensevoice官方webui修改而来，可以单文件或批量输出SRT字幕。（**注意：Gradio不支持中文符号如《》、（）等作为文件名称使用。**）

English: Based on the official sensevoice webui, it can output SRT subtitles for single files or in batch mode. 

日本語: これはsensevoiceの公式webuiを基にしたものです。単一ファイルまたはバッチ処理でSRT字幕を出力できます。

环境配置：

一、使用uv创建虚拟环境，如：
```uv venv --python 3.12```。

二、使用如下命令配置环境：
```uv pip install -r requirements``` or ```uv add -r requirements```

三、请自行安装torch和torchaudio：<https://pytorch.org/get-started/locally/>
for example:
```uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126```

四、如需要使用CPU推理，请将```device="cuda:0"```修改成```device="cpu"```，再使用如下命令配置环境：
```uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu```

五、关于模型下载：
修改```sensevoicesmall```和```fmsn_vad```的模型加载参数，当```disable_update=False```时，会自动下载模型。
下载完成后请改为```disable_update=True```，以缩短程序启动时间。


**需要批量转录时，务必先尝试单个文件转录，找到理想的静音阈值，确保断句的准确性。**

![WebUI](屏幕.jpg)