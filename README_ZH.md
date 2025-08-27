[中文](README_ZH.md) | [English](README.md)

基于sensevoice官方webui修改而来，可以单文件或批量输出SRT字幕。（**注意：Gradio不支持中文符号如《》、（）等作为文件名称使用。**）

环境配置：

一、使用uv创建虚拟环境，如：
```uv venv --python 3.12```

二、使用如下命令配置环境：
```uv pip install -r requirements``` 或者 ```uv add -r requirements```

三、使用**CUDA**转录，安装torch和torchaudio：<https://pytorch.org/get-started/locally/>，
比如:```uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126```

四、使用**CPU**转录，安装torch和torchaudio：
```uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu```

五、关于模型下载：
修改```sensevoicesmall```和```fmsn_vad```的模型加载参数，当```disable_update=False```时，会自动下载模型。
下载完成后改为```disable_update=True```，可以缩短程序启动时间。


**需要批量转录时，务必先尝试单个文件转录，找到理想的静音阈值，确保断句的准确性。**

![WebUI](屏幕.jpg)