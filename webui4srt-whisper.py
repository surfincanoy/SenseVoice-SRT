# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# To install requirements: uv pip install -U openai-whisper

import io  # 用于在内存中操作文件
import shutil
import threading
import webbrowser
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf  # 用于读取和裁剪音频文件
import torch
import torchaudio
from funasr import AutoModel

# 模型路径
model_dir = "iic/Whisper-large-v3-turbo"
vad_model_dir = "fsmn-vad"  # VAD模型路径


# 加载SenseVoice模型
model = AutoModel(
    model=model_dir,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    vad_kwargs={"max_single_segment_time": 30000},
    disable_update=True,
)


def open_page():
    webbrowser.open_new_tab("http://127.0.0.1:7860")


# 定义时间戳格式
def reformat_time(second):
    hours = int(second // 3600)
    minutes = int((second % 3600) // 60)
    secs = int(second % 60)
    millis = int((second % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# 将语音识别得到的文本转写为srt字幕文件
def write_srt(results, srt_file):
    with Path.open(srt_file, "w", encoding="utf-8") as f:
        f.writelines(results)


# 将语音识别得到的文本转写为txt字幕文件
def write_txt(txt_result, txt_file):
    with Path.open(txt_file, "w", encoding="utf-8") as f:
        f.writelines(txt_result)


# 定义一个函数来裁剪音频
def crop_audio(audio_data, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate / 1000)  # 转换为样本数
    end_sample = int(end_time * sample_rate / 1000)  # 转换为样本数
    return audio_data[start_sample:end_sample]


# 模型推理函数
def model_inference(input_wav, language, silence_threshold, fs=16000):
    srt_file = Path(input_wav).with_suffix(".srt")
    txt_file = Path(input_wav).with_suffix(".txt")
    language_abbr = {
        "auto": None,  # 自动识别，whisper全部99种语言均可识别，如果需要指定语言，请自己修改代码
        "zh": "zh",
        "en": "en",
        "ja": "ja",
    }

    language = "auto" if len(language) < 1 else language
    selected_language = language_abbr[language]

    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        if fs != 16000:
            print(f"audio_fs: {fs}")
            resampler = torchaudio.transforms.Resample(fs, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()

    # 加载VAD模型
    vad_model = AutoModel(
        model=vad_model_dir,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        disable_update=True,
        max_end_silence_time=silence_threshold,  # 静音阈值，范围500ms～6000ms，默认值800ms。
    )

    # 使用VAD模型处理音频文件
    vad_res = vad_model.generate(
        input=input_wav,
        cache={},
        max_single_segment_time=30000,  # 最大单个片段时长
    )

    # 从VAD模型的输出中提取每个语音片段的开始和结束时间
    segments = vad_res[0]["value"]  # 假设只有一段音频，且其片段信息存储在第一个元素中

    # 加载原始音频数据
    audio_data, sample_rate = sf.read(input_wav)

    # 对单个语音片段进行处理
    srt_result = ""
    txt_result = []
    srt_id = 1

    prompt_dict = {
        "auto": "",  # auto
        "en": "Tom, There is a Chinese person among them.",
        "zh": "我是一个台湾人，也是一个中国人。",
        "ja": "その中に、一人の日本人がいます。誰だと思いますか？",
    }

    DecodingOptions = {
        "task": "transcribe",
        "language": selected_language,  # zh,en,ja, and None for auto
        "beam_size": None,
        "fp16": True,
        "without_timestamps": True,
        "prompt": prompt_dict.get(selected_language, ""),
    }
    for segment in segments:
        start_time, end_time = segment  # 获取开始和结束时间
        cropped_audio = crop_audio(audio_data, start_time, end_time, sample_rate)

        # 将裁剪后的音频保存到内存中
        with io.BytesIO() as temp_audio_buffer:
            sf.write(temp_audio_buffer, cropped_audio, sample_rate, format="WAV")
            temp_audio_buffer.seek(0)  # 重置缓冲区指针到开头

            # 语音转文字处理
            res = model.generate(
                input=temp_audio_buffer,
                DecodingOptions=DecodingOptions,
                batch_size_s=0,
            )

        # 处理输出结果
        cleaned_text = res[0]["text"]
        srt_result += (
            str(srt_id)
            + "\n"
            + str(reformat_time(start_time / 1000))
            + " --> "
            + str(reformat_time(end_time / 1000))
            + "\n"
            + cleaned_text
            + "\n\n"
        )
        txt_result.append(cleaned_text)
        srt_id += 1
    # 输出结果并保存为srt文件
    write_srt(srt_result, srt_file)
    write_txt(txt_result, txt_file)
    gr.Info(f"音频{Path(input_wav).name}转录完成。")
    return srt_result


# 字幕文件保存到选定文件夹
def save_file(audio_inputs, path_input_text):
    if not Path(path_input_text).is_dir() or path_input_text.strip() == "":
        gr.Warning("请输入有效路径！")
    else:
        try:
            srt_file = Path(audio_inputs).with_suffix(".srt")
            shutil.copy2(srt_file, path_input_text)  # 如果有同名文件会覆盖保存，没有则复制
            txt_file = Path(audio_inputs).with_suffix(".txt")
            shutil.copy2(txt_file, path_input_text)  # 如果有同名文件会覆盖保存，没有则复制
            gr.Info(f"文件{srt_file.name}已保存。")
        except Exception as e:
            gr.Warning(f"保存文件时出错: {e}")


# 多文件转录
def multi_file_asr(multi_files_upload, language, silence_threshold):
    num = 0
    for audio_inputs in multi_files_upload:
        model_inference(audio_inputs, language, silence_threshold, fs=16000)
        num += 1
    gr.Info(f"总共转录{num}个音频，已全部完成")


# 字幕文件保存到选定文件夹
def save_multi_srt(multi_files_upload, path_input_text):
    for audio_inputs in multi_files_upload:
        save_file(audio_inputs, path_input_text)


html_content = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">SenseVoice-Small</h2>
    <p style="font-size: 18px;margin-left: 20px;">SenseVoice-Small 是一种纯编码器语音基础模型，专为快速语音理解而设计</p>
    <p style="margin-left: 20px;"><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice阿里官方GitHub</a>
    <a href="https://github.com/jianchang512/sense-api" target="_blank">Sense-Api仓库</a> 
    <a href="https://github.com/jianchang512/pyvideotrans" target="_blank">pyVideoTrans仓库</a></p>
</div>
"""


def launch():
    with gr.Blocks(theme=gr.themes.Soft(), title="SenseVoice 在线web界面") as demo:
        gr.HTML(html_content)

        with gr.Tab(label="单文件转录"), gr.Column():
            audio_inputs = gr.Audio(label="上传音频或录制麦克风", type="filepath")
            with gr.Accordion("配置"), gr.Row():
                language_inputs = gr.Dropdown(choices=["auto", "zh", "en", "ja"], value="auto", label="说话语言")
                end_silence_time = gr.Slider(
                    label="静音阈值", minimum=0, maximum=6000, step=50, value=800, interactive=True
                )
            with gr.Row():
                stre_btn = gr.Button("开始转录", variant="primary")
                save_btn = gr.Button("保存字幕", variant="primary")
            path_input_text = gr.Text(label="保存路径", interactive=True, placeholder="请输入正确的目标文件夹")
            text_outputs = gr.Textbox(label="识别结果", lines=20)

        stre_btn.click(model_inference, inputs=[audio_inputs, language_inputs, end_silence_time], outputs=text_outputs)

        save_btn.click(save_file, inputs=[audio_inputs, path_input_text], outputs=[])

        with gr.Tab(label="多文件转录"), gr.Column():
            multi_files_upload = gr.File(
                label="上传音频", file_count="directory", file_types=[".mp3", ".wav", ".flac", ".m4a", ".ogg"]
            )
            with gr.Accordion("配置"), gr.Row():
                language_inputs = gr.Dropdown(choices=["auto", "zh", "en", "ja"], value="auto", label="说话语言")
                end_silence_time = gr.Slider(
                    label="静音阈值", minimum=0, maximum=6000, step=50, value=800, interactive=True
                )
            with gr.Row():
                stre_btn = gr.Button("开始转录", variant="primary")
                save_btn = gr.Button("保存字幕", variant="primary")
            path_input_text = gr.Text(label="保存路径", interactive=True, placeholder="请输入正确的目标文件夹")

        stre_btn.click(multi_file_asr, inputs=[multi_files_upload, language_inputs, end_silence_time], outputs=[])

        save_btn.click(save_multi_srt, inputs=[multi_files_upload, path_input_text], outputs=[])

    threading.Thread(target=open_page).start()
    demo.launch(css=".gradio-textbox {font-family: 微软雅黑}")


if __name__ == "__main__":
    launch()
