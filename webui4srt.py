# coding=utf-8

import shutil
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf  # 用于读取和裁剪音频文件
import torch
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 模型路径
model_dir = "iic/SenseVoiceSmall"
vad_model_dir = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"  # VAD模型路径


# 首先加载VAD模型
vad_model = AutoModel(
    model=vad_model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:0",
    disable_update=True,
)


def open_page():
    import webbrowser

    time.sleep(5)
    webbrowser.open_new_tab("http://127.0.0.1:7860")


# 定义时间戳格式
def reformat_time(second):
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    hms = "%02d:%02d:%s" % (h, m, str("%.3f" % s).zfill(6))
    hms = hms.replace(".", ",")
    return hms


# 将语音识别得到的文本转写为srt字幕文本
def write_srt(results, srt_file):
    with open(srt_file, "w", encoding="utf-8") as f:
        id = 1
        for segment in results:
            f.writelines(
                str(id)
                + "\n"
                + reformat_time(segment["start"])
                + " --> "
                + reformat_time(segment["end"])
                + "\n"
                + segment["text"]
                + "\n\n"
            )
            id += 1


# 定义一个函数来裁剪音频
def crop_audio(audio_data, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate / 1000)  # 转换为样本数
    end_sample = int(end_time * sample_rate / 1000)  # 转换为样本数
    return audio_data[start_sample:end_sample]


def model_inference(input_wav, language, fs=16000):
    temp_path = Path(input_wav).parent
    srt_file = Path(input_wav).with_suffix(".srt")
    # task_abbr = {"Speech Recognition": "ASR", "Rich Text Transcription": ("ASR", "AED", "SER")}
    language_abbr = {
        "auto": "auto",
        "zh": "zh",
        "en": "en",
        "yue": "yue",
        "ja": "ja",
        "ko": "ko",
        "nospeech": "nospeech",
    }

    # task = "Speech Recognition" if task is None else task
    language = "auto" if len(language) < 1 else language
    selected_language = language_abbr[language]
    # selected_task = task_abbr.get(task)

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

    # 使用VAD模型处理音频文件
    vad_res = vad_model.generate(
        input=input_wav,
        cache={},
        max_single_segment_time=30000,  # 最大单个片段时长
    )

    # 加载SenseVoice模型
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cuda:0",
        disable_update=True,
    )
    # 从VAD模型的输出中提取每个语音片段的开始和结束时间
    segments = vad_res[0]["value"]  # 假设只有一段音频，且其片段信息存储在第一个元素中

    # 加载原始音频数据
    audio_data, sample_rate = sf.read(input_wav)

    # 对每个语音片段进行处理
    results = []
    for segment in segments:
        start_time, end_time = segment  # 获取开始和结束时间
        cropped_audio = crop_audio(audio_data, start_time, end_time, sample_rate)

        # 将裁剪后的音频保存为临时文件
        temp_audio_file = f"{temp_path}/temp_cropped.wav"
        sf.write(temp_audio_file, cropped_audio, sample_rate)

        # 语音转文字处理
        res = model.generate(
            input=temp_audio_file,
            cache={},
            language=selected_language,  # 自动检测语言
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  # 启用 VAD 断句
            merge_length_s=10000,  # 合并长度，单位为毫秒
        )
        # 处理输出结果
        text = rich_transcription_postprocess(res[0]["text"])
        # 添加时间戳
        # results.append(
        #     {"start": start_time / 1000, "end": end_time / 1000, "text": text}
        # )  # 转换为秒
        results.append(
            {
                "start": start_time / 1000,
                "end": end_time / 1000,
                "text": text.replace(" ", ""),
            }
        )
        # 输出结果   保存为srt文件
    write_srt(results, srt_file)
    gr.Info("音频转录完成。")
    return srt_file


def display_srt(srt_file):
    with open(srt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


# 字幕文件保存到选定文件夹
def save_file(srt_file, path_input_text):
    if not Path(path_input_text).exists():
        gr.Warning("请输入有效路径！")
    else:
        try:
            shutil.copy2(srt_file, path_input_text)  # 如果有同名文件会覆盖保存，没有则复制
            gr.Info("文件已保存。")
        except Exception as e:
            gr.Warning(f"保存文件时出错: {e}")


# 多文件转录
def multi_file_asr(multi_files_upload, language):
    num = 1
    for audio_inputs in enumerate(multi_files_upload, start=1):
        model_inference(audio_inputs, language, fs=16000)
        num += 1
    gr.Info(f"总共转录{num}个音频，已全部完成")


# 字幕文件保存到选定文件夹
def save_multi_srt(multi_files_upload, path_input_text):
    if not Path(path_input_text).exists():
        gr.Warning("请输入有效路径！")
    else:
        try:
            for audio_inputs in multi_files_upload:
                srt_file = Path(audio_inputs).with_suffix(".srt")
                shutil.copy2(srt_file, path_input_text)  # 如果有同名文件会覆盖保存，没有则复制
                gr.Info("文件已保存。")
        except Exception as e:
            gr.Warning(f"保存文件时出错: {e}")


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

        with gr.Tab(label="单文件转录"):
            with gr.Column():
                audio_inputs = gr.Audio(label="上传音频或录制麦克风", type="filepath")
                with gr.Accordion("配置"):
                    language_inputs = gr.Dropdown(
                        choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
                        value="auto",
                        label="说话语言",
                    )
                with gr.Row():
                    stre_btn = gr.Button("开始转录", variant="primary")
                    save_btn = gr.Button("保存字幕", variant="primary")
                path_input_text = gr.Text(
                    label="保存路径",
                    interactive=True,
                    placeholder="请输入正确的目标文件夹",
                )
                text_outputs = gr.Textbox(label="识别结果")
                srt_reg = gr.Textbox(label="信息传递", visible=False)  # 只用于传递转录完成的信息

        stre_btn.click(
            model_inference,
            inputs=[audio_inputs, language_inputs],
            outputs=srt_reg,
        )

        srt_reg.change(
            display_srt,
            inputs=srt_reg,
            outputs=text_outputs,
        )

        save_btn.click(
            save_file,
            inputs=[srt_reg, path_input_text],
            outputs=[],
        )

        with gr.Tab(label="多文件转录"):
            with gr.Column():
                multi_files_upload = gr.File(label="上传音频", file_count="directory", file_types=[".mp3"])
                with gr.Accordion("配置"):
                    language_inputs = gr.Dropdown(
                        choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
                        value="auto",
                        label="说话语言",
                    )
                with gr.Row():
                    stre_btn = gr.Button("开始转录", variant="primary")
                    save_btn = gr.Button("保存字幕", variant="primary")
                path_input_text = gr.Text(
                    label="保存路径",
                    interactive=True,
                    placeholder="请输入正确的目标文件夹",
                )

        stre_btn.click(
            multi_file_asr,
            inputs=[multi_files_upload, language_inputs],
            outputs=[],
        )

        save_btn.click(
            save_multi_srt,
            inputs=[multi_files_upload, path_input_text],
            outputs=[],
        )

    threading.Thread(target=open_page).start()
    demo.launch()


if __name__ == "__main__":
    launch()
