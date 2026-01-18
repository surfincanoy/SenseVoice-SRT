import shutil
import threading
import webbrowser
from pathlib import Path

import emoji
import gradio as gr
import numpy as np
import soundfile as sf  # 用于读取和裁剪音频文件
import torch
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# VAD模型路径
vad_model_dir = "fsmn-vad"

# ASR模型路径
asr_models = {"SenseVoice": "iic/SenseVoiceSmall", "Whisper": "iic/Whisper-large-v3-turbo"}

# 语言选项
language_options = {
    "SenseVoice": ["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
    "Whisper": ["auto", "zh", "en", "ja"],
}

# 加载的模型缓存
loaded_models = {}


def get_model(model_name):
    if model_name not in loaded_models:
        if model_name == "SenseVoice":
            loaded_models[model_name] = AutoModel(
                model=asr_models[model_name],
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                disable_update=True,
            )
        elif model_name == "Whisper":
            loaded_models[model_name] = AutoModel(
                model=asr_models[model_name],
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                vad_kwargs={"max_single_segment_time": 30000},
                disable_update=True,
            )
    return loaded_models[model_name]


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
def write_srt(srt_result, srt_file):
    with Path.open(srt_file, "w", encoding="utf-8") as f:
        f.writelines(srt_result)


# 将语音识别得到的文本转写为txt字幕文件
def write_txt(txt_result, txt_file):
    with Path.open(txt_file, "w", encoding="utf-8") as f:
        f.writelines(txt_result)


def cut_wav_to_ndarray(wav_path: str, start_s: float, end_s: float) -> np.ndarray:
    if end_s <= start_s:
        raise ValueError("end_s must be > start_s")

    with sf.SoundFile(wav_path) as f:
        sr = f.samplerate
        channels = f.channels
        start_frame = max(0, int(start_s / 1000 * sr))
        end_frame = max(start_frame + 1, int(end_s / 1000 * sr))
        frames = end_frame - start_frame
        f.seek(start_frame)
        audio = f.read(frames, dtype="float32")  # 读取为 float32
    # 如果是立体声，转换为单声道
    if channels > 1:
        audio = np.mean(audio, axis=1)  # 取平均
    if sr != 16000:
        # 转换为张量，重采样，转回 NumPy
        audio_tensor = torch.tensor(audio)
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio_tensor).numpy()
    return audio


# 模型推理函数
def model_inference(input_wav, model_name, language, silence_threshold):
    srt_file = Path(input_wav).with_suffix(".srt")
    txt_file = Path(input_wav).with_suffix(".txt")

    asr_model = get_model(model_name)

    if model_name == "SenseVoice":
        language_abbr = {
            "auto": "auto",
            "zh": "zh",
            "en": "en",
            "yue": "yue",
            "ja": "ja",
            "ko": "ko",
            "nospeech": "nospeech",
        }
        selected_language = language_abbr.get(language, "auto")
    elif model_name == "Whisper":
        language_abbr = {
            "auto": None,
            "zh": "zh",
            "en": "en",
            "ja": "ja",
        }
        selected_language = language_abbr.get(language, None)

    # 加载VAD模型
    vad_model = AutoModel(
        model=vad_model_dir,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        disable_update=True,
        max_end_silence_time=silence_threshold,
    )

    # 使用VAD模型处理音频文件
    vad_res = vad_model.generate(
        input=input_wav,
        cache={},
        max_single_segment_time=30000,
    )

    segments = vad_res[0]["value"]

    srt_result = ""
    txt_result = []
    srt_id = 1

    for segment in segments:
        start_time, end_time = segment
        audio_temp = cut_wav_to_ndarray(input_wav, start_time, end_time)

        if model_name == "SenseVoice":
            res = asr_model.generate(
                input=audio_temp,
                cache={},
                language=selected_language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
                ban_emo_unk=False,
            )
            cleaned_text = rich_transcription_postprocess(res[0]["text"])
            cleaned_text = emoji.replace_emoji(cleaned_text, replace="")
            if selected_language not in ["en", "ko"]:
                cleaned_text = cleaned_text.replace(" ", "").strip()
        elif model_name == "Whisper":
            prompt_dict = {
                "auto": "",
                "en": "Tom, There is a Chinese person among them.",
                "zh": "我是一个台湾人，也是一个中国人。",
                "ja": "その中に、一人の日本人がいます。誰だと思いますか？",
            }
            DecodingOptions = {
                "task": "transcribe",
                "language": selected_language,
                "beam_size": None,
                "fp16": True,
                "without_timestamps": True,
                "prompt": prompt_dict.get(language, ""),
            }
            res = asr_model.generate(
                input=audio_temp,
                DecodingOptions=DecodingOptions,
                batch_size_s=0,
            )
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
            txt_file = Path(audio_inputs).with_suffix(".txt")
            shutil.copy2(srt_file, path_input_text)
            shutil.copy2(txt_file, path_input_text)
            gr.Info(f"转录结果：{srt_file.name}和{txt_file.name}已保存到指定目录。")
        except Exception as e:
            gr.Warning(f"保存文件时出错: {e}")


# 多文件转录
def multi_file_asr(multi_files_upload, model_name, language, silence_threshold):
    num = 0
    for audio_inputs in multi_files_upload:
        model_inference(audio_inputs, model_name, language, silence_threshold)
        num += 1
    gr.Info(f"总共转录{num}个音频，已全部完成")


# 字幕文件保存到选定文件夹
def save_multi_srt(multi_files_upload, path_input_text):
    for audio_inputs in multi_files_upload:
        save_file(audio_inputs, path_input_text)


html_content = """
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">SenseVoice & Whisper</h2>
    <p style="font-size: 18px;margin-left: 20px;">SenseVoice-Small 和 Whisper 语音基础模型，用于快速语音理解</p>
    <p style="margin-left: 20px;"><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice阿里官方GitHub</a>
    <a href="https://github.com/jianchang512/sense-api" target="_blank">Sense-Api仓库</a>
    <a href="https://github.com/jianchang512/pyvideotrans" target="_blank">pyVideoTrans仓库</a></p>
</div>
"""


def update_language_options(model_name):
    return gr.Dropdown(choices=language_options[model_name], value="auto", label="说话语言")


def launch():
    with gr.Blocks(theme=gr.themes.Soft(), title="SenseVoice & Whisper 在线web界面") as demo:
        gr.HTML(html_content)

        model_selector = gr.Radio(choices=["SenseVoice", "Whisper"], value="SenseVoice", label="选择ASR模型")

        with gr.Tab(label="单文件转录"), gr.Column():
            audio_inputs = gr.Audio(label="上传音频或录制麦克风", type="filepath")
            with gr.Accordion("配置"), gr.Row():
                language_inputs = gr.Dropdown(choices=language_options["SenseVoice"], value="auto", label="说话语言")
                end_silence_time = gr.Slider(
                    label="静音阈值", minimum=0, maximum=6000, step=50, value=800, interactive=True
                )
            with gr.Row():
                stre_btn = gr.Button("开始转录", variant="primary")
                save_btn = gr.Button("保存字幕", variant="primary")
            path_input_text = gr.Text(label="保存路径", interactive=True, placeholder="请输入正确的目标文件夹")
            text_outputs = gr.Textbox(label="识别结果", lines=20)

        model_selector.change(update_language_options, inputs=model_selector, outputs=language_inputs)

        stre_btn.click(
            model_inference,
            inputs=[audio_inputs, model_selector, language_inputs, end_silence_time],
            outputs=text_outputs,
        )

        save_btn.click(save_file, inputs=[audio_inputs, path_input_text], outputs=[])

        with gr.Tab(label="多文件转录"), gr.Column():
            multi_files_upload = gr.File(
                label="上传音频", file_count="directory", file_types=[".mp3", ".wav", ".flac", ".m4a", ".ogg"]
            )
            with gr.Accordion("配置"), gr.Row():
                language_inputs_multi = gr.Dropdown(
                    choices=language_options["SenseVoice"], value="auto", label="说话语言"
                )
                end_silence_time_multi = gr.Slider(
                    label="静音阈值", minimum=0, maximum=6000, step=50, value=800, interactive=True
                )
            with gr.Row():
                stre_btn_multi = gr.Button("开始转录", variant="primary")
                save_btn_multi = gr.Button("保存字幕", variant="primary")
            path_input_text_multi = gr.Text(label="保存路径", interactive=True, placeholder="请输入正确的目标文件夹")

        model_selector.change(update_language_options, inputs=model_selector, outputs=language_inputs_multi)

        stre_btn_multi.click(
            multi_file_asr,
            inputs=[multi_files_upload, model_selector, language_inputs_multi, end_silence_time_multi],
            outputs=[],
        )

        save_btn_multi.click(save_multi_srt, inputs=[multi_files_upload, path_input_text_multi], outputs=[])

    threading.Thread(target=open_page).start()
    demo.launch(css=".gradio-textbox {font-family: 微软雅黑}")


if __name__ == "__main__":
    launch()
