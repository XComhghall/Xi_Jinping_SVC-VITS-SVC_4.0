import io
import os

# os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import librosa
import numpy as np
import soundfile
from inference.infer_tool import Svc
import logging

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

config_path = "configs/config.json"

model = Svc("logs/44k/G_40000.pth", "configs/config.json", cluster_model_path="logs/44k/kmeans_10000.pt")



def vc_fn(sid, input_audio, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale):
    if input_audio is None:
        return "请上传音频。", None
    sampling_rate, audio = input_audio
    # print(audio.shape,sampling_rate)
    duration = audio.shape[0] / sampling_rate
    if duration > 90:
        return "请上传小于 90 秒的音频。", None
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    print(audio.shape)
    out_wav_path = "temp.wav"
    soundfile.write(out_wav_path, audio, 16000, format="wav")
    print( cluster_ratio, auto_f0, noise_scale)
    _audio = model.slice_inference(out_wav_path, sid, vc_transform, slice_db, cluster_ratio, auto_f0, noise_scale)
    return "完成。", (44100, _audio)


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Basic"):
            gr.Markdown(value="""
                习主席玉音转换器 线上 demo<br />
                基于 SVC-VITS-SVC 4.0 https://github.com/svc-develop-team/so-vits-svc<br />
                SVC-VITS-SVC 4.0 与 VITS 的不同之处在于，VITS 为文字转语音。SVC-VITS-SVC 为语音转语音，可保留原音调等，适用于转换歌声。

                此企画继承 MIT 许可条款。欢迎再分发及二次创作。我不对此企画的使用附加任何限制。其他限制以 MIT 许可条款为准。<br />
                因此企画一般娱乐，请于 community 报错、提问、讨论，或发帖说明、协商使用安全的方式私下联络。

                鸣谢：<br />
                innnky 原企画作者。<br />
                BOT-666 原技术人员。后失联。<br />
                chika0801 贡献了海量习近平音源。未经授权，数据集不公开。

                Duplicated from 1. WitchHuntTV/XJP_Singing<br />
                via 2. CLTV/WinnieThePoohSVC_sovits4<br />
                via 3. pitaogou/Qingfeng-Sing-sovits4
                """)
            spks = list(model.spk2id.keys())
            sid = gr.Dropdown(label="音色", choices=spks, value=spks[0])
            vc_input3 = gr.Audio(label="上传音频 长度须小于 90 秒。转换长音频请于本地执行。")
            vc_transform = gr.Number(label="变调 半音，整数，可设为负数。高八度即 12。", value=0)
            cluster_ratio = gr.Number(label="聚类模型混合比例 0–1。提升音色相似，但降低咬字准确、清晰。预设为 0，不启用。若使用，建议 0.5 左右。", value=0)
            auto_f0 = gr.Checkbox(label="自动 F0 预测 配合聚类模型效果更好。仅限语音。歌声勿选此项，会究极跑调。会导致变调功能失效。", value=False)
            slice_db = gr.Number(label="切片阈值", value=-40)
            noise_scale = gr.Number(label="Noise scale 建议不要动，会影响音质。玄学参数。", value=0.4)
            vc_submit = gr.Button("转换", variant="primary")
            vc_output1 = gr.Textbox(label="状态")
            vc_output2 = gr.Audio(label="习主席玉音")
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale], [vc_output1, vc_output2])

    app.launch()
