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
        return "You need to upload an audio", None
    sampling_rate, audio = input_audio
    # print(audio.shape,sampling_rate)
    duration = audio.shape[0] / sampling_rate
    if duration > 90:
        return "请上传小于 90 s 的音频，需要转换长音频请本地进行转换", None
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
    return "Success", (44100, _audio)


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Basic"):
            gr.Markdown(value="""
                Duplicate of CLTV/WinnieThePoohSVC_sovits4 via pitaogou/Qingfeng-Sing-sovits4
                
                「维尼包歌声转换」在线 demo 基于 so-vits-svc 4.0 项目原地址：https://github.com/svc-develop-team/so-vits-svc
                
                so-vits-svc 与 VITS 的不同之处在于，VITS 乃文字转语音，so-vits-svc 为语音转语音，可保留原音调等，适合转换歌声。
                本项目继承 MIT 协议，欢迎再分发及二次创作，我不对该项目的使用做任何附加限制，其他限制以 MIT 协议为准。

                鸣谢人员：innnky（原项目作者）、BOT-666（前技术人员，后失联）、chika0801（贡献了海量习近平音源，因未取得许可，数据集不公开）
                因项目一般娱乐，一般问题请在 Community 内提问，如需私密交流，请先开帖说明来意后协商使用安全的联络手段。
                """)
            spks = list(model.spk2id.keys())
            sid = gr.Dropdown(label="音色", choices=spks, value=spks[0])
            vc_input3 = gr.Audio(label="上传音频（长度小于 90 秒）")
            vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是 12）", value=0)
            cluster_ratio = gr.Number(label="聚类模型混合比例，0–1 之间，默认为 0 不启用聚类，能提升音色相似度，但会导致咬字下降（如果使用建议 0.5 左右）", value=0)
            auto_f0 = gr.Checkbox(label="自动 f0 预测，配合聚类模型 f0 预测效果更好,会导致变调功能失效（仅限转换语音，歌声不要勾选此项会究极跑调）", value=False)
            slice_db = gr.Number(label="切片阈值", value=-40)
            noise_scale = gr.Number(label="noise_scale 建议不要动，会影响音质，玄学参数", value=0.4)
            vc_submit = gr.Button("转换", variant="primary")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform,auto_f0,cluster_ratio, slice_db, noise_scale], [vc_output1, vc_output2])

    app.launch()



