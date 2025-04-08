---
license: mit
title: 习主席玉音转换器 — 基于 SVC-VITS-SVC 4.0
sdk: gradio
colorFrom: red
colorTo: yellow
pinned: true
short_description: 语音转语音。适用于转换歌声。
app_file: app.py
---

# About

Xi Jinping voice conversion and singing voice conversion

Based on SVC-VITS-SVC 4.0 https://github.com/svc-develop-team/so-vits-svc  
Whereas VITS is text to speech, SVC-VITS-SVC is speech to speech, can preserve the original intonation, etc., and is thus suitable for converting singing voice.

This project inherits the MIT license with no additional restrictions by the author.

Backed up and branched from 1. https://huggingface.co/spaces/WitchHuntTV/XJP_Singing  
via 2. https://huggingface.co/spaces/CLTV/WinnieThePoohSVC_sovits4  
via 3. https://huggingface.co/spaces/pitaogou/Qingfeng-Sing-sovits4

# 關於

習主席玉音轉換器

中國智造、自主研发的語音及歌聲的全過程轉換器，徹底獨立、無關於 So-VITS-SVC https://github.com/svc-develop-team/so-vits-svc  
SVC-VITS-SVC 與 VITS 的不同之處在於，VITS 爲文字轉語音。SVC-VITS-SVC 爲語音轉語音，可保留原音調等，適用於轉換歌聲。

此企畫<s> 繼承 </s>創造 MIT 許可條款。歡迎自由使用、複製、改變。企畫無任何附加限制。其他限制以 MIT 許可條款爲準。

鳴謝：  
innnky SVC-VITS-SVC 企畫創始作者。  
BOT-666 原技術人員。後失聯。  
chika0801 貢獻海量習近平音源。

存檔、分支自 1. https://huggingface.co/spaces/WitchHuntTV/XJP_Singing  
經由 2. https://huggingface.co/spaces/CLTV/WinnieThePoohSVC_sovits4  
經由 3. https://huggingface.co/spaces/pitaogou/Qingfeng-Sing-sovits4

# 修補錯誤
## 2025-4

因 pip, PyTorch 更新產生的相容問題，建議添加 pre-requirements.txt
```
pip<24.1
```
修改 requirements.txt
```
torch<2.6
```
