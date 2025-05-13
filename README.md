# IndexTTS Voice Cloning Nodes for ComfyUI

High-quality voice cloning, very fast, supports Chinese and English, and allows custom voice styles.

![image](https://github.com/billwuhao/ComfyUI_IndexTTS/blob/main/images/2025-04-30_19-22-46.png)

## 📣 Updates

[2025-05-02] ⚒️: DeepSpeed acceleration is available, but DeepSpeed needs to be installed. For Windows, please refer to [DeepSpeed Installation](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/windows/08-2024/chinese/README.md). The acceleration is not obvious.

[2025-04-30] ⚒️: Released v1.0.0.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_IndexTTS.git
cd ComfyUI_IndexTTS
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

- Models can be automatically downloaded to `ComfyUI\models\TTS\Index-TTS` folder:
- Recommended for China users. 如果中国用户下载速度慢或无法连接至huggingface.co，可以使用镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

[Index-TTS](https://huggingface.co/IndexTeam/Index-TTS/tree/main) structure as follows:

```
bigvgan_generator.pth
bpe.model
gpt.pth
```

## Acknowledgements

- [index-tts](https://github.com/index-tts/index-tts)

