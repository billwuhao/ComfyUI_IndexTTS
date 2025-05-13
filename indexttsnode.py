import os
from subprocess import CalledProcessError
from typing import List
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
from tqdm import tqdm
import folder_paths
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from indextts.BigVGAN.models import BigVGAN as Generator
from indextts.gpt.model import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.utils.front import TextNormalizer, TextTokenizer


models_dir = folder_paths.models_dir
models_path = os.path.join(models_dir, "TTS", "Index-TTS")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch, "mps") and torch.backends.mps.is_available():
    device = "mps"

def statistical_compare(tensor1, tensor2):
    """通过统计特征快速比较"""
    stats1 = {
        'mean': tensor1.mean(),
        'std': tensor1.std(),
        'max': tensor1.max(),
        'min': tensor1.min()
    }
    stats2 = {
        'mean': tensor2.mean(),
        'std': tensor2.std(),
        'max': tensor2.max(),
        'min': tensor2.min()
    }
    return all(torch.allclose(stats1[k], stats2[k], rtol=1e-3) for k in stats1)

class IndexTTS:
    def __init__(
        self, cfg_path=f"{current_dir}/checkpoints/config.yaml", model_dir=models_path, device=device, text_language="zh"):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
        """
        self.device = device
        if device == "cuda":
            self.is_fp16 = True
            self.use_cuda_kernel = True
        else:
            self.is_fp16 = False
            self.use_cuda_kernel = False

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        # Comment-off to load the VQ-VAE model for debugging tokenizer
        #   https://github.com/index-tts/index-tts/issues/34
        #
        # from indextts.vqvae.xtts_dvae import DiscreteVAE
        # self.dvae = DiscreteVAE(**self.cfg.vqvae)
        # self.dvae_path = os.path.join(self.model_dir, self.cfg.dvae_checkpoint)
        # load_checkpoint(self.dvae, self.dvae_path)
        # self.dvae = self.dvae.to(self.device)
        # if self.is_fp16:
        #     self.dvae.eval().half()
        # else:
        #     self.dvae.eval()
        # print(">> vqvae weights restored from:", self.dvae_path)
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        if not os.path.exists(models_path):
            print(f"Downloading Index-TTS model to: {models_path}")
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="IndexTeam/Index-TTS",
                allow_patterns=["bigvgan_generator.pth", "bpe.model", "gpt.pth"],
                local_dir=models_path,
                local_dir_use_symlinks=False,
            )
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        if self.is_fp16:
            try:
                import deepspeed

                use_deepspeed = True
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> DeepSpeed failed to load, fallback to standard inference: {e}")

            self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=True)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load

                anti_alias_activation_cuda = load.load()
                print(">> Preload custom CUDA kernel for BigVGAN", anti_alias_activation_cuda)
            except:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                self.use_cuda_kernel = False

        self.bigvgan = Generator(self.cfg.bigvgan, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan_path = os.path.join(self.model_dir, self.cfg.bigvgan_checkpoint)
        vocoder_dict = torch.load(self.bigvgan_path, map_location="cpu")
        self.bigvgan.load_state_dict(vocoder_dict["generator"])
        self.bigvgan = self.bigvgan.to(self.device)
        # remove weight norm on eval mode
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", self.bigvgan_path)
        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load(lang=text_language)
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)
        # 缓存参考音频mel：
        self.cache_audio_prompt = None
        self.cache_cond_mel = None
        # 进度引用显示（可选）
        self.gr_progress = None

    def clean(self):
        import gc
        self.gpt = None
        self.bigvgan = None
        self.tokenizer = None
        gc.collect()
        self.torch_empty_cache()

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if self.cfg.gpt.stop_mel_token not in code:
                code_lens.append(len(code))
                len_ = len(code)
            else:
                # len_ = code.cpu().tolist().index(8193)+1
                len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                len_ = len_ - 2

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                code = code.cpu().tolist()
                ncode = []
                n = 0
                for k in range(0, len_):
                    if code[k] != silent_token:
                        ncode.append(code[k])
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode.append(code[k])
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                len_ = len(ncode)
                ncode = torch.LongTensor(ncode)
                codes_list.append(ncode.to(device, dtype=dtype))
                isfix = True
                # codes[i] = self.stop_mel_token
                # codes[i, 0:len_] = ncode
            else:
                codes_list.append(codes[i])
            code_lens.append(len_)

        codes = pad_sequence(codes_list, batch_first=True) if isfix else codes[:, :-2]
        code_lens = torch.LongTensor(code_lens).to(device, dtype=dtype)
        return codes, code_lens

    def bucket_sentences(self, sentences, enable=False):
        """
        Sentence data bucketing
        """
        max_len = max(len(s) for s in sentences)
        half = max_len // 2
        outputs = [[], []]
        for idx, sent in enumerate(sentences):
            if enable is False or len(sent) <= half:
                outputs[0].append({"idx": idx, "sent": sent})
            else:
                outputs[1].append({"idx": idx, "sent": sent})
        return [item for item in outputs if item]

    def pad_tokens_cat(self, tokens: List[torch.Tensor]):
        if len(tokens) <= 1:
            return tokens[-1]
        max_len = max(t.size(1) for t in tokens)
        outputs = []
        for tensor in tokens:
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                n = min(8, pad_len)
                tensor = torch.nn.functional.pad(tensor, (0, n), value=self.cfg.gpt.stop_text_token)
                tensor = torch.nn.functional.pad(tensor, (0, pad_len - n), value=self.cfg.gpt.start_text_token)
            tensor = tensor[:, :max_len]
            outputs.append(tensor)
        tokens = torch.cat(outputs, dim=0)
        return tokens

    def torch_empty_cache(self):
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception as e:
            pass

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    # 快速推理：对于“多句长文本”，可实现至少 2~10 倍以上的速度提升~ （First modified by sunnyboxs 2025-04-16）
    def infer_fast(self, audio_prompt, text, top_k=30, top_p=0.8, temperature=1.0, max_mel_tokens=600, bucket_enable=True, verbose=False):
        print(">> start fast inference...")
        self._set_gr_progress(0, "start fast inference...")
        if verbose:
            print(f"origin text:{text}")

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        audio, sr = audio_prompt["waveform"].squeeze(0), audio_prompt["sample_rate"]
        if self.cache_cond_mel is None or not statistical_compare(self.cache_audio_prompt, audio):
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)
            cond_mel_frame = cond_mel.shape[-1]
            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            cond_mel_frame = cond_mel.shape[-1]
            pass

        auto_conditioning = cond_mel
        cond_mel_lengths = torch.tensor([cond_mel_frame], device=self.device)

        # text_tokens
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list)
        if verbose:
            print("text token count:", len(text_tokens_list))
            print("sentences count:", len(sentences))
            print(*sentences, sep="\n")

        autoregressive_batch_size = 1
        length_penalty = 0.0
        num_beams = 3
        repetition_penalty = 10.0
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []

        # text processing
        all_text_tokens: List[List[torch.Tensor]] = []
        self._set_gr_progress(0.1, "text processing...")
        # bucket_enable 预分桶开关，优先保证质量=True。优先保证速度=False。
        all_sentences = self.bucket_sentences(sentences, enable=bucket_enable) 
        for sentences in all_sentences:
            temp_tokens: List[torch.Tensor] = []
            all_text_tokens.append(temp_tokens)
            for item in sentences:
                sent = item["sent"]
                text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
                text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
                if verbose:
                    print(text_tokens)
                    print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                    # debug tokenizer
                    text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                    print("text_token_syms is same as sentence tokens", text_token_syms == sent) 
                temp_tokens.append(text_tokens)
            
        # Sequential processing of bucketing data
        all_batch_num = 0
        all_batch_codes = []
        for item_tokens in all_text_tokens:
            batch_num = len(item_tokens)
            batch_text_tokens = self.pad_tokens_cat(item_tokens)
            batch_cond_mel_lengths = torch.cat([cond_mel_lengths] * batch_num, dim=0)
            batch_auto_conditioning = torch.cat([auto_conditioning] * batch_num, dim=0)
            all_batch_num += batch_num

            # gpt speech
            self._set_gr_progress(0.2, "gpt inference speech...")
            with torch.no_grad():
                with torch.amp.autocast(batch_text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    temp_codes = self.gpt.inference_speech(batch_auto_conditioning, batch_text_tokens,
                                        cond_mel_lengths=batch_cond_mel_lengths,
                                        # text_lengths=text_len,
                                        do_sample=True,
                                        top_p=top_p,
                                        top_k=top_k,
                                        temperature=temperature,
                                        num_return_sequences=autoregressive_batch_size,
                                        length_penalty=length_penalty,
                                        num_beams=num_beams,
                                        repetition_penalty=repetition_penalty,
                                        max_generate_length=max_mel_tokens)
                    all_batch_codes.append(temp_codes)

        # gpt latent
        self._set_gr_progress(0.5, "gpt inference latents...")
        all_idxs = []
        all_latents = []
        for batch_codes, batch_tokens, batch_sentences in zip(all_batch_codes, all_text_tokens, all_sentences):
            for i in range(batch_codes.shape[0]):
                codes = batch_codes[i]  # [x]
                codes = codes[codes != self.cfg.gpt.stop_mel_token]
                codes, _ = torch.unique_consecutive(codes, return_inverse=True)
                codes = codes.unsqueeze(0)  # [x] -> [1, x]
                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                text_tokens = batch_tokens[i]
                all_idxs.append(batch_sentences[i]["idx"])
                with torch.no_grad():
                    with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                        latent = \
                            self.gpt(auto_conditioning, text_tokens,
                                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                        code_lens*self.gpt.mel_length_compression,
                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                        return_latent=True, clip_inputs=False)
                        all_latents.append(latent)

        # bigvgan chunk
        chunk_size = 2
        all_latents = [all_latents[all_idxs.index(i)] for i in range(len(all_latents))]
        chunk_latents = [all_latents[i : i + chunk_size] for i in range(0, len(all_latents), chunk_size)]
        chunk_length = len(chunk_latents)
        latent_length = len(all_latents)
        all_latents = None

        # bigvgan chunk decode
        self._set_gr_progress(0.7, "bigvgan decode...")
        tqdm_progress = tqdm(total=latent_length, desc="bigvgan")
        for items in chunk_latents:
            tqdm_progress.update(len(items))
            latent = torch.cat(items, dim=1)
            with torch.no_grad():
                with torch.amp.autocast(latent.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    wav = wav.squeeze(1)
                    pass
            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            wavs.append(wav)

        # clear cache
        tqdm_progress.close()  # 确保进度条被关闭
        chunk_latents.clear()
        self.torch_empty_cache()

        # wav audio output
        self._set_gr_progress(0.9, "save audio...")
        wav = torch.cat(wavs, dim=1)

        # save audio
        wav = wav / 32768.0
        wav = wav.cpu().float()  # to cpu
        return {"waveform": wav.unsqueeze(0), "sample_rate": sampling_rate}

    # 原始推理模式
    def infer(self, audio_prompt, text, top_p=0.8, top_k=30, temperature=1.0, max_mel_tokens=600, verbose=False):
        print(">> start inference...")
        self._set_gr_progress(0, "start inference...")
        if verbose:
            print(f"origin text:{text}")

        # 如果参考音频改变了，才需要重新生成 cond_mel, 提升速度
        audio, sr = audio_prompt["waveform"].squeeze(0), audio_prompt["sample_rate"]
        if self.cache_cond_mel is None or not statistical_compare(self.cache_audio_prompt, audio):
            audio = torch.mean(audio, dim=0, keepdim=True)
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            cond_mel = MelSpectrogramFeatures()(audio).to(self.device)

            if verbose:
                print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

            self.cache_audio_prompt = audio
            self.cache_cond_mel = cond_mel
        else:
            cond_mel = self.cache_cond_mel
            pass

        auto_conditioning = cond_mel
        text_tokens_list = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text_tokens_list)
        if verbose:
            print("text token count:", len(text_tokens_list))
            print("sentences count:", len(sentences))
            print(*sentences, sep="\n")

        autoregressive_batch_size = 1
        length_penalty = 0.0
        num_beams = 3
        repetition_penalty = 10.0
        sampling_rate = 24000
        # lang = "EN"
        # lang = "ZH"
        wavs = []

        for sent in sentences:
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            # text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
            # text_tokens = F.pad(text_tokens, (1, 0), value=0)
            # text_tokens = F.pad(text_tokens, (0, 1), value=1)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as sentence tokens", text_token_syms == sent)

            # text_len = torch.IntTensor([text_tokens.size(1)], device=text_tokens.device)
            # print(text_len)

            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    codes = self.gpt.inference_speech(auto_conditioning, text_tokens,
                                                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]],
                                                                                      device=text_tokens.device),
                                                        # text_lengths=text_len,
                                                        do_sample=True,
                                                        top_p=top_p,
                                                        top_k=top_k,
                                                        temperature=temperature,
                                                        num_return_sequences=autoregressive_batch_size,
                                                        length_penalty=length_penalty,
                                                        num_beams=num_beams,
                                                        repetition_penalty=repetition_penalty,
                                                        max_generate_length=max_mel_tokens)

                # codes = codes[:, :-2]
                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                if verbose:
                    print(codes, type(codes))
                    print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                # remove ultra-long silence if exits
                # temporarily fix the long silence bug.
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                # latent, text_lens_out, code_lens_out = \
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = \
                        self.gpt(auto_conditioning, text_tokens,
                                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                    code_lens*self.gpt.mel_length_compression,
                                    cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                    return_latent=True, clip_inputs=False)

                    wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav)

        wav = torch.cat(wavs, dim=1)

        # save audio
        wav = wav / 32768.0
        wav = wav.cpu().float()  # to cpu
        return {"waveform": wav.unsqueeze(0), "sample_rate": sampling_rate}

class IndexTTSRun:
    def __init__(self):
        self.index_tts = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_prompt":("AUDIO",),
                "text": ("STRING", {"forceInput": True}),
                "text_language": (["zh", "en"], {"default": "zh"}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_mel_tokens": ("INT", {"default": 1000, "min": 0, "max": 100000, "step": 1}),
                "bucket_enable": ("BOOLEAN", {"default": True}),
                "fast_inference": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "🎤MW/MW-IndexTTS"

    def clone(self, 
        audio_prompt, 
        text, 
        text_language,
        top_k=30, 
        top_p=0.8, 
        temperature=1.0, 
        max_mel_tokens=600, 
        bucket_enable=True, 
        fast_inference=True, 
        unload_model=True
        ):
        if self.index_tts is None:
            self.index_tts = IndexTTS(text_language=text_language)

        if fast_inference:
            res = self.index_tts.infer_fast(
                audio_prompt, 
                text, 
                top_p=top_p, 
                top_k=top_k, 
                temperature=temperature, 
                max_mel_tokens=max_mel_tokens, 
                bucket_enable=bucket_enable)
        else:
            res = self.index_tts.infer(
                audio_prompt, 
                text, 
                top_p=top_p, 
                top_k=top_k, 
                temperature=temperature, 
                max_mel_tokens=max_mel_tokens)

        if unload_model:
            self.index_tts.clean()
            self.index_tts = None
            torch.cuda.empty_cache()

        return (res,)


class MultiLinePromptIndex:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "multi_line_prompt": ("STRING", {
                    "multiline": True, 
                    "default": ""}),
                },
        }

    CATEGORY = "🎤MW/MW-IndexTTS"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "promptgen"
    
    def promptgen(self, multi_line_prompt: str):
        return (multi_line_prompt.strip(),)


NODE_CLASS_MAPPINGS = {
    "IndexTTSRun": IndexTTSRun,
    "MultiLinePromptIndex": MultiLinePromptIndex,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTSRun": "IndexTTS Run",
    "MultiLinePromptIndex": "Multi Line Prompt",
}