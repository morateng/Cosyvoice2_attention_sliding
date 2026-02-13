import sys
sys.path.append('third_party/Matcha-TTS')

import torch
torch.backends.cuda.preferred_blas_library("cublaslt")

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

def main():
    cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')
    cosyvoice.model.llm.attention_window_size = None  # None으로 설정하면 sliding window 비활성화

    # zero-shot: 프롬프트 음성의 목소리로 원하는 텍스트를 생성
    chunks = []
    for j in cosyvoice.inference_zero_shot(
        """In our previous work, we introduced CosyVoice, a multilingual speech synthesis model based on supervised discrete speech tokens. By employing progressive semantic decoding with two popular generative models, language models (LMs) and
Flow Matching, CosyVoice demonstrated high prosody naturalness, content consistency, and speaker similarity in speech in-context learning. Recently, significant progress has been made in multi-modal large language models (LLMs), where the response latency and real-time factor of speech synthesis play a crucial role in the interactive experience. Therefore, in this report, we present an improved streaming speech synthesis model, CosyVoice 2, which incorporates comprehensive and
systematic optimizations. Specifically, we introduce finite-scalar quantization to improve the codebook utilization of speech token""",
        '希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav',
        stream=False,
    ):
        chunks.append(j['tts_speech'])
    audio = torch.cat(chunks, dim=-1)
    torchaudio.save('output_attention_window_none.wav', audio, cosyvoice.sample_rate)
    print('output_attention_window_none.wav 저장 완료')

if __name__ == '__main__':
    main()