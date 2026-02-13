import sys
sys.path.append('third_party/Matcha-TTS')

import torch
torch.backends.cuda.preferred_blas_library("cublaslt")

from cosyvoice.cli.cosyvoice import AutoModel
import matplotlib.pyplot as plt

cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

TEXT = """In our previous work, we introduced CosyVoice, a multilingual speech synthesis
model based on supervised discrete speech tokens. By employing progressive se-
mantic decoding with two popular generative models, language models (LMs) and
Flow Matching , CosyVoice demonstrated high prosody naturalness, content con
sistency, and speaker similarity in speech in-context learning. Recently, significant progress has been made in multi-modal large language models (LLMs), where the response latency and real-time factor of speech synthesis play a crucial role in the interactive experience.
"""

PROMPT_TEXT = '希望你以后能够做的比我还好呦。'
PROMPT_WAV = './asset/zero_shot_prompt.wav'

results = {}

for label, window in [('sliding_window_32', 32), ('full_attention', None)]:
    cosyvoice.model.llm.attention_window_size = window
    cosyvoice.model.llm._token_times = []
    # actual run
    for _ in cosyvoice.inference_zero_shot(TEXT, PROMPT_TEXT, PROMPT_WAV, stream=False):
        pass
    results[label] = list(cosyvoice.model.llm._token_times)
    print(f'{label}: {len(results[label])} tokens, avg {sum(results[label])/len(results[label]):.4f}s/token')

# plot — skip first 5 tokens (warmup noise), add moving average
import numpy as np
SKIP = 5
WINDOW = 10

fig, ax = plt.subplots(figsize=(10, 5))
for label, times in results.items():
    t = times[SKIP:]
    x = np.arange(SKIP, SKIP + len(t))
    ax.plot(x, t, alpha=0.25)
    ma = np.convolve(t, np.ones(WINDOW) / WINDOW, mode='valid')
    ax.plot(np.arange(SKIP, SKIP + len(ma)), ma, label=label, linewidth=2)

all_vals = [v for t in results.values() for v in t[SKIP:]]
margin = (max(all_vals) - min(all_vals)) * 0.1
ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

ax.set_xlabel('Token index')
ax.set_ylabel('Time per token (s)')
ax.set_title('Per-token latency: Sliding Window vs Full Attention')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('benchmark4.png', dpi=150)
print('saved benchmark4.png')
