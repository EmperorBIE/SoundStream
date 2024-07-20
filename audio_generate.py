import os
import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write

# 创建保存音频文件的目录
train_dir = "./data/nsynth-train.jsonwav/nsynth-train/audio"
valid_dir = "./data/nsynth-valid.jsonwav/nsynth-valid/audio"
test_dir = "./data/nsynth-test.jsonwav/nsynth-test/audio"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 生成伪造的音频数据
def generate_sine_wave(freq, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)

sample_rate = 16000  # 16kHz
# durations = [1] * 10
durations = [1, 2, 3, 1.5, 2.5, 3.5, 4, 1.8, 2.8, 3.8]
frequencies = [440, 880, 1760, 660, 1320, 1980, 550, 770, 990, 1210]
# frequencies = [440] * 10

# 生成并保存训练数据
for i, (duration, freq) in enumerate(zip(durations, frequencies)):
    audio = generate_sine_wave(freq, sample_rate, duration)
    file_path = os.path.join(train_dir, f"audio_{i}.wav")
    write(file_path, sample_rate, (audio * 32767).astype(np.int16))  # 保存为16位PCM格式

# 生成并保存验证数据
for i, (duration, freq) in enumerate(zip(durations, frequencies)):
    audio = generate_sine_wave(freq, sample_rate, duration)
    file_path = os.path.join(valid_dir, f"audio_{i}.wav")
    write(file_path, sample_rate, (audio * 32767).astype(np.int16))

# 生成并保存测试数据
for i, (duration, freq) in enumerate(zip(durations, frequencies)):
    audio = generate_sine_wave(freq, sample_rate, duration)
    file_path = os.path.join(test_dir, f"audio_{i}.wav")
    write(file_path, sample_rate, (audio * 32767).astype(np.int16))