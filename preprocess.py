import torch
import torchaudio
import os
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade

output_dir = "separatedfiles/"
files = []

def separate_sources(model, mix, sample_rate, segment=10.0, overlap=0.1, device=None):
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    audio_channels = model.audio_channels

    if channels != audio_channels:
        if channels < audio_channels:
            padding = torch.zeros(batch, audio_channels - channels, length, device=device)
            mix = torch.cat((mix, padding), dim=1)
        else:
            mix = mix[:, :audio_channels, :]

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), audio_channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0

    return final

def save_audio(audio_tensor, source_name, sample_rate):
    output_path = os.path.join(output_dir, f"{source_name}.wav")
    torchaudio.save(output_path, audio_tensor, sample_rate)

def mixed_sep(audio):
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sample_rate = bundle.sample_rate

    waveform, sample_rate = torchaudio.load(audio)
    waveform = waveform.to(device)

    segment = 10
    overlap = 0.1

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()

    sources = separate_sources(
        model,
        waveform[None],
        sample_rate,
        device=device,
        segment=segment,
        overlap=overlap
    )[0]
    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))

    os.makedirs(output_dir, exist_ok=True)

    for source_name, audio_tensor in audios.items():
        save_audio(audio_tensor, source_name, sample_rate)
        files.append(f"{source_name}.wav")

    return files

# Example usage
# audio_file = 'music\song2.wav'
# separated_files = mixed_sep(audio_file)
# print("Separated files:", separated_files)
