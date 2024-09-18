---
layout: post
author: bookstall
tags: AI
categories: [AI]
description: AutoCut：基于 Whisper 的视频剪辑工具
keywords: AI
title: AutoCut：基于 Whisper 的视频剪辑工具
mathjax: true
---

## OpenAI's Whisper

<a href="https://raw.githubusercontent.com/openai/whisper/main/approach.png" data-caption="Whisper 总体架构图">
<img src="https://raw.githubusercontent.com/openai/whisper/main/approach.png" alt="Whisper 总体架构图" style="zoom: 20%;">
</a>


### 安装

```shell
pip install -U openai-whisper
```

同时还需要安装 ffmpeg 工具：

```shell
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

### 支持的模型和语言

<a href="https://raw.githubusercontent.com/openai/whisper/main/language-breakdown.svg" data-fancybox="images" data-caption="Whisper 的 WER (Word Error Rate) 结果">
<img src="https://raw.githubusercontent.com/openai/whisper/main/language-breakdown.svg" alt="Whisper 的 WER (Word Error Rate) 结果" style="zoom: 67%;">
</a>



### Python 用法

```python
import whisper

# 加载模型，这里是 base 模型
model = whisper.load_model("base")

# 转录
# audio.mp3
result = model.transcribe("./九转大肠最全前因后果（补档）.mp4")

# 打印转录结果
print(result["text"])
```

`transcribe()` 方法读取整个视频文件，并且使用 **大小为 30 秒 的滑动窗口** 来处理音频，对每个窗口执行自回归的序列到序列（sequence-to-sequence）的预测。



## AutoCut：通过字幕来剪切视频

AutoCut 对你的视频自动生成字幕。然后你选择需要保留的句子，AutoCut 将对你视频中对应的片段裁切并保存。你无需使用视频编辑软件，只需要编辑文本文件即可完成剪切。

<a href="https://github.com/mli/autocut/raw/main/imgs/typora.jpg" data-fancybox="images" data-caption="AutoCut 示例">
<img src="https://github.com/mli/autocut/raw/main/imgs/typora.jpg" alt="AutoCut 示例" style="zoom: 33%;">
</a>

### 安装

```shell
pip install git+https://github.com/mli/autocut.git
```

AutoCut 默认需要使用 **Python 3.9** 及以上版本，因为只有 Python 3.9 及以上版本才支持 `argparse.BooleanOptionalAction`。如果要使用 Python 3.9 以下的版本（这里我尝试的是 python 3.8），需要进行如下调整：

- 在 `setup.cfg` 中对 `python_requires = >= 3.9` 进行修改

- 在 `main.py` 中将所有的 `argparse.BooleanOptionalAction` 进行注释

在进行 AutoCut 的安装时，还可能会遇到无法安装 `whisper`，进而无法完成 AutoCut 安装的情况，下面是报错的输出：

```shell
WARNING: Generating metadata for package whisper produced metadata for project name openai-whisper. Fix your #egg=whisper fragments.

Discarding git+https://github.com/openai/whisper.git: Requested openai-whisper from git+https://github.com/openai/whisper.git (from autocut==0.0.3) has inconsistent name: expected 'whisper', but metadata has 'openai-whisper'

ERROR: Could not find a version that satisfies the requirement whisper (unavailable) (from autocut) (from versions: 0.9.5, 0.9.6, 0.9.7, 0.9.8, 0.9.9, 0.9.10, 0.9.11, 0.9.12, 0.9.13, 0.9.14, 0.9.15, 0.9.16, 1.0.0, 1.0.1, 1.0.2, 1.1.0, 1.1.1, 1.1.2, 1.1.3, 1.1.4, 1.1.5, 1.1.6, 1.1.7, 1.1.8, 1.1.9, 1.1.10)

ERROR: No matching distribution found for whisper (unavailable)
```

根据报错输出的信息，我们可以看到主要的原因是：whisper 库的 metadata name 不匹配（应该是被 OpenAI 更改了），只需要在 `setup.py` 中进行如下的更改：

```python
# 更改前
requirements = [
    "whisper @ git+https://github.com/openai/whisper.git",
]

# 更改后
requirements = [
    "openai-whisper @ git+https://github.com/openai/whisper.git",
]
```

经过上述的修改之后，我们可以按照以下命令对 AutoCut 进行本地安装，如下所示：

```shell
# 本地安装
git clone https://github.com/mli/autocut
cd autocut
pip install .
```


### Transcribe（转录）

主要使用 OpenAI 的 `Whisper` 模型对（视频中的）音频进行读取和转录，并将转录后的结果输出为 SRT 字幕文件和 Markdown 文件。

为了识别未出声的片段（**Voice Activity Detect，VAD**），可以调用现有的 VAD 方法（这里调用 [Silero VAD 方法](https://github.com/snakers4/silero-vad)），参见 `transcribe.py` 中的 `_detect_voice_activity()` 函数：

```python
# torch load limit https://github.com/pytorch/vision/issues/4156
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
self.vad_model, funcs = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
)

self.detect_speech = funcs[0]

speeches = self.detect_speech(
    audio, self.vad_model, sampling_rate=self.sampling_rate
)
```

#### 使用方法

```shell
!autocut -t inputs 九转大肠最全前因后果（补档）.mp4

!autocut -t inputs 九转大肠最全前因后果（补档）.mp4 --whisper-model medium
```

#### 关键代码

> 代码具体参考：AutoCut 中的 `transcribe.py` 脚本：

```python
# 读取音频/视频文件
audio = whisper.load_audio(input, sr=self.sampling_rate)

# 加载 Whisper 语言模型
self.whisper_model = whisper.load_model(
    self.args.whisper_model, self.args.device
)

# 进行转录
# 对于 CPU，使用多线程进行加速（默认使用双线程）
r = self.whisper_model.transcribe(
    audio[int(seg["start"]) : int(seg["end"])],
    task="transcribe",
    language=self.args.lang,
    initial_prompt=self.args.prompt,
    verbose=False if len(speech_timestamps) == 1 else None,
)
```

#### 结果展示

```shell
!autocut -t inputs 九转大肠最全前因后果（补档）.mp4
```

```shell
[autocut:driver.py:L120] INFO   Generating grammar tables from /usr/lib/python3.8/lib2to3/Grammar.txt
[autocut:driver.py:L120] INFO   Generating grammar tables from /usr/lib/python3.8/lib2to3/PatternGrammar.txt
[autocut:utils.py:L160] INFO   NumExpr defaulting to 2 threads.
[autocut:transcribe.py:L37] INFO   Transcribing 九转大肠最全前因后果（补档）.mp4
Using cache found in /root/.cache/torch/hub/snakers4_silero-vad_master
[autocut:transcribe.py:L86] INFO   Done voice activity detection in 28.9 sec
  0% 0/42 [00:00<?, ?it/s]/usr/local/lib/python3.8/dist-packages/whisper/transcribe.py:79: UserWarning: FP16 is not supported on CPU; using FP32 instead
  warnings.warn("FP16 is not supported on CPU; using FP32 instead")
100% 42/42 [19:15<00:00, 27.51s/it]
[autocut:transcribe.py:L138] INFO   Done transcription in 1164.9 sec
[autocut:srt.py:L303] INFO   Skipped subtitle at index 0: Subtitle start time >= end time
[autocut:transcribe.py:L55] INFO   Transcribed 九转大肠最全前因后果（补档）.mp4 to 九转大肠最全前因后果（补档）.srt
[autocut:transcribe.py:L57] INFO   Saved texts to 九转大肠最全前因后果（补档）.md to mark sentences
```

### Cut

> 代码具体参考：AutoCut 中的 `cut.py` 脚本

主要使用 `srt` 和 `moviepy` 这两个开源库来对视频/音频进行剪辑。`srt` 库负责对 SRT 字幕文件的读取，`moviepy` 库负责对视频/音频文件进行读写和剪辑。

#### 使用方法

```shell
!autocut -c 九转大肠最全前因后果（补档）.mp4 九转大肠最全前因后果（补档）.srt
```

#### 关键代码

- `srt.parse()`：解析 SRT 字幕文件

- `editor.VideoFileClip()`、`editor.AudioFileClip()`

- `subclip()`：根据 SRT 字幕文件，将视频/音频剪切为多个片段（clips/segments）

- `editor.concatenate_videoclips()`、`editor.concatenate_audioclips()`：将多个片段进行拼接

- `editor.VideoClip.write_videofile()`、`editor.AudioClip.write_audiofile()`：将多个片段写入到一个视频/音频文件中

```python
import srt
from moviepy import editor

with open(fns["srt"], encoding=self.args.encoding) as f:
    subs = list(srt.parse(f.read()))

# 重新编排 SRT 字幕文件的顺序
segments = []
# Avoid disordered subtitles
subs.sort(key=lambda x: x.start)

if is_video_file:
    media = editor.VideoFileClip(fns["media"])
else:
    media = editor.AudioFileClip(fns["media"])

clips = [media.subclip(s["start"], s["end"]) for s in segments]
if is_video_file:
    final_clip: editor.VideoClip = editor.concatenate_videoclips(clips)
    final_clip.write_videofile()
else:
    final_clip: editor.AudioClip = editor.concatenate_audioclips(clips)
    final_clip.write_audiofile()
```

#### 问题

在进行剪辑时，出现以下错误：

```shell
Traceback (most recent call last):
  File "/usr/local/bin/autocut", line 8, in <module>
    sys.exit(main())
  File "/usr/local/lib/python3.8/dist-packages/autocut/main.py", line 113, in main
    from .cut import Cutter
  File "/usr/local/lib/python3.8/dist-packages/autocut/cut.py", line 6, in <module>
    from moviepy import editor
  File "/usr/local/lib/python3.8/dist-packages/moviepy/editor.py", line 26, in <module>
    imageio.plugins.ffmpeg.download()
  File "/usr/local/lib/python3.8/dist-packages/imageio/plugins/ffmpeg.py", line 37, in download
    raise RuntimeError(
RuntimeError: imageio.ffmpeg.download() has been deprecated. Use 'pip install imageio-ffmpeg' instead.'
```

根据 [StackOverflow](https://stackoverflow.com/questions/55965507/runtimeerror-imageio-ffmpeg-download-has-been-deprecated-use-pip-install-im) 中的方法，只需要对 `imageio` 进行降版本，如下所示：

```shell
!pip install imageio==2.4.1
```

#### 结果展示

```shell
!autocut -c inputs 九转大肠最全前因后果（补档）.mp4 九转大肠最全前因后果（补档）.srt
```

```shell
[autocut:cut.py:L110] INFO   Cut 九转大肠最全前因后果（补档）.mp4 based on 九转大肠最全前因后果（补档）.srt
[autocut:cut.py:L143] INFO   Reduced duration from 796.3 to 567.8
[MoviePy] >>>> Building video 九转大肠最全前因后果（补档）_cut.mp4
[MoviePy] Writing audio in 九转大肠最全前因后果（补档）_cutTEMP_MPY_wvf_snd.mp4
100% 12520/12520 [00:43<00:00, 288.93it/s]
[MoviePy] Done.
[MoviePy] Writing video 九转大肠最全前因后果（补档）_cut.mp4
100% 17030/17034 [03:18<00:00, 80.45it/s][autocut:warnings.py:L109] WARNING /usr/local/lib/python3.8/dist-packages/moviepy/video/io/ffmpeg_reader.py:123: UserWarning: Warning: in file 九转大肠最全前因后果（补档）.mp4, 552960 bytes wanted but 0 bytes read,at frame 23885/23888, at time 796.17/796.26 sec. Using the last valid frame instead.
  warnings.warn("Warning: in file %s, "%(self.filename)+

[autocut:warnings.py:L109] WARNING /usr/local/lib/python3.8/dist-packages/moviepy/video/io/ffmpeg_reader.py:123: UserWarning: Warning: in file 九转大肠最全前因后果（补档）.mp4, 552960 bytes wanted but 0 bytes read,at frame 23886/23888, at time 796.20/796.26 sec. Using the last valid frame instead.
  warnings.warn("Warning: in file %s, "%(self.filename)+

[autocut:warnings.py:L109] WARNING /usr/local/lib/python3.8/dist-packages/moviepy/video/io/ffmpeg_reader.py:123: UserWarning: Warning: in file 九转大肠最全前因后果（补档）.mp4, 552960 bytes wanted but 0 bytes read,at frame 23887/23888, at time 796.23/796.26 sec. Using the last valid frame instead.
  warnings.warn("Warning: in file %s, "%(self.filename)+

100% 17034/17034 [03:18<00:00, 85.76it/s]
[MoviePy] Done.
[MoviePy] >>>> Video ready: 九转大肠最全前因后果（补档）_cut.mp4 

[autocut:cut.py:L167] INFO   Saved media to 九转大肠最全前因后果（补档）_cut.mp4
```

## faster-whisper

> 参考：
>
> - github: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)



## 更多

- [MoviePy 文档](http://doc.moviepy.com.cn/index.html#)

- [srt 文档](https://srt.readthedocs.io/en/latest/quickstart.html)

- [github: srt](https://github.com/cdown/srt)

- 基于 Whisper 的开源应用程序：[buzz](https://github.com/chidiwilliams/buzz)


## 参考

- [github: autocut](https://github.com/mli/autocut)

- [github: whisper](https://github.com/openai/whisper)

- [OpenAI Whisper Blog: Introducing Whisper](https://openai.com/blog/whisper/)

- [StackOverflow](https://stackoverflow.com/questions/55965507/runtimeerror-imageio-ffmpeg-download-has-been-deprecated-use-pip-install-im)