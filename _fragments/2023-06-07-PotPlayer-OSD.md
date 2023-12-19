---
layout: fragment
title: PotPlayer 使用：打开/关闭 OSD
tags: [PotPlayer]
excerpt: PotPlayer 使用：打开/关闭 OSD
keywords: PotPlayer
mathjax: true
---

## PotPlayer 的 OSD

**OSD** 是 PotPlayer 播放工具中的一个功能，它是指 **On Screen Display** 的缩写，即屏幕上显示的信息。

**OSD 打开 / 关闭的默认快捷键都是 `Tab`。**

从 PotPlayer 的 OSD 就能直接获取到当前视频和播放器的大部分状态，这也是 PotPlayer 吸引人的一大重要特色。PotPlayer 的 OSD 给出的信息很直观，且信息量大，多数情况下需要直接按 `Tab` 根据 OSD 信息来分析当前状态。

如下图所示：

![](http://www.potplayercn.com/wp-content/uploads/2019-05/2019053011450225566.jpg)


打开 OSD 我们可看到：视频编码器、编码、分辨率、帧率、位率（码率）、音频解码器等信息。

### 各种信息的详细解释

- OSD 中的视频/音频解码器表示当前正在使用的解码器，`Built-in Video Codec/Transform` 和 `Built-in AudioCodec/Transform` 即为 POT 默认解码器+图像/语音处理滤镜，后面括号内的 `DXVA Decoder` 和 `FFmpegMininum.dll` 为默认解码器所对应的具体内置解码器名称，如果具体视频解码器中出现了 `DXVA` 字样则为开启了内置传统` DXVA` 硬解；当是外部第三方解码器时，视频/音频解码器则直接显示外部解码器的具体名称；当音频解码器不显示或显示为 `DirectSound` 时，则为音频不软解而直通渲染器。

- OSD 中的视频/音频输入格式为当前视频/音频的具体编码格式，如图中的 AVC1 即为 `H.264` 格式。音频输入格式也可能出现 `FLAC`、`AC3`、`DTS`等。

- OSD 中的视频输出为视频解码器解码后输出的具体色彩格式，如 8bit 的 `YV12`、`YUY2`、`NV12`、`IMC3` 等。当显示 `DXVA` 时，意思是开启了 `DXVA` 纯硬解；由于大部分显卡，特别是 A 卡硬解后输出的色彩为 `NV12`，所以输出格式当显示为 `NV12` 或者 `YV12` 或者 `IMC3` 等格式时，也有可能表明开启了硬解，此时需要根据经验来判断。

- OSD 中的音频输入/输出/渲染输入中可以看到当前音频的采样率、声道数、位深和码率。OSD 中的视频/音频渲染器为正在呈现视频/音频的具体渲染器，如视频的 `VMR/EVR/madVR` 和音频的 `DirectSound` 等。

- OSD 中的 CPU 占用率以 `X%/Y%` 格式显示，X 表示 POT 进程本身的 CPU 占用率，而 Y 表示整个系统的 CPU 占用率。OSD 中的 GPU 占用率仅当使用 N 卡或 A 卡播放才会显示，核显则不支持显示。

- OSD 中的视频码率可以动态显示当前实际码率。

- OSD 中的输出帧率是判断当前播放的视频是否掉帧或倍帧成功关键数据。

### 其他方式查看视频的详细信息

此外在播放视频时，底部菜单栏也会显示视频编码及音频编码，例如下图的 `AVC1` 及 `AAC`。

![](http://www.potplayercn.com/wp-content/uploads/2022-02/2022022415303983201.jpg)


也可以在播放器上右击，属性查看播放信息（快捷键ctrl+f1），如下图所示：

![](http://www.potplayercn.com/wp-content/uploads/2022-02/2022022415242159020.jpg)

文件信息里可查看更为详细的信息，可以复制到剪贴板。如下图所示：

![](http://www.potplayercn.com/wp-content/uploads/2022-02/2022022415230992113.jpg)


## 参考

- PotPlayer中文网：[打开PotPlayer的OSD 获取视频码率编码等信息](http://www.potplayercn.com/course/2978.html)
