---
layout: post
author: bookstall
tags: 工具, Windows
categories: [工具, Windows]
description: Microsoft PowerToys Usage
keywords: 工具, Windows
title: 效率工具 Microsoft PowerToys 的使用
mathjax: true
---

## 介绍

![Microsoft PowerToys Logo](https://github.com/microsoft/PowerToys/raw/main/doc/images/overview/PT_hero_image.png)

**Microsoft PowerToys** 是一组免费的系统工具软件，由微软为 Windows 操作系统上的系统管理员设计。这些程序为系统加入或变更了一些功能，并加入更多自定义选项以提高生产力。

PowerToys 可用于 Windows 95、Windows XP、Windows 10 和 Windows 11。Windows 10 版 PowerToys 为自由及开放源代码软件，并使用 MIT 授权条款托管于 GitHub。


Windows 10 发表 4 年后，微软于 2019 年 5 月 8 日重启了 PowerToys，并在 Github 上开源发布，使得用户可以自行新增和自定义 PowerToys 的功能。首个预览版本于 2019 年 9 月发布，其中包含了 FancyZones 和 Windows 键的快捷键指南（Shortcut Guide）。

目前，PowerToys 提供以下语言版本：简体中文、中文（繁体）、捷克语、荷兰语、英语、法语、德语、匈牙利语、意大利语、日语、韩语、波兰语、葡萄牙语、葡萄牙语（巴西）、俄语、西班牙语和土耳其语。

## 下载

PowerToys 需要满足以下系统要求：Windows 11 or Windows 10 version 2004 (code name 20H1 / build number 19041) or newer.

可以通过以下几个方式进行安装。

### Via GitHub with EXE [Recommended]

在 [Microsoft PowerToys GitHub releases page](https://github.com/microsoft/PowerToys/releases) 下载对应的版本即可。其中：

- `x64`: 表示 Internet X64 处理器（CPU）

- `arm64`：表示 ARM 64 位处理器（CPU）


### Via Microsoft Store

Install from the [Microsoft Store's PowerToys page](https://aka.ms/getPowertoys).


### Via WinGet

Download PowerToys from [WinGet](https://github.com/microsoft/winget-cli#installing-the-client). To install PowerToys, run the following command from the command line or PowerShell:

```shell
winget install Microsoft.PowerToys -s winget
```

### Other install methods

There are [community driven install methods](https://github.com/microsoft/PowerToys/blob/main/doc/unofficialInstallMethods.md) such as **Chocolatey and Scoop**. If these are your preferred install solutions, you can find the install instructions there.

## 当前支持的实用工具

当前 PowerToys 可用的实用工具包括：

- Always on Top

- PowerToys Awake

- 颜色选取器

- FancyZones

- File Explorer 加载项

- File Locksmith

- 主机文件编辑器

- 图像大小调整器

- 键盘管理器

- 鼠标实用程序

- PowerRename

- Quick Accent

- PowerToys Run

- 屏幕标尺

- 快捷键指南

- 文本提取器

- 视频会议静音


### 1、Always on Top（始终置顶）

通过 Always on Top，可使用快捷键方式（`Win` + `Ctrl` + `T`）**将窗口固定在其他窗口的顶部**。

![Always on Top](https://learn.microsoft.com/zh-cn/windows/images/pt-always-on-top-menu.png)


### 2、PowerToys Awake（唤醒）

PowerToys Awake 旨在 **使计算机保持唤醒状态**，且无需管理其电源和睡眠设置。 运行耗时较长的任务时，此行为非常有用，**可确保计算机不会进入睡眠状态或关闭其屏幕**。

![PowerToys Awake](https://learn.microsoft.com/zh-cn/windows/images/pt-awake-menu.png)


### 3、Color Picker（颜色选取器）

颜色选取器是一种系统范围的颜色选取实用工具，通过 `Win` + `Shift` + `C` 进行激活。 

**从当前正在运行的任何应用程序中选取颜色**，然后选取器会自动将颜色按设置的格式 **复制到剪贴板** 中。 颜色选取器还包含一个编辑器，其中显示了之前选取的颜色的历史记录，你可用它来微调所选颜色并复制不同的字符串表示形式。 该代码基于 [马丁·克尔赞的颜色选取器](https://github.com/martinchrzan/ColorPicker)。

![颜色选取器](https://learn.microsoft.com/zh-cn/windows/images/pt-color-picker.png)


### 4、FancyZones

FancyZones 是一种 **窗口管理器**，可用于轻松创建复杂的窗口布局，并将窗口快速放入到这些布局中。

![FancyZones](https://learn.microsoft.com/zh-cn/windows/images/pt-fancy-zones.png)

### 5、File Explorer 加载项

通过 File Explorer 加载项，可在 **File Explorer（文件资源管理器）** 中实现 **预览窗格（Preview Panes）**呈现，从而显示 <u>SVG 图标 (.svg)、Markdown (.md) 和 PDF 文件</u> 预览。

若要启用预览窗格，请在 File Explorer 中选择 `视图` 选项卡，然后选择 `预览窗格`。

![File Explorer 加载项](https://learn.microsoft.com/zh-cn/windows/images/pt-file-explorer.png)

### 6、File Locksmith

File Locksmith 是一个 Windows shell 扩展，用于 **检查哪些文件正在使用以及由哪些进程使用**。

**右键单击** File Explorer 中的一个或多个选定文件，然后从菜单中选择 `使用此文件的进程`。

![File Locksmith](https://learn.microsoft.com/zh-cn/windows/images/powertoys-file-locksmith.png)

### 7、Hosts File Editor（主机文件编辑器）

主机文件编辑器是一种编辑包含域名和匹配 IP 地址的 **hosts 文件** 的便捷方式，充当一个用于识别和定位 IP 网络上主机的映射。

![Hosts File Editor](https://learn.microsoft.com/zh-cn/windows/images/pt-hosts-file-editor-facade.png)


### 8、图像大小调整器

图像大小调整器是一种用于快速调整图像大小的 Windows Shell 扩展。 只需在 File Explorer 中简单 **右键单击** 一下，立即就能 **调整一张或多张图像的大小**。

此代码基于 [Brice Lambson 的图像大小调整器](https://github.com/bricelam/ImageResizer)。

![图像大小调整器](https://learn.microsoft.com/zh-cn/windows/images/pt-image-resizer.png)


### 9、键盘管理器

通过键盘管理器，可 **重新映射键** 和 **创建自己的键盘快捷方式**，从而 **自定义键盘** 来提高工作效率。

![键盘管理器](https://learn.microsoft.com/zh-cn/windows/images/pt-keyboard-manager.png)

### 10、鼠标实用程序

鼠标实用程序添加了用于 **增强鼠标和光标** 的功能。

使用 `查找我的鼠标`，通过聚焦于光标的焦点 **快速查找鼠标的位置**，此功能基于由 [Raymond Chen 开发的源代码](https://github.com/oldnewthing)。

单击鼠标左键或右键时，鼠标荧光笔会显示可视指示器。鼠标指针十字准线以鼠标指针为中心绘制十字准线。

![鼠标实用程序](https://learn.microsoft.com/zh-cn/windows/images/pt-mouse-utils.png)

### 11、PowerRename

通过 PowerRename，可执行 **批量重命名、搜索和替换文件名称**。

它附带高级功能，例如 <u>使用正则表达式、面向特定文件类型、预览预期结果和撤消更改</u> 的能力。

此代码基于 [Chris Davis 的 SmartRename](https://github.com/chrdavis/SmartRename)。

![PowerRename](https://learn.microsoft.com/zh-cn/windows/images/pt-rename.png)

### 12、Quick Accent

Quick Accent 是 **键入重音字符** 的替代方法，当键盘不支持具有快捷键组合的特定重音时，此方法非常有用。

![Quick Accent](https://learn.microsoft.com/zh-cn/windows/images/pt-keyboard-accent.png)

### 13、PowerToys Run

PowerToys Run 可帮助你 **立即搜索和启动应用**。 如需打开，可使用快捷方式 `Alt` + `空格键`，然后开始键入。

对其他插件来说，它是开源和模块化的，现在还包含窗口切换器。

![PowerToys Run](https://learn.microsoft.com/zh-cn/windows/images/pt-run.png)

### 14、屏幕标尺

借助屏幕标尺，可根据图像边缘检测快速测量屏幕上的像素。

如需激活，可使用快捷方式 `Win` + `Shift` + `M`。

此灵感来自于 [Pete Blois 的 Rooler](https://github.com/peteblois/rooler)。

![屏幕标尺](https://learn.microsoft.com/zh-cn/windows/images/pt-screen-ruler.png)

### 15、快捷键指南

按下 `Win` + `Shift` + `/`（或者我们喜欢的 `Win` + `?`）时，会出现 **Windows 快捷键指南**，并显示桌面当前状态的可用快捷方式。

还可更改此设置，然后按住 `Win`。

![快捷键指南](https://learn.microsoft.com/zh-cn/windows/images/pt-shortcut-guide.png)

### 16、文本提取器

文本提取器是一种 **从屏幕上任意位置复制文本** 的便捷方法。

如需激活，可使用快捷方式 `Win` + `Shift` + `T`。

此代码基于 [Joe Finney 的 Text Grab](https://github.com/TheJoeFin/Text-Grab)。

![文本提取器](https://learn.microsoft.com/zh-cn/windows/images/pt-image-to-text.png)

### 17、视频会议静音

视频会议静音是在会议通话期间使用 `Win` + `Shift` + `Q` 对麦克风和相机 **“全局”静音** 的一种 **快捷方式**，它不考虑当前聚焦在哪个应用程序上。

- `Win` + `Shift` + `Q`：同时切换麦克风和视频

- `Win` + `Shift` + `A`：切换麦克风

- `Win` + `Shift` + `O`：切换视频

![视频会议静音](https://learn.microsoft.com/zh-cn/windows/images/pt-video-conference-mute.png)


## 教学视频

在该视频中，PowerToys 的项目经理 Clint Rutkas 演示了如何安装和使用各种提供的实用工具，还分享了一些提示，介绍了如何参与等等。

<video poster="https://learn.microsoft.com/video/media/5a7489f2-d38e-4557-be3f-4ccbb2737b8b/powertoys-tabsvspaces_960.jpg" style="height: 100%;width: 100%;overflow: hidden" crossorigin="anonymous" src="https://learn.microsoft.com/video/media/5a7489f2-d38e-4557-be3f-4ccbb2737b8b/powertoys-tabsvspaces_mid.mp4" oncontextmenu="return false;" preload="none" tabindex="-1" controls>
<source src="https://learn.microsoft.com/video/media/5a7489f2-d38e-4557-be3f-4ccbb2737b8b/powertoys-tabsvspaces_mid.mp4" type="video/mp4">
</video>


## 参考

- [PowerToys repo](https://github.com/microsoft/PowerToys)

- [PowerToys 中文文档](https://learn.microsoft.com/zh-cn/windows/powertoys/)

- 维基百科：[Microsoft PowerToys](https://zh.wikipedia.org/zh-cn/Microsoft_PowerToys)

