---
layout: fragment
title: USB 启动盘制作工具：Ventoy 和 Rufus
tags: [操作系统]
excerpt: USB 启动盘制作工具：Ventoy 和 Rufus
keywords: 操作系统
mathjax: true
---

本文主要介绍两种开源的 USB 启动盘制作工具：**Ventoy** 和 **Rufus**。

## Ventoy

> - GitHub Repo：[ventoy](https://github.com/ventoy/Ventoy)
> 
> - [主页](https://www.ventoy.net/cn/index.html)

![](https://pica.zhimg.com/v2-ccbb3ca61db474b84dcbdb4aac3bbbb0_1440w.jpg?source=172ae18b)

简单来说，Ventoy 是一个制作可启动 U 盘的开源工具。

有了 Ventoy 你就无需反复地格式化 U 盘，你只需要把 ISO / WIM / IMG / VHD(x) / EFI 等类型的文件直接拷贝到 U 盘里面就可以启动了，无需其他操作。

你可以一次性拷贝很多个不同类型的镜像文件，Ventoy 会在启动时显示一个菜单来供你进行选择。

![](https://www.ventoy.net/static/img/screen/screen_uefi_cn.png?v=4)

你还可以在 Ventoy 的界面中直接浏览并启动本地硬盘中的 ISO / WIM / IMG/ VHD(x) / EFI 等类型的文件。

Ventoy 安装之后，同一个 U 盘可以同时支持 x86 Legacy BIOS、IA32 UEFI、x86_64 UEFI、ARM64 UEFI 和 MIPS64EL UEFI 模式，同时还不影响 U 盘的日常使用。

Ventoy 支持大部分常见类型的操作系统，包括 Windows / WinPE / Linux / ChromeOS / Unix / VMware / Xen 等。

### 在 Windows 中使用

首先下载最新版的 Ventoy，Ventoy 是一个 **绿色免安装工具**，因此将其解压后便可直接使用了。

双击 V`entoy2Disk.exe` 运行，可以看到如下画面。其中「1.0.39」代表当前 Ventoy 的版本。设备处是空的，表示当前还没有插入 U 盘。我们可以先插入 U 盘再双击运行 Ventoy，也可以先运行，再插入 U 盘后点击下绿色的刷新按钮检测 U 盘。

![](https://pic2.zhimg.com/80/v2-dab546a8625ec9faa563f92682edff99_720w.webp)

接着配置分区的类型。在 `配置选项-分区类型-GPT` 中选择 GPT 分区，这个分区格式选择指的是 U 盘所使用的格式，它默认的是 MBR ，因为我个人不在使用老设备，所以选择了 GPT。如下图所示：

![](/images/fragments/bootable-usb-tool/ventoy-setting.png)

如果你是 **首次安装**，那么点击安装即可开始安装，这是 **会格式化这个 U 盘** 的。所以你要提前备份好 U 盘里面的数据。而 **如果你是升级 U 盘的 Ventoy 版本，则是不会影响到其中的 ISO 镜像文件的**。

点击安装或升级后会弹出提示，表示安装成功。如下图所示：

![](/images/fragments/bootable-usb-tool/ventoy-install-success.png)

经过以上步骤 Ventoy 已成功安装到 U 盘。也就是说，该 U 盘现在已经可以引导启动系统了，但是它还不包含操作系统安装镜像。现在我们来将 ISO 镜像文件放入其中吧，如下图所示：

![](https://pic1.zhimg.com/80/v2-87db4624b89e90f55cc15f0ed8434324_720w.webp)

如你所见，我放了 3 个镜像，分别是：「Windows10」、「ArchLinux」、「WePE64」。

现在，当我用这个 U 盘引导系统启动后便可以选择这 3 个其中的一个系统镜像来安装系统了。日后更新系统只需替换这些镜像即可。


### 在 Linux 中使用

接下来是在 Linux 系统中安装 Ventoy 的方法。由于 Linux 系统大多面向的是老用户了，所以在这里快速说明：

```shell
cd ~/Downloads # 进入下载文件夹
```

下载 Ventoy 压缩包：

```shell
wget https://github.com/ventoy/Ventoy/releases/download/v1.0.91/ventoy-1.0.91-linux.tar.gz 
```

解压压缩包：

```shell
tar xvf ventoy-1.0.91-linux.tar.gz
```

进入 Ventoy 目录后执行脚本：

```shell
cd ventoy-1.0.91 # 进入 Ventoy 目录
sudo sh VentoyWeb.sh # 执行脚本
```

脚本启动后，会出现如下提示：

```shell
Ventoy Server 1.0.91 已经启动 ...
请打开浏览器，访问 http://127.0.0.1:24680
 ----------------------------
### Press Ctrl + C to exit ###
```

浏览器访问：http://127.0.0.1:24680 确认 U 盘无误，通常默认安装即可。

### 小结

安装完成后，我们通过观察可以发现，U盘中包含一个小的引导分区和一个大的默认使用 exFAT 格式的分区。

![](/images/fragments/bootable-usb-tool/ventoy-usb.png)

而我们上面所说的直接将 ISO 镜像文件放进 U 盘的就是 exFAT 格式的分区。Ventoy 的方便之处在于它让你不必像以往那样，次次都要重新格式化 U 盘，这样一来既节省了时间，也减少了麻烦。

哦对了，由于 Ventoy 的特性，你可以在该分区下放入 mp3、mp4、电子书、绿色版软件等，升级也不会干扰到其中的内容，也就是说它还可以当作普通U盘一样日用。

## Rufus

> - GitHub Repo：[rufus](https://github.com/pbatard/rufus)
> 
> - [主页](https://rufus.ie/zh/)

Rufus 是一款格式化和创建 USB 启动盘的辅助工具。

本软件适用于以下场景：

- 需要将可引导 ISO (Windows、Linux、UEFI 等) 刻录到 USB 安装媒介的情况

- 需要处理未安装操作系统的设备的情况

- 需要在 DOS 环境下刷写 BIOS 或其他固件的情况

- 需要运行低级工具的情况

Rufus 麻雀虽小，但五脏俱全！

Rufus 需要 **Windows 8 或更高版本的操作系统**，并且即开即用。

![](https://img.ithome.com/newsuploadfiles/2023/6/fba9470e-552a-4c12-97d4-e7e8ad9d0b1f.png?x-bce-process=image/format,f_auto)




## 哪种更好？

我个人更倾向于使用 Ventoy，有以下几个原因：

- Ventoy 支持 Linux 安装

- Ventoy 支持多种操作系统的切换

因此，选择何种工具，取决于你的需求。择优选择即可~



## 参考

- Ventoy

  - GitHub Repo：[ventoy](https://github.com/ventoy/Ventoy)

  - [主页](https://www.ventoy.net/cn/index.html)

  - 知乎：[新一代多系统启动U盘 Ventoy 使用指南](https://zhuanlan.zhihu.com/p/361447843)

- Rufus 

  - GitHub Repo：[rufus](https://github.com/pbatard/rufus)

  - [主页](https://rufus.ie/zh/)



