---
layout: fragment
title: HuggingFace 镜像
tags: [HuggingFace]
excerpt: HuggingFace 镜像
keywords: HuggingFace
mathjax: true
---

## 方式一：modelee

[modelee](https://gitee.com/modelee) 当前主要是提供 huggingface 模型和数据集的镜像，方便国内开发者快速获取。

- 本站供您下载模型镜像文件（以下称 “镜像文件”）的知识产权归属于该镜像文件的开发者所有。您的下载和使用镜像文件的行为应遵守该等知识产权的约定。若违侵犯该等知识产权，在法律规定的范围内，由此带来的责任由您承担。

- 除第 1 条的镜像文件外，本站的其他内容，包括图片、网站架构与画面的安排、网页设计、文字、图表、代码、SDK、API、LOGO 等知识产权及其他合法权益，包括商标权、著作权与专利权等知识产权，均归我们或我们的关联方所有。未经我们或关联公司的事先书面许可，任何人不得以包括通过机器人、蜘蛛等程序或设备监视、复制、传播、展示、镜像、上载、下载等方式擅自使用本站内除第 1 条镜像文件之外的所有内容。


## 方式二：hf-mirror（推荐）

> [hf-mirror](https://hf-mirror.com/) 是一个 HuggingFace 镜像站，提供 HuggingFace 模型和数据集的搜索。

> hf-mirror.com，用于镜像 huggingface.co 域名。作为一个公益项目，致力于帮助国内 AI 开发者快速、稳定的下载模型、数据集。捐赠支持请看网页左下角，感谢支持！

- 使用 url 下载模型或者数据集时，将 `huggingface.co` 直接替换为 `hf-mirror.com`。使用浏览器或者 `wget -c`、`curl -L`、`aria2c` 等命令行方式即可。

- 下载需登录的模型需命令行添加 `--header hf_***` 参数，添加 HuggingFace 的 `token`。

更多详细用法请看[《这篇教程》](https://zhuanlan.zhihu.com/p/663712983)。

### 1）网页下载

在 hf-mirror.com 中搜索，并在模型主页的 `Files and Version` 中下载文件。

### 2）huggingface-cli

huggingface-cli 是 Hugging Face 官方提供的命令行工具，自带完善的下载功能。

1. 安装依赖

```shell
pip install -U huggingface_hub
```

2. 设置环境变量

- Linux

```shell
export HF_ENDPOINT=https://hf-mirror.com
```

- Windows Powershell

```shell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

建议将上面这一行写入 `~/.bashrc`。

3.1 下载模型

```shell
huggingface-cli download --resume-download jetmoe/jetmoe-8b --local-dir jetmoe/jetmoe-8b
```

3.2 下载数据集

```shell
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
```

可以添加 `--local-dir-use-symlinks False` 参数禁用文件软链接，这样下载路径下所见即所得，详细解释请见上面提到的教程。

### 3）使用 hfd

[hfd](https://gist.github.com/padeoe/697678ab8e528b85a2a7bddafea1fa4f) 是 hf-mirror 作者开发的 huggingface 专用下载工具，基于成熟工具 `git + aria2`，可以做到稳定下载不断线。

0. 安装 aria2（Ubuntu）

```shell
sudo apt update
sudo apt install aria2
```

1. 下载 hfd

```shell
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
```

2. 设置环境变量

- Linux

```shell
export HF_ENDPOINT=https://hf-mirror.com
```

- Windows Powershell

```shell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

3.1 下载模型

```shell
# 使用 aria2 工具进行下载，4 线程
./hfd.sh jetmoe/jetmoe-8b --tool aria2c -x 4
```

运行结果为：

```shell
Downloading to jetmoe-8b
Testing GIT_REFS_URL: https://hf-mirror.com/jetmoe/jetmoe-8b/info/refs?service=git-upload-pack
git clone https://hf-mirror.com/jetmoe/jetmoe-8b jetmoe-8b
Cloning into 'jetmoe-8b'...
remote: Enumerating objects: 104, done.
remote: Counting objects: 100% (100/100), done.
remote: Compressing objects: 100% (100/100), done.
remote: Total 104 (delta 49), reused 0 (delta 0), pack-reused 4
Receiving objects: 100% (104/104), 1.81 MiB | 167.00 KiB/s, done.
Resolving deltas: 100% (49/49), done.

Start Downloading lfs files, bash script:
cd jetmoe-8b
aria2c --console-log-level=error -x 4 -s 4 -k 1M -c "https://hf-mirror.com/jetmoe/jetmoe-8b/resolve/main/model-00001-of-00004.safetensors" -d "." -o "model-00001-of-00004.safetensors"
aria2c --console-log-level=error -x 4 -s 4 -k 1M -c "https://hf-mirror.com/jetmoe/jetmoe-8b/resolve/main/model-00002-of-00004.safetensors" -d "." -o "model-00002-of-00004.safetensors"
aria2c --console-log-level=error -x 4 -s 4 -k 1M -c "https://hf-mirror.com/jetmoe/jetmoe-8b/resolve/main/model-00003-of-00004.safetensors" -d "." -o "model-00003-of-00004.safetensors"
aria2c --console-log-level=error -x 4 -s 4 -k 1M -c "https://hf-mirror.com/jetmoe/jetmoe-8b/resolve/main/model-00004-of-00004.safetensors" -d "." -o "model-00004-of-00004.safetensors"
aria2c --console-log-level=error -x 4 -s 4 -k 1M -c "https://hf-mirror.com/jetmoe/jetmoe-8b/resolve/main/tokenizer.model" -d "." -o "tokenizer.model"
Start downloading model-00001-of-00004.safetensors.
 *** Download Progress Summary as of Sat Apr 13 14:38:13 2024 ***                                          
===========================================================================================================
[#73a50f 684MiB/4.5GiB(14%) CN:4 DL:11MiB ETA:5m47s]
FILE: ./model-00001-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:39:14 2024 ***                                          
===========================================================================================================
[#73a50f 1.3GiB/4.5GiB(29%) CN:4 DL:11MiB ETA:4m48s]
FILE: ./model-00001-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:40:15 2024 ***                                          
===========================================================================================================
[#73a50f 2.0GiB/4.5GiB(44%) CN:4 DL:11MiB ETA:3m48s]
FILE: ./model-00001-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:41:15 2024 ***                                          
===========================================================================================================
[#73a50f 2.6GiB/4.5GiB(59%) CN:4 DL:11MiB ETA:2m47s]
FILE: ./model-00001-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:42:16 2024 ***                                          
===========================================================================================================
[#73a50f 3.3GiB/4.5GiB(74%) CN:4 DL:11MiB ETA:1m46s]
FILE: ./model-00001-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:43:17 2024 ***                                          
===========================================================================================================
[#73a50f 4.0GiB/4.5GiB(88%) CN:4 DL:11MiB ETA:45s]
FILE: ./model-00001-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

[#73a50f 4.5GiB/4.5GiB(99%) CN:1 DL:10MiB]                                                                 
Download Results:
gid   |stat|avg speed  |path/URI
======+====+===========+=======================================================
73a50f|OK  |    11MiB/s|./model-00001-of-00004.safetensors

Status Legend:
(OK):download completed.
Downloaded https://hf-mirror.com/jetmoe/jetmoe-8b/resolve/main/model-00001-of-00004.safetensors successfully.
Start downloading model-00002-of-00004.safetensors.
 *** Download Progress Summary as of Sat Apr 13 14:45:04 2024 ***                                          
===========================================================================================================
[#bef033 668MiB/4.5GiB(14%) CN:4 DL:11MiB ETA:5m56s]
FILE: ./model-00002-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:46:04 2024 ***                                          
===========================================================================================================
[#bef033 1.3GiB/4.5GiB(28%) CN:4 DL:11MiB ETA:4m56s]
FILE: ./model-00002-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:47:05 2024 ***                                          
===========================================================================================================
[#bef033 1.9GiB/4.5GiB(43%) CN:4 DL:11MiB ETA:3m55s]
FILE: ./model-00002-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:48:06 2024 ***                                          
===========================================================================================================
[#bef033 2.6GiB/4.5GiB(58%) CN:4 DL:11MiB ETA:2m52s]
FILE: ./model-00002-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:49:07 2024 ***                                          
===========================================================================================================
[#bef033 3.3GiB/4.5GiB(72%) CN:4 DL:11MiB ETA:1m53s]
FILE: ./model-00002-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:50:08 2024 ***                                          
===========================================================================================================
[#bef033 4.0GiB/4.5GiB(87%) CN:4 DL:11MiB ETA:53s]
FILE: ./model-00002-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

[#bef033 4.5GiB/4.5GiB(99%) CN:2 DL:10MiB]                                                                 
Download Results:
gid   |stat|avg speed  |path/URI
======+====+===========+=======================================================
bef033|OK  |    11MiB/s|./model-00002-of-00004.safetensors

Status Legend:
(OK):download completed.
Downloaded https://hf-mirror.com/jetmoe/jetmoe-8b/resolve/main/model-00002-of-00004.safetensors successfully.
Start downloading model-00003-of-00004.safetensors.
 *** Download Progress Summary as of Sat Apr 13 14:52:03 2024 ***                                          
===========================================================================================================
[#aee76a 684MiB/4.5GiB(14%) CN:4 DL:11MiB ETA:5m53s]
FILE: ./model-00003-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:53:04 2024 ***                                          
===========================================================================================================
[#aee76a 1.3GiB/4.5GiB(29%) CN:4 DL:11MiB ETA:4m53s]
FILE: ./model-00003-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:54:04 2024 ***                                          
===========================================================================================================
[#aee76a 2.0GiB/4.5GiB(43%) CN:4 DL:11MiB ETA:3m55s]
FILE: ./model-00003-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:55:05 2024 ***                                          
===========================================================================================================
[#aee76a 2.6GiB/4.5GiB(58%) CN:4 DL:11MiB ETA:2m50s]
FILE: ./model-00003-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:56:06 2024 ***                                          
===========================================================================================================
[#aee76a 3.3GiB/4.5GiB(73%) CN:4 DL:11MiB ETA:1m50s]
FILE: ./model-00003-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:57:06 2024 ***                                          
===========================================================================================================
[#aee76a 4.0GiB/4.5GiB(87%) CN:4 DL:11MiB ETA:50s]
FILE: ./model-00003-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

[#aee76a 4.5GiB/4.5GiB(99%) CN:2 DL:11MiB]                                                                 
Download Results:
gid   |stat|avg speed  |path/URI
======+====+===========+=======================================================
aee76a|OK  |    11MiB/s|./model-00003-of-00004.safetensors

Status Legend:
(OK):download completed.
Downloaded https://hf-mirror.com/jetmoe/jetmoe-8b/resolve/main/model-00003-of-00004.safetensors successfully.
Start downloading model-00004-of-00004.safetensors.
 *** Download Progress Summary as of Sat Apr 13 14:58:58 2024 ***                                          
===========================================================================================================
[#b781a0 665MiB/2.1GiB(30%) CN:4 DL:11MiB ETA:2m14s]
FILE: ./model-00004-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 14:59:59 2024 ***                                          
===========================================================================================================
[#b781a0 1.3GiB/2.1GiB(61%) CN:4 DL:11MiB ETA:1m13s]
FILE: ./model-00004-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

 *** Download Progress Summary as of Sat Apr 13 15:00:59 2024 ***                                          
===========================================================================================================
[#b781a0 1.9GiB/2.1GiB(93%) CN:4 DL:11MiB ETA:13s]
FILE: ./model-00004-of-00004.safetensors
-----------------------------------------------------------------------------------------------------------

[#b781a0 2.1GiB/2.1GiB(99%) CN:4 DL:11MiB]                                                                 
Download Results:
gid   |stat|avg speed  |path/URI
======+====+===========+=======================================================
b781a0|OK  |    11MiB/s|./model-00004-of-00004.safetensors

Status Legend:
(OK):download completed.
Downloaded https://hf-mirror.com/jetmoe/jetmoe-8b/resolve/main/model-00004-of-00004.safetensors successfully.
Start downloading tokenizer.model.
[#1bca27 0B/0B CN:1 DL:0B]                                                                                 
Download Results:
gid   |stat|avg speed  |path/URI
======+====+===========+=======================================================
1bca27|OK  |   2.1MiB/s|./tokenizer.model

Status Legend:
(OK):download completed.
Downloaded https://hf-mirror.com/jetmoe/jetmoe-8b/resolve/main/tokenizer.model successfully.
Download completed successfully.
```

3.2 下载数据集

```shell
./hfd.sh wikitext --dataset --tool aria2c -x 4
```

---

`hfd` 的完整用法如下：

```shell
./hfd.sh -h

Usage:
  hfd <repo_id> [--include include_pattern] [--exclude exclude_pattern] [--hf_username username] [--hf_token token] [--tool aria2c|wget] [-x threads] [--dataset] [--local-dir path]

Description:
  Downloads a model or dataset from Hugging Face using the provided repo ID.

Parameters:
  repo_id        The Hugging Face repo ID in the format 'org/repo_name'.
  --include       (Optional) Flag to specify a string pattern to include files for downloading.
  --exclude       (Optional) Flag to specify a string pattern to exclude files from downloading.
  include/exclude_pattern The pattern to match against filenames, supports wildcard characters. e.g., '--exclude *.safetensor', '--include vae/*'.
  --hf_username   (Optional) Hugging Face username for authentication. **NOT EMAIL**.
  --hf_token      (Optional) Hugging Face token for authentication.
  --tool          (Optional) Download tool to use. Can be aria2c (default) or wget.
  -x              (Optional) Number of download threads for aria2c. Defaults to 4.
  --dataset       (Optional) Flag to indicate downloading a dataset.
  --local-dir     (Optional) Local directory path where the model or dataset will be stored.

Example:
  hfd bigscience/bloom-560m --exclude *.safetensors
  hfd meta-llama/Llama-2-7b --hf_username myuser --hf_token mytoken -x 4
  hfd lavita/medical-qa-shared-task-v1-toy --dataset
```



### 4）使用环境变量（非侵入式）

非侵入式，能解决大部分情况。huggingface 工具链会获取 `HF_ENDPOINT` 环境变量来确定下载文件所用的网址，所以可以使用通过设置变量来解决。

```shell
HF_ENDPOINT=https://hf-mirror.com python your_script.py
```

不过有些数据集有内置的下载脚本，那就需要手动改一下脚本内的地址来实现了。


### 常见问题

Q: **有些项目需要登录，如何下载？**

A：部分 Gated Repo 需登录申请许可。为保障账号安全，本站不支持登录，需先前往 Hugging Face 官网登录、申请许可，在官网这里获取 Access Token 后回镜像站用命令行下载。

部分工具下载 Gated Repo 的方法：

- huggingface-cli： 添加 `--token` 参数

```shell
huggingface-cli download --token hf_*** --resume-download meta-llama/Llama-2-7b-hf --local-dir Llama-2-7b-hf
```

- hfd： 添加 `--hf_username`、`--hf_token` 参数

```shell
hfd meta-llama/Llama-2-7b --hf_username YOUR_HF_USERNAME --hf_token hf_***
```

其余如 `from_pretrained`、`wget`、`curl` 如何设置认证 token，详见上面第一段提到的教程。


## 总结

如果各位有魔法，可以直接使用魔法来访问、下载 HuggingFace 的模型和数据集。

如果没有魔法，可以尝试使用 HuggingFace 镜像进行下载。

如果需要在 Linux 中尝试使用魔法，可以参考 [Clash-for-Linux 的安装与使用](/fragment/2024-01-03-clash-for-linux)


## 参考

- 知乎问题：[有没有下载 Hugging Face 模型的国内站点？](https://www.zhihu.com/question/371644077)

- 知乎：[如何快速下载huggingface模型——全方法总结](https://zhuanlan.zhihu.com/p/663712983)