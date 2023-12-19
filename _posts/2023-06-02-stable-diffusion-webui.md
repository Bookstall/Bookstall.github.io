---
layout: post
author: bookstall
tags: Diffusion
categories: [Diffusion]
excerpt: 初次使用 Stable-Diffusion-WebUI，纯新手篇
keywords: Diffusion
title: Stable-Diffusion-WebUI 初尝
mathjax: true
---


## Stable-Diffusion-WebUI

> 软硬件信息：
>
> - Stable-Diffusion-WebUI `v1.3.0` 版本
> 
> - Ubuntu 20.04 的服务器
> 
> - Nvidia Tesla V100S 32GB


### 安装过程（Ubuntu）

创建 Conda 虚拟环境 `sdwebui`：

```shell
$ conda create -n sdwebui python==3.10.8
```



使用 git 把项目拉下来，使用 [ghproxy.com](https://ghproxy.com/) 代理绕开：

```shell
$ git clone https://ghproxy.com/https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
```

激活 `sdwebui` 虚拟环境：

```shell
$ conda activate sdwebui
```

进入 `stable-diffusion-webui` 项目目录：

```shell
$ cd stable-diffusion-webui
```



安装依赖：

```shell
$ pip install -r requirements_versions.txt
```

如果显示下面的信息，说明安装成功：

```shell
Successfully built basicsr jsonmerge antlr4-python3-runtime ffmpy future filterpy
Installing collected packages: trampoline, tokenizers, safetensors, resize-right, pytz, pydub, mpmath, lmdb, lit, lark, ffmpy, einops, cmake, boltons, antlr4-python3-runtime, aenum, addict, websockets, urllib3, uc-micro-py, tzdata, typing-extensions, tqdm, toolz, tomli, tensorboard-data-server, sympy, sniffio, smmap, six, semantic-version, regex, pyyaml, python-multipart, pyrsistent, pyparsing, pygments, pyasn1, psutil, protobuf, Pillow, piexif, packaging, orjson, oauthlib, nvidia-nvtx-cu11, nvidia-nccl-cu11, nvidia-cusparse-cu11, nvidia-curand-cu11, nvidia-cufft-cu11, nvidia-cuda-runtime-cu11, nvidia-cuda-nvrtc-cu11, nvidia-cuda-cupti-cu11, nvidia-cublas-cu11, numpy, networkx, multidict, mdurl, markupsafe, markdown, llvmlite, lazy_loader, kiwisolver, inflection, idna, h11, grpcio, future, fsspec, frozenlist, fonttools, filelock, exceptiongroup, cycler, click, charset-normalizer, certifi, cachetools, attrs, async-timeout, aiofiles, absl-py, yarl, yapf, werkzeug, uvicorn, tifffile, scipy, rsa, requests, PyWavelets, python-dateutil, pydantic, pyasn1-modules, opencv-python, omegaconf, nvidia-cusolver-cu11, nvidia-cudnn-cu11, numba, markdown-it-py, linkify-it-py, lightning-utilities, jsonschema, jinja2, imageio, gitdb, deprecation, contourpy, anyio, aiosignal, starlette, scikit-image, requests-oauthlib, pandas, mdit-py-plugins, matplotlib, jsonmerge, huggingface-hub, httpcore, google-auth, GitPython, blendmodes, aiohttp, transformers, httpx, google-auth-oauthlib, filterpy, fastapi, altair, tb-nightly, gradio-client, gradio, triton, torch, torchvision, facexlib, basicsr, torchmetrics, gfpgan, torchsde, torchdiffeq, tomesd, timm, realesrgan, pytorch_lightning, kornia, clean-fid, accelerate

Successfully installed GitPython-3.1.30 Pillow-9.5.0 PyWavelets-1.4.1 absl-py-1.4.0 accelerate-0.18.0 addict-2.4.0 aenum-3.1.12 aiofiles-23.1.0 aiohttp-3.8.4 aiosignal-1.3.1 altair-5.0.1 antlr4-python3-runtime-4.9.3 anyio-3.7.0 async-timeout-4.0.2 attrs-23.1.0 basicsr-1.4.2 blendmodes-2022 boltons-23.0.0 cachetools-5.3.1 certifi-2023.5.7 charset-normalizer-3.1.0 clean-fid-0.1.35 click-8.1.3 cmake-3.26.3 contourpy-1.0.7 cycler-0.11.0 deprecation-2.1.0 einops-0.4.1 exceptiongroup-1.1.1 facexlib-0.3.0 fastapi-0.94.0 ffmpy-0.3.0 filelock-3.12.0 filterpy-1.4.5 fonttools-4.39.4 frozenlist-1.3.3 fsspec-2023.5.0 future-0.18.3 gfpgan-1.3.8 gitdb-4.0.10 google-auth-2.19.0 google-auth-oauthlib-1.0.0 gradio-3.31.0 gradio-client-0.2.5 grpcio-1.54.2 h11-0.12.0 httpcore-0.15.0 httpx-0.24.1 huggingface-hub-0.14.1 idna-3.4 imageio-2.30.0 inflection-0.5.1 jinja2-3.1.2 jsonmerge-1.8.0 jsonschema-4.17.3 kiwisolver-1.4.4 kornia-0.6.7 lark-1.1.2 lazy_loader-0.2 lightning-utilities-0.8.0 linkify-it-py-2.0.2 lit-16.0.5 llvmlite-0.40.0 lmdb-1.4.1 markdown-3.4.3 markdown-it-py-2.2.0 markupsafe-2.1.2 matplotlib-3.7.1 mdit-py-plugins-0.3.3 mdurl-0.1.2 mpmath-1.3.0 multidict-6.0.4 networkx-3.1 numba-0.57.0 numpy-1.23.5 nvidia-cublas-cu11-11.10.3.66 nvidia-cuda-cupti-cu11-11.7.101 nvidia-cuda-nvrtc-cu11-11.7.99 nvidia-cuda-runtime-cu11-11.7.99 nvidia-cudnn-cu11-8.5.0.96 nvidia-cufft-cu11-10.9.0.58 nvidia-curand-cu11-10.2.10.91 nvidia-cusolver-cu11-11.4.0.1 nvidia-cusparse-cu11-11.7.4.91 nvidia-nccl-cu11-2.14.3 nvidia-nvtx-cu11-11.7.91 oauthlib-3.2.2 omegaconf-2.2.3 opencv-python-4.7.0.72 orjson-3.8.14 packaging-23.1 pandas-2.0.2 piexif-1.1.3 protobuf-4.23.2 psutil-5.9.5 pyasn1-0.5.0 pyasn1-modules-0.3.0 pydantic-1.10.8 pydub-0.25.1 pygments-2.15.1 pyparsing-3.0.9 pyrsistent-0.19.3 python-dateutil-2.8.2 python-multipart-0.0.6 pytorch_lightning-1.9.4 pytz-2023.3 pyyaml-6.0 realesrgan-0.3.0 regex-2023.5.5 requests-2.31.0 requests-oauthlib-1.3.1 resize-right-0.0.2 rsa-4.9 safetensors-0.3.1 scikit-image-0.20.0 scipy-1.10.1 semantic-version-2.10.0 six-1.16.0 smmap-5.0.0 sniffio-1.3.0 starlette-0.26.1 sympy-1.12 tb-nightly-2.14.0a20230531 tensorboard-data-server-0.7.0 tifffile-2023.4.12 timm-0.6.7 tokenizers-0.13.3 tomesd-0.1.2 tomli-2.0.1 toolz-0.12.0 torch-2.0.1 torchdiffeq-0.2.3 torchmetrics-0.11.4 torchsde-0.2.5 torchvision-0.15.2 tqdm-4.65.0 trampoline-0.1.2 transformers-4.25.1 triton-2.0.0 typing-extensions-4.6.2 tzdata-2023.3 uc-micro-py-1.0.2 urllib3-1.26.16 uvicorn-0.22.0 websockets-11.0.3 werkzeug-2.3.4 yapf-0.33.0 yarl-1.9.2
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
```

接着继续安装：

```shell
$ pip install -r requirements.txt
```

如果显示下面的信息，说明安装成功：

```shell
Successfully installed astunparse-1.6.3 opencv-contrib-python-4.7.0.72 pyDeprecate-0.3.2 pytorch_lightning-1.7.7 rich-13.4.1 tensorboard-2.13.0 timm-0.4.12
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
```



如果网络环境不太好，可尝试使用 `pip` 的 **镜像源（例如，清华镜像）** 来进行安装：

```shell
$ pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements_versions.txt

$ pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```



启动 stable-diffusion-webui

运行 `launch.py`，即可启动：

```shell
$ python launch.py
```

第一次打开 sd webui 过程中，`launch.py` 会从 github 再把其他依赖项目拉下来，没有梯子速度非常慢。

可以先尝试看能否正常启动，可以的话就完成了。

---

如果不能正常启动，无法从 github 安装部分依赖，可以使用 [ghproxy.com](https://ghproxy.com/) 代理绕开。

```shell
Python 3.10.8 (main, Nov 24 2022, 14:13:03) [GCC 11.2.0]
Version: v1.3.0
Commit hash: 20ae71faa8ef035c31aa3a410b707d792c8203a3
Installing clip
Installing open_clip
Traceback (most recent call last):
  File "/root/code/stable-diffusion-webui/launch.py", line 38, in <module>
    main()
  File "/root/code/stable-diffusion-webui/launch.py", line 29, in main
    prepare_environment()
  File "/root/code/stable-diffusion-webui/modules/launch_utils.py", line 269, in prepare_environment
    run_pip(f"install {openclip_package}", "open_clip")
  File "/root/code/stable-diffusion-webui/modules/launch_utils.py", line 124, in run_pip
    return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)
  File "/root/code/stable-diffusion-webui/modules/launch_utils.py", line 101, in run
    raise RuntimeError("\n".join(error_bits))
RuntimeError: Couldn't install open_clip.
Command: "/opt/conda/envs/sdwebui/bin/python" -m pip install https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip --prefer-binary
Error code: 1
stdout: Collecting https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip

stderr:   WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='github.com', port=443): Read timed out. (read timeout=15)")': /mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip
  WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ConnectTimeoutError(<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f6dfacc1db0>, 'Connection to github.com timed out. (connect timeout=15)')': /mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip
  WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ConnectTimeoutError(<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f6dfacc1f30>, 'Connection to github.com timed out. (connect timeout=15)')': /mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip
  WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ConnectTimeoutError(<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f6dfacc2020>, 'Connection to github.com timed out. (connect timeout=15)')': /mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip
  WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ConnectTimeoutError(<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f6dfacc21a0>, 'Connection to github.com timed out. (connect timeout=15)')': /mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip
ERROR: Could not install packages due to an OSError: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip (Caused by ConnectTimeoutError(<pip._vendor.urllib3.connection.HTTPSConnection object at 0x7f6dfacc2320>, 'Connection to github.com timed out. (connect timeout=15)'))
```

查看 `launch.py` 文件，可以看到配置信息主要在 `./modules/launch_utils.py` 脚本中，因此我们需要使用 `vi` 修改 `./modules/launch_utils.py` 中的 `prepare_environment()`：

```python
def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu118")
    torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.17')
    gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "https://github.com/TencentARC/GFPGAN/archive/8d2447a2d918f8eba5a4a01463fd48e45126a379.zip")
    clip_package = os.environ.get('CLIP_PACKAGE', "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip")
    openclip_package = os.environ.get('OPENCLIP_PACKAGE', "https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip")

    stable_diffusion_repo = os.environ.get('STABLE_DIFFUSION_REPO', "https://github.com/Stability-AI/stablediffusion.git")
    taming_transformers_repo = os.environ.get('TAMING_TRANSFORMERS_REPO', "https://github.com/CompVis/taming-transformers.git")
    k_diffusion_repo = os.environ.get('K_DIFFUSION_REPO', 'https://github.com/crowsonkb/k-diffusion.git')
    codeformer_repo = os.environ.get('CODEFORMER_REPO', 'https://github.com/sczhou/CodeFormer.git')
    blip_repo = os.environ.get('BLIP_REPO', 'https://github.com/salesforce/BLIP.git')
```

将其中的 `https://github.com/` 均修改为 `https://ghproxy.com/https://github.com/`，即：

```python
def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu118")
    torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.17')
    gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "https://ghproxy.com/https://github.com/TencentARC/GFPGAN/archive/8d2447a2d918f8eba5a4a01463fd48e45126a379.zip")
    clip_package = os.environ.get('CLIP_PACKAGE', "https://ghproxy.com/https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip")
    openclip_package = os.environ.get('OPENCLIP_PACKAGE', "https://ghproxy.com/https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip")

    stable_diffusion_repo = os.environ.get('STABLE_DIFFUSION_REPO', "https://ghproxy.com/https://github.com/Stability-AI/stablediffusion.git")
    taming_transformers_repo = os.environ.get('TAMING_TRANSFORMERS_REPO', "https://ghproxy.com/https://github.com/CompVis/taming-transformers.git")
    k_diffusion_repo = os.environ.get('K_DIFFUSION_REPO', 'https://ghproxy.com/https://github.com/crowsonkb/k-diffusion.git')
    codeformer_repo = os.environ.get('CODEFORMER_REPO', 'https://ghproxy.com/https://github.com/sczhou/CodeFormer.git')
    blip_repo = os.environ.get('BLIP_REPO', 'https://ghproxy.com/https://github.com/salesforce/BLIP.git')
```

然后重新启动：

```shell
$ python launch.py
```

可以看到，现在能够成功从 github 拉取项目，并且成功启动了 Stable-Diffusion-WebUI：

```shell
Python 3.10.8 (main, Nov 24 2022, 14:13:03) [GCC 11.2.0]
Version: v1.3.0
Commit hash: 20ae71faa8ef035c31aa3a410b707d792c8203a3
Installing open_clip
Cloning Stable Diffusion into /root/code/stable-diffusion-webui/repositories/stable-diffusion-stability-ai...
Cloning Taming Transformers into /root/code/stable-diffusion-webui/repositories/taming-transformers...
Cloning K-diffusion into /root/code/stable-diffusion-webui/repositories/k-diffusion...
Cloning CodeFormer into /root/code/stable-diffusion-webui/repositories/CodeFormer...
Cloning BLIP into /root/code/stable-diffusion-webui/repositories/BLIP...
Installing requirements for CodeFormer
Installing requirements
Launching Web UI with arguments: 
No module 'xformers'. Proceeding without it.
Downloading: "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors" to /root/code/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors

100%|██████████████████████████████████████████████████████████████████| 3.97G/3.97G [05:58<00:00, 11.9MB/s]
Calculating sha256 for /root/code/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors: Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
Startup time: 369.2s (import torch: 1.1s, import gradio: 1.1s, import ldm: 0.4s, other imports: 2.1s, list SD models: 363.1s, load scripts: 0.6s, create ui: 0.5s, gradio launch: 0.1s).
6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa
Loading weights [6ce0161689] from /root/code/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors
Creating model from config: /root/code/stable-diffusion-webui/configs/v1-inference.yaml
LatentDiffusion: Running in eps-prediction mode
DiffusionWrapper has 859.52 M params.
Downloading (…)olve/main/vocab.json: 100%|███████████████████████████████| 961k/961k [00:00<00:00, 2.06MB/s]
Downloading (…)olve/main/merges.txt: 100%|████████████████████████████████| 525k/525k [00:01<00:00, 473kB/s]
Downloading (…)cial_tokens_map.json: 100%|█████████████████████████████████| 389/389 [00:00<00:00, 1.47MB/s]
Downloading (…)okenizer_config.json: 100%|█████████████████████████████████| 905/905 [00:00<00:00, 3.70MB/s]
Downloading (…)lve/main/config.json: 100%|█████████████████████████████| 4.52k/4.52k [00:00<00:00, 12.4MB/s]
Applying optimization: sdp-no-mem... done.
Textual inversion embeddings loaded(0): 
Model loaded in 19.7s (calculate hash: 12.0s, load weights from disk: 0.2s, create model: 5.1s, apply weights to model: 0.3s, apply half(): 0.3s, move model to device: 1.8s).
```

同时，我们可以看到：

- stable-diffusion-webui 默认的端口为 `7860`，即本地 URL 为 ` http://127.0.0.1:7860`

- stable-diffusion-webui 默认会下载 `v1-5-pruned-emaonly.safetensors` 模型，并且保存在 `./models/Stable-diffusion/` 目录

---

由于使用的是 Linux 服务器，并且不带图形界面，因此可以使用 `ssh -L` 将服务器的 `7860` 映射到本地操作系统的端口号（可以是任意未被占用的端口号）：

```shell
ssh [服务器用户名]@[服务器IP] -L 127.0.0.1:[本地端口号]:127.0.0.1:7860
```

然后，在本地操作系统的浏览器中打开 `127.0.0.1:[本地端口号]` 即可：



![](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2b373c26b9b142ceba95126d4ea2630a~tplv-k3u1fbpfcp-zoom-in-crop-mark:1512:0:0:0.awebp?)



![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2eb6bfd6bf714a838dace85bc4dab0cd~tplv-k3u1fbpfcp-zoom-in-crop-mark:1512:0:0:0.awebp?)



### start.sh 脚本

> 参考：
>
> - [Linux Ubuntu22.04 安装stable diffusion webui（不借助科学上网的方式）](https://blog.csdn.net/weixin_42735060/article/details/130321331)



在 stable-diffusion-webui 目录下，创建一个 linux 的启动脚本 `start.sh`：

```shell
#!/bin/bash
 
export COMMANDLINE_ARGS="--listen --port 7860 --no-half --lowvram --precision full --xformers --reinstall-xformers --medvram"
 
python_cmd="python"
LAUNCH_SCRIPT="launch.py"
 
"${python_cmd}" "${LAUNCH_SCRIPT}" "$@"
```

- `--listen` 会让启动的端口变为 `http://127.0.0.1:7860`

- `--port` 规定需要的端口

- `--no-half`：不要开启 float-32，改用 float-64

- `--lowvram`：意为可以降低模组速度，为了适配电压情况

  ```python
  ("--lowvram", action='store_true', help="enable stable diffusion model optimizations for sacrificing a lot of speed for very low VRM usage")
  ```

  - `VRM`：电压调解模组

- `--precision`：精度

  - `full`：全精度

- `--xformers`：开启 `xformers` 来加速 Transformer

- `--reinstall-xformers`：重新安装 `xformers` 库

- `--medvram`：medium VRM usage

然后执行 `bash start.sh`

需要加什么参数也可以直接在 `COMMANDLINE_ARGS` 这个位置加



### 生成的图片保存的地址

Stable-Diffusion-WebUI 默认会将生成的结果保存到 `./outputs/txt2img-images/` 和 `./outputs/txt2img-grids/` 目录中，并且会根据当前的日期进行进一步分类。

其中，

- `./outputs/txt2img-images/` 保存的是每张具体的图片

- `./outputs/txt2img-grids/` 保存的是所有样本的整体预览图

```shell
Folder "outputs/txt2img-images" does not exist. After you create an image, the folder will be created.
```

也就是说，在 2023-06-01 生成的图片会被保存到 `./outputs/txt2img/2023-06-01` 目录中。



### 指定运行的显卡

python 默认使用的是 0 号显卡，如果你有多张显卡，可以通过 `CUDA_VISIBLE_DEVICES` 来指定需要使用的显卡，例如：

```shell
$ CUDA_VISIBLE_DEVICES=3 python launch.py
```



### xformers 提速

默认是没有 `xformers` 的，从下面的运行结果也可以看出：

```shell
$ CUDA_VISIBLE_DEVICES=3 python launch.py
Python 3.10.8 (main, Nov 24 2022, 14:13:03) [GCC 11.2.0]
Version: v1.3.0
Commit hash: 20ae71faa8ef035c31aa3a410b707d792c8203a3
Installing requirements
Launching Web UI with arguments: 
No module 'xformers'. Proceeding without it.
Loading weights [6ce0161689] from /root/code/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
Startup time: 5.5s (import torch: 1.0s, import gradio: 1.1s, import ldm: 0.4s, other imports: 1.4s, load scripts: 0.5s, create ui: 0.6s, gradio launch: 0.3s).
Creating model from config: /root/code/stable-diffusion-webui/configs/v1-inference.yaml
LatentDiffusion: Running in eps-prediction mode
DiffusionWrapper has 859.52 M params.
Applying optimization: sdp-no-mem... done.
Textual inversion embeddings loaded(0): 
Model loaded in 2.7s (load weights from disk: 1.0s, create model: 0.5s, apply weights to model: 0.3s, apply half(): 0.4s, move model to device: 0.5s).
```

也就是这里的 `No module 'xformers'. Proceeding without it.`。

---

如果需要使用 `xformers`，只需要在运行 Stable-Diffusion-WebUI 时添加上 `xformers` 相应的参数即可：

```shell
$ CUDA_VISIBLE_DEVICES=3 python launch.py --xformers --reinstall-xformers --medvram
```

其中，

- `--reinstall-xformers`：重新安装 `xformers` 库

- `--xformers`：开启 `xformers` 来加速 Transformer

- `--medvram`：medium VRM usage

如果没有事先安装 `xformers` 库，则会自动添加 `xformers==0.0.17`。这可以通过相应的源码（`modules/launch_utils.py`）看出：

```python
def prepare_environment():
	xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.17')
    
    # 根据条件判断是否需要安装 xformers 库
    if (not is_installed("xformers") or args.reinstall_xformers) and args.xformers:
        if platform.system() == "Windows":
            # xformers 库需要 Python 3.10 及以上版本
            if platform.python_version().startswith("3.10"):
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
            else:
                print("Installation of xformers is not supported in this version of Python.")
                print("You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                if not is_installed("xformers"):
                    exit(0)
        elif platform.system() == "Linux":
            run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")
```

需要注意的是：`xformers` 库需要 Python 3.10 及以上版本。

---

最终的运行结果如下所示：

```shell
Python 3.10.8 (main, Nov 24 2022, 14:13:03) [GCC 11.2.0]
Version: v1.3.0
Commit hash: 20ae71faa8ef035c31aa3a410b707d792c8203a3
Installing xformers
Installing requirements
Launching Web UI with arguments: --xformers
Loading weights [6ce0161689] from /root/code/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
Startup time: 6.1s (import torch: 1.1s, import gradio: 1.1s, import ldm: 0.9s, other imports: 1.6s, load scripts: 0.5s, create ui: 0.5s, gradio launch: 0.2s).
Creating model from config: /root/code/stable-diffusion-webui/configs/v1-inference.yaml
LatentDiffusion: Running in eps-prediction mode
DiffusionWrapper has 859.52 M params.
Applying optimization: xformers... done.
Textual inversion embeddings loaded(0): 
Model loaded in 2.8s (load weights from disk: 0.9s, create model: 0.5s, apply weights to model: 0.3s, apply half(): 0.3s, move model to device: 0.7s).
```



### NSFW





上述操作后，项目中 `models` 目录下会有一个自带的模型 `model/Stable-diffusion/v1-5-pruned-emaonly.safetensors`。如果你想生成人像比较逼真的话需要下载一个 `chilloutmix_NiPrunedFp32Fix.safetensors` 模型文件，放到 `model/Stable-diffusion/` 目录下，另外还有很多具体人物形象的需要加载到 `model/Lora` 底下，例如我上图像迪丽热巴的那张就是下载了 `dilrabaDilmurat_v1.safetensors`，加载到了 `model/Lora`。

另外想生成更多更好的图片需要好的 prompt 和 negative prompt

还是迪丽热巴那张对应的 prompt：

```shell
hair ornament, earrings, necklace, t-shirts, looking at viewer, solo, <lora:dilrabaDilmurat_v1:1>,full body, water,
```

negative prompt：

```shell
(worst quality, low quality:1.2), watermark, username, signature, text
```

![](https://pic3.zhimg.com/80/v2-c757488323540b44140999c24b0f233a_720w.webp)

这边给大家两个网站有一个 prompt 和模型可以下载：

- [AI绘画咒语tag在线生成器](http://tag.muhou.net/)：这个可以让你自己去对应的 tag

- [https://civitai.com](https://civitai.com)：这个里面有很多好看的模型





### C 站



![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d63a6d5a752c431f8feafda87f557cc2~tplv-k3u1fbpfcp-zoom-in-crop-mark:1512:0:0:0.awebp?)



### 训练自己的模型





### 试用

- positive prompt：

  ```shell
  A tall sexy young woman with glasses and red high heels is exposing her body with a clear view of her full breasts, nipples and buttocks.
  ```

- negative prompt：

  ```shell
  man
  ```

- stable diffusion checkpoint = v1-5-pruned-emaonly.safetensors

- width = 512

- height = 512

- CFG scale = 7

- batch count = 8

- batch size = 8

- sampling method = Euler a

- sampling steps = 150

- seed = -1
- script = None
- styles = None

```shell
A tall sexy young woman with glasses and red high heels is exposing her body with a clear view of her full breasts, nipples and buttocks.
Negative prompt: man
Steps: 150, Sampler: Euler a, CFG scale: 7, Seed: 2775165695, Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, Version: v1.3.0
```

> 这是 result-1

---

positive prompt：

```shell
A tall sexy young woman with glasses and red high heels is exposing her body, her full breasts, nipples and buttocks can be clearly seen, as well as her completely naked body.
```



negative prompt：

```shell
man, NSFW
```



```shell
A tall sexy young woman with glasses and red high heels is exposing her body, her full breasts, nipples and buttocks can be clearly seen, as well as her completely naked body.
Negative prompt: man, NSFW
Steps: 150, Sampler: Euler a, CFG scale: 7, Seed: 3699179964, Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, Version: v1.3.0

Time taken: 8m 57.38sTorch active/reserved: 10891/15022 MiB, Sys VRAM: 15433/32501 MiB (47.48%)
```

> 这是 result-2



---

```shell
A tall sexy young woman with glasses and red high heels is exposing her body, her full breasts, nipples and buttocks can be clearly seen, as well as her completely naked body; NSFW
Negative prompt: man
Steps: 150, Sampler: Euler a, CFG scale: 7, Seed: 1738445376, Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, Version: v1.3.0

Time taken: 8m 55.01sTorch active/reserved: 10894/15022 MiB, Sys VRAM: 15433/32501 MiB (47.48%)
```

> 这是 result-3



---

```shell
A tall sexy young woman with glasses and red high heels is exposing her body, her full breasts, nipples and buttocks can be clearly seen; completely naked body; full body; solo; water; NSFW; photo; highly detailed; sharp focus
Negative prompt: man; watermark; username; signature; text
Steps: 150, Sampler: Euler a, CFG scale: 7, Seed: 2337875848, Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, Version: v1.3.0

Time taken: 8m 55.63sTorch active/reserved: 10894/15022 MiB, Sys VRAM: 15433/32501 MiB (47.48%)
```

> 这是 result-4

---

```shell
A tall sexy young woman with black glasses and red high heels; busty breasts, pink nipples and big buttocks; completely naked body; full body; solo; water; NSFW; photo; highly detailed; sharp focus
Negative prompt: man; watermark; username; signature; text
Steps: 150, Sampler: Euler a, CFG scale: 7, Seed: 2675034623, Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, Version: v1.3.0

Time taken: 8m 57.14sTorch active/reserved: 10882/15022 MiB, Sys VRAM: 15433/32501 MiB (47.48%)
```

> 这是 result-5

---

```shell
A tall sexy young woman with black glasses and red high heels; busty breasts; pink nipples; big buttocks; completely naked body; full body; solo; water; NSFW; photo; highly detailed; sharp focus
Negative prompt: man, male, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy
Steps: 150, Sampler: Euler a, CFG scale: 7, Seed: 4054320170, Face restoration: CodeFormer, Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, Version: v1.3.0

Time taken: 12m 9.80sTorch active/reserved: 7091/9178 MiB, Sys VRAM: 9591/32501 MiB (29.51%)
```

> 这是 result-6，使用了 **人脸修复**

---

```shell
A tall sexy young woman with black glasses and red high heels; busty breasts; pink nipples; black eyes; big buttocks; full naked body; solo; water; NSFW; photo; highly detailed; sharp focus; full face; hyperrealistic; 8k; unreal engine; dramatic;
Negative prompt: man, male, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy
Steps: 150, Sampler: Euler a, CFG scale: 7, Seed: 3986850084, Face restoration: CodeFormer, Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, Version: v1.3.0

Time taken: 9m 35.96sTorch active/reserved: 7089/9178 MiB, Sys VRAM: 9591/32501 MiB (29.51%)
```

> 这是 result-7，使用了人脸修复

---

```shell
A tall sexy young woman with black glasses and red high heels; busty breasts; pink nipples; black eyes; big buttocks; full naked body; solo; water; NSFW; photo; highly detailed; sharp focus; full face; hyperrealistic; 8k; unreal engine; dramatic;
Negative prompt: man, male, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy
Steps: 150, Sampler: Euler a, CFG scale: 7, Seed: 3986850084, Face restoration: CodeFormer, Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, Version: v1.3.0
```

> 这是 result-8，使用了人脸修复，以及 **高分辨率修复**（从 $$512 \times 512$$ 到 $$768 \times 768$$）

---

```shell
A tall sexy young woman with black glasses and red high heels; 25 years old; busty breasts; pink nipples; big buttocks; full naked body; solo; water; NSFW; highly detailed; sharp focus; hyperrealistic; 8k; dramatic;
Negative prompt: man, male, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy
Steps: 150, Sampler: Euler a, CFG scale: 7, Seed: 1473491050, Face restoration: CodeFormer, Size: 512x512, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, Denoising strength: 0.7, Hires upscale: 1.5, Hires steps: 150, Hires upscaler: Latent, Version: v1.3.0

Time taken: 33m 10.48sTorch active/reserved: 3870/5160 MiB, Sys VRAM: 5583/32501 MiB (17.18%)
```

> 这是 result-9，使用了人脸修复、**高分辨率修复**（从 $$512 \times 512$$ 到 $$768 \times 768$$）以及使用 `xformers` 来加速 Transformer

---

```shell

```

> 这是 result-10，使用了人脸修复、**高分辨率修复**（从 $$512 \times 512$$ 到 $$1024 \times 1024$$）以及使用 `xformers` 来加速 Transformer










### 试用 2

比如你看到了人家发出来搓的图，很好看，你也想搓一个类似的，但是你不知道他的咒语 把他的图下载下来，然后丢到搭建的 SD 的 Png Info 里

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/95df35c64812400dbe103b6abca8c48d~tplv-k3u1fbpfcp-zoom-in-crop-mark:1512:0:0:0.awebp?)

然后，你会得到类似于下面的咒语。然后点击Generate，搓它！！！

![](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6b7c38f0cb8e4f0eadd74ba8720218e0~tplv-k3u1fbpfcp-zoom-in-crop-mark:1512:0:0:0.awebp?)



#### LoRA

需要选择 LoRA，自己将下载好的 LoRA 丢在 `stable-diffusion-webui/models/Lora` 目录后， 这里点击 refresh 就好，不需要重启服务

然后在咒语栏里，以及下面的各项参数中，结合自己的电脑配置，进行随意配置 然后，点击 generate 等待结果出现就好：

![](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2b4ccd096add49babac0e4646f217e0f~tplv-k3u1fbpfcp-zoom-in-crop-mark:1512:0:0:0.awebp?)



不单单，可以搓类似于人物的，只要你模型和 LoRA 以及你的咒语对应，你可以搓⻛景，二次元，甚至⻋，高达，等等等。 文字生图，图生图，比较常用，我也就摸索了这么多，后面复杂的功能，比如训练 lora 之类，如有兴趣可以自行摸索~

文字生图之后，这张图也可以继续图生图~





### Batch Size & Batch Count

- **Batch Count:** 批次数量
  
  - 与占用的显存无关

- **Batch size**：每一批次要生成的图像数量。您可以在测试提示时多生成一些，因为每个生成的图像都会有所不同
  
  - 更大的 Batch Size 会占用更大的显存

生成的图像总数等于 **Batch Count** $$\times$$ **Batch size**。





### CFG scale

**Classifier Free Guidance scale**，用于控制模型应在多大程度上遵从您的提示。

- 1：大多忽略你的提示

- 3：更有创意

- 7：遵循提示和自由之间的良好平衡

- 15：更加遵守提示

- 30：严格按照提示操作

下图显示了使用固定种子值时更改 CFG 的效果。一般不要将 CFG 值设置得太高或太低。

- 如果 CFG 值太低，Stable Diffusion 将忽略您的提示

- 当它太高时，图像的颜色会饱和，形态较为固定

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4aad4857453a4db7be767e69fab06c88~tplv-k3u1fbpfcp-zoom-in-crop-mark:1512:0:0:0.awebp)



### Restore Faces（人脸修复）

勾选 restore faces 之后，Stable-Diffusion-WebUI 会自动从 GitHub 下载一些相关的模型：

```shell
# Code Former 人脸修复模型（Face Restoration）
Downloading: "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" to /root/code/stable-diffusion-webui/models/Codeformer/codeformer-v0.1.0.pth

100%|█████████████████████████████████████████████████████████| 359M/359M [00:37<00:00, 10.0MB/s]

# 人脸检测的模型
Downloading: "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" to /root/code/stable-diffusion-webui/repositories/CodeFormer/weights/facelib/detection_Resnet50_Final.pth

# Code Former（Face Inpainting）
Downloading: "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth" to /root/code/stable-diffusion-webui/repositories/CodeFormer/weights/facelib/parsing_parsenet.pth
```

如果网络无法正常访问 GitHub，还是老样子，**使用 `ghproxy.com` 代理**。

在分析了源代码之后，对 `./modules/codeformer_model.py` 进行修改：

```python
model_dir = "Codeformer"
model_path = os.path.join(models_path, model_dir)
# model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
# 修改为
model_url = 'https://ghproxy.com/https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
```

同时，在 `./modules/codeformer_model.py` 文件中可以看到：

```python
 try:
        from torchvision.transforms.functional import normalize
        from modules.codeformer.codeformer_arch import CodeFormer
        from basicsr.utils import img2tensor, tensor2img
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        from facelib.detection.retinaface import retinaface
```

`retinaface` 模型直接引用的是 `facelib` 库的。

因此，我们可以通过 `https://ghproxy.com/https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth` 这一链接进行下载，然后再使用 FTP 将 `detection_Resnet50_Final.pth` 模型权重上传到服务器的 `/stable-diffusion-webui/repositories/CodeFormer/weights/facelib/` 目录中。

同理，对于 `https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth`，也可以采用先下载，再上传到服务器相应的位置。



### Hires. fix（高分辨率修复）

高分辨率修复选项应用 **放大器** 来放大图像。您觉得需要这个选项，因为 Stable Diffussion 的本机分辨率为 512 像素（对于某些 v2 模型为 768 像素）。图像太小的话，无法用于多种用途。

为什么你不能把宽度和高度设置得更高一些，比如 1024 像素？偏离本机分辨率会影响构图，并产生一些其它问题，比如生成具有两个头部的图像。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3b82e18f64bd478cbb90982b1c68d015~tplv-k3u1fbpfcp-zoom-in-crop-mark:1512:0:0:0.awebp)

首先生成一个 512 像素的图像，然后把它放大到更大的。





### Mochi Diffusion

> 如果你是 Mac M1+，你还可以玩个新奇的东西

> 参考：
>
> - 掘金：[自己搭个StableDiffusion来AI搓图吧](https://juejin.cn/post/7239206188164563005)


### 参考

- Github Repo：[Stable-Diffusion-WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

- Github Proxy
  
  - [网址](https://ghproxy.com/)
  
  - [Github Repo](https://github.com/hunshcn/gh-proxy)

- 安装教程
  
  - Ubuntu
    
    - CSDN：[Linux Ubuntu22.04 安装stable diffusion webui（不借助科学上网的方式）](https://blog.csdn.net/weixin_42735060/article/details/130321331)
    
    - 知乎：[Ubuntu安装stable-diffusion-webui详细教程](https://zhuanlan.zhihu.com/p/611519270)
    
    - 掘金：[自己搭个StableDiffusion来AI搓图吧](https://juejin.cn/post/7239206188164563005)
  
  - Windows
    
    - CSDN：[AI绘画第一步，安装Stable-Diffusion-WebUI全过程 !](https://blog.csdn.net/wpgdream/article/details/129255469)

- 训练自己的模型
  
  - bilibili：[生成你的专属定制老婆!——使用stable-diffusion-webui的Textual Inversion功能](https://www.bilibili.com/read/cv19040576)

- 详细用法
  
  - 掘金：[Stable Diffusion 实践: 基本技法及微调](https://juejin.cn/post/7215041621040267323)
  
  - 知乎：[【翻译】Stable Diffusion prompt: a definitive guide](https://zhuanlan.zhihu.com/p/611479852)

