---
layout: fragment
title: Clash-for-Linux 的安装与使用
tags: [Linux, HuggingFace]
excerpt: Clash-for-Linux 的安装与使用
keywords: Linux, HuggingFace
mathjax: true
---

## 1、Clash-for-Linux 的使用

由于原始的 [Clash](https://github.com/Dreamacro/clash) 作者删库跑路了，这里使用另外的一个仓库 [clash-for-linux](https://github.com/wnlen/clash-for-linux)。

### 1.1、下载

下载项目：

```shell
$ git clone https://github.com/wnlen/clash-for-linux.git
```

### 1.2、启动

进入到项目目录，编辑 `.env` 文件，修改变量 `CLASH_URL` 和 `Secret` 的值。

```shell
$ cd clash-for-linux
$ vi .env 
```

> 注意： `.env` 文件中的变量 `CLASH_SECRET `为自定义 `Clash Secret`，值为空时，脚本将 **自动生成随机字符串**。

运行启动脚本：

> 注意：
> 
> 每次 `bash start.sh` 都会下载 `config.yaml`，从而覆盖掉我们已经修改的 `config.yaml`

```shell
$ sudo bash start.sh 
CPU architecture: amd64

正在检测订阅地址...
Clash订阅地址可访问！                                      [  OK  ]

正在下载Clash配置文件...
配置文件config.yaml下载成功！                              [  OK  ]

判断订阅内容是否符合clash配置文件标准:
解码后的内容不符合clash标准，尝试将其转换为标准格式
配置文件已成功转换成clash标准格式

正在启动Clash服务...
服务启动成功！                                             [  OK  ]

Clash Dashboard 访问地址: http://<ip>:9090/ui
Secret: f5954a5fc38d9ee0761f17b4d1227b8e9ec1213695a6f7f9c46fe864b5d17163

请执行以下命令加载环境变量: source /etc/profile.d/clash.sh

请执行以下命令开启系统代理: proxy_on

若要临时关闭系统代理，请执行: proxy_off

$ source /etc/profile.d/clash.sh

$ proxy_on
[√] 已开启代理
```

- 检查服务端口

```shell
$ netstat -tln | grep -E '9090|789.'
tcp6       0      0 :::9090                 :::*                    LISTEN     
tcp6       0      0 :::7892                 :::*                    LISTEN     
tcp6       0      0 :::7890                 :::*                    LISTEN     
tcp6       0      0 :::7891                 :::*                    LISTEN 
```

如果 Linux 系统中还未安装 `netstat` 命令，则需要安装 `net-tools`：

```shell
$ apt-get install net-tools
```

- 检查环境变量

```shell
$ env | grep -E 'http_proxy|https_proxy'
https_proxy=http://127.0.0.1:7890
http_proxy=http://127.0.0.1:
```

### 1.3、重启

如果需要对 Clash 配置进行修改，请修改 `conf/config.yaml` 文件，然后运行 `restart.sh` 脚本进行重启。

```shell
$ sudo bash restart.sh
服务关闭成功！                                             [  OK  ]
服务启动成功！                                             [  OK  ]
```

### 1.4、停止 / 退出

关闭 Clash 服务：

```shell
$ sudo bash shutdown.sh
服务关闭成功，请执行以下命令关闭系统代理：proxy_off

$ proxy_off
[×] 已关闭代理
```

然后检查程序端口、进程以及环境变量 `http_proxy|https_proxy`，若都没则说明服务正常关闭。

### 1.5、Clash Dashboard

访问 Clash Dashboard：

通过浏览器访问 `start.sh` 执行成功后输出的地址，例如：http://127.0.0.1:9090/ui。

登录管理界面：

在 `API Base URL` 一栏中输入：`http://<ip>:9090`，在 `Secret(optional)` 一栏中输入启动成功后输出的 Secret。

点击 `Add` 并选择刚刚输入的管理界面地址，之后便可在浏览器上进行一些配置。

> 关于 Clash Dashboard 更多的内容可以参见 [yacd](https://github.com/haishanh/yacd) 项目


## 2、批量下载 HuggingFace 模型

### 2.0、背景

之前下载 huggingface 上模型的时候，要么是用类似如下脚本的方式下载：

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
  
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
```

但是脚本自动下载下来的文件命名非常的乱……

---

要么是在 Files and versions 中点击目标文件逐一下载：

![HuggingFace 中的 Files and versions](https://img-blog.csdnimg.cn/742abe36267d49dd940e0e70bd4451a8.png)

这种手动点击下载的方式非常麻烦，模型文件一旦多起来，手要废掉了……

---

因此，我们使用 `git lfs clone` 的方式来进行自动下载。但是，由于 `huggingface.co` 被墙了，直接使用 `git lfs clone` 往往会出现网络问题。这个时候，我们就需要考虑使用一些魔法来解决。

### 2.1、Github 代理配置

由于 HuggingFace 的部分模型和数据集在国外服务器，不使用代理比较慢，所以要先配置 `git` 代理。

- 全局代理配置方式：

```shell
git config --global https.proxy http://127.0.0.1:7890
```

- 只对 `clone` 使用代理的配置方式：

```shell
git clone XXX.git -c http.proxy="http://127.0.0.1:7890"
```

### 2.2、使用 git lfs 进行安装

我们首先点击下图中圈起来的地方，然后点击【Clone repository】，在命令行中，输入以下命令进行安装：

```shell
git lfs install # 安装了这个，才会下载大文件，不然图中的 .bin 文件都是不会被下载的

git clone https://huggingface.co/THUDM/chatglm2-6b -c http.proxy="http://127.0.0.1:7890"
```

![](https://img-blog.csdnimg.cn/cd931622e07145c8a5d557fc5839a5e7.png)

![Clone this model repository](https://img-blog.csdnimg.cn/b38747b3c72f4392a574f845b57e9d1a.png)


---

当未使用代理的情况：

```shell
$ git lfs install
Updated git hooks.
Git LFS initialized.

$ git clone https://huggingface.co/THUDM/chatglm2-6b 
Cloning into 'chatglm2-6b'...
fatal: unable to access 'https://huggingface.co/THUDM/chatglm2-6b/': Failed to connect to huggingface.co port 443: Connection refused
```

当使用了代理的情况，可以成功下载 HuggingFace 模型：

```shell
$ git lfs install
Updated git hooks.
Git LFS initialized.

$ git clone https://huggingface.co/THUDM/chatglm2-6b -c http.proxy="http://127.0.0.1:7890"
Cloning into 'chatglm2-6b'...
remote: Enumerating objects: 186, done.
remote: Counting objects: 100% (186/186), done.
remote: Compressing objects: 100% (81/81), done.
remote: Total 186 (delta 104), reused 186 (delta 104), pack-reused 0
Receiving objects: 100% (186/186), 1.92 MiB | 1.98 MiB/s, done.
Resolving deltas: 100% (104/104), done.
Filtering content: 100% (8/8), 11.63 GiB | 10.70 MiB/s, done.
```


## 3、参考

- Github：[clash-for-linux](https://github.com/wnlen/clash-for-linux)

- CSDN：

  - [如何批量下载 hugging face 模型和数据集文件](https://blog.csdn.net/zhaohongfei_358/article/details/126222999)

  - [通过 clone 的方式，下载 huggingface 中的大模型（git lfs install）](https://blog.csdn.net/qq_40600379/article/details/132006217)

  - [下载 huggingface 上模型的正确姿势](https://blog.csdn.net/ljp1919/article/details/125977360)
