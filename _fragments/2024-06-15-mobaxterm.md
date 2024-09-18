---
layout: fragment
title: MobaXterm：Window 操作系统的连接工具
tags: [工具]
description: MobaXterm：Window 操作系统的连接工具
keywords: 工具
mathjax: true
---


## MobaXterm

> Enhanced terminal for Windows with X11 server, tabbed SSH client, network tools and much more

这里 [下载](https://mobaxterm.mobatek.net/download-home-edition.html) 的是最新的 MobaXterm 24.1 Protable 版本。

## 破解：MobaXterm-GenKey

> 参考：
>
> - github: [MobaXterm-GenKey](https://github.com/malaohu/MobaXterm-GenKey)

### 使用方法

环境：

- Python 3.8

- Flask=2.1.0

- Werkzeug=2.2.2

使用步骤：

```shell
$ git clone https://github.com/malaohu/MobaXterm-GenKey.git

$ cd MobaXterm-GenKey

$ pip install --no-cache-dir -r requirements.txt

$ python app.py
```

运行结果：

```shell
$ python app.py
 * Serving Flask app 'app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.11:5000
Press CTRL+C to quit
```

然后，访问 http://127.0.0.1:5000：

![MobaXterm-GenKey 主页](/images/fragments/MobaXterm/MobaXterm_keygen_website.png)


### 注意：werkzeug 版本问题

默认 werkzeug 安装的是 `3.0.3` 版本，当运行 `python app.py` 会报错：

```shell
    from werkzeug.urls import url_quote
ImportError: cannot import name 'url_quote' from 'werkzeug.urls' (C:\Users\13697\miniconda3\envs\py3.8_test\lib\site-packages\werkzeug\urls.py)
(py3.8_test)
```

参考 [Why did Flask start failing with "ImportError: cannot import name 'url_quote' from 'werkzeug.urls'"?](https://stackoverflow.com/questions/77213053/why-did-flask-start-failing-with-importerror-cannot-import-name-url-quote-fr)，需要将其进行降级，这里降为 `2.2.2` 版本：

```shell
$ pip install werkzeug==2.2.2
```

### 结果展示

我们将得到的 `Custom.mxtpro` 文件拷贝到 MobaXterm 所在目录下：

![拷贝激活文件到 MobaXterm 所在目录](/images/fragments/MobaXterm/add_custom_file.png)

然后重新打开 MobaXterm 即可。

激活之前：

![激活之前的 MobaXterm](/images/fragments/MobaXterm/MobaXTrem_before_activated.png)

激活之后：

![激活之后的 MobaXterm](/images/fragments/MobaXterm/MobaXTrem_after_activated.png)

## 参考

- github: [MobaXterm-GenKey](https://github.com/malaohu/MobaXterm-GenKey)

- [Why did Flask start failing with "ImportError: cannot import name 'url_quote' from 'werkzeug.urls'"?](https://stackoverflow.com/questions/77213053/why-did-flask-start-failing-with-importerror-cannot-import-name-url-quote-fr)
