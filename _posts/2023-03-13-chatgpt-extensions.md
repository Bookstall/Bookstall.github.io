---
layout: post
author: bookstall
tags: 工具, ChatGPT
categories: [工具, ChatGPT]
excerpt: 本文整理了 7 款 ChatGPT 浏览器插件
keywords: 工具， ChatGPT
title: ChatGPT 浏览器插件
mathjax: true
---

根据博客 [推荐7款非常实用的 ChatGPT 浏览器插件！](https://www.freedidi.com/8670.html)，本文整理了 7 款 ChatGPT 浏览器插件，包括：

- WebChatGPT：让 ChatGPT 联网

  - 开源

- ChatGPT for Google：可以在所有主流搜索引擎（Google 等）的右侧同步给出 ChatGPT 的结果

  - 开源

  - **ChatGPT 版的 New Bing**

- YouTube Summary with ChatGPT：对 YouTube 视频做 summary

  - Transcript -> Summary

- Summarize：通过 ChatGPT 帮助我们读文章

- ChatGPT Writer：使用 ChatGPT 来写 Email、messages

- LINER：基于 ChatGPT 的搜索助手，高亮标注插件

  - 需要创建账号

- Voice Control for ChatGPT：基于 ChatGPT 的语音控制插件

  - 与 ChatGPT 进行语音对话

  - Transcript -> ChatGPT

其中，本人正在使用的是 **ChatGPT for Google** 插件，极力推荐。



## WebChatGPT

目前 ChatGPT 仅限于 2021 年以前的信息，但是通过这款扩展，它可以访问互联网上的最新信息。

> WebChatGPT: **A browser extension that augments your ChatGPT prompts with web results.**

大致原理：首先将问题丢给搜索引擎，再将搜索引擎的结果丢给 ChatGPT，以实现 ChatGPT 联网。

> - [github repo](https://github.com/qunash/chatgpt-advanced/)
>
> - [Google Chrome 下载地址](https://chrome.google.com/webstore/detail/chatgpt-advanced/lpfemeioodjbpieminkklglpmhlngfcn)
>
> - [Firefox 下载地址](https://addons.mozilla.org/en-US/firefox/addon/web-chatgpt/)

WebChatGPT 演示视频如下所示：

<video src="https://user-images.githubusercontent.com/3750161/214155508-5c1ad4d8-b565-4fe0-9ce7-e68aed11e73d.mp4" width="100%" height="360" controls="controls"></video>


## ChatGPT for Google

ChatGPT for Google：在搜索引擎结果中同时显示 ChatGPT 的回答

> - [github repo](https://github.com/wong2/chatgpt-google-extension)
>
> - [Google Chrome 下载地址](https://chatgpt4google.com/chrome?utm_source=github)
>
> - [Firefox 下载地址](https://chatgpt4google.com/firefox?utm_source=github)

![ChatGPT for Google's Demo](/images/posts/google-chatgpt-demo.jpg)

### 功能点

* 支持所有主流的搜索引擎

  * Google, Baidu, Bing, DuckDuckGo, Brave, Yahoo, Naver, Yandex, Kagi, Searx

* 支持 OpenAI 官方 API

* 从插件弹窗里快速使用 ChatGPT

* 支持 Markdown 渲染

* 支持代码高亮

* 支持深色模式

* 可自定义 ChatGPT 触发模式

- 支持 Chat Contents Share


### 结果分享

通过 ChatGPT for Google 插件，我们还可以将聊天结果进行 **分享**。但是，在分享的时候，需要将聊天结果 <u>上传至插件作者的服务器上</u>。如下所示：

![结果分享](/images/posts/share-chatgpt-content.png)

成功分享之后，会跳转到一个新的页面中，里面包含了我们的聊天结果。下面是两个分享聊天结果的例子：

- https://webapp.chatgpt4google.com/s/MjE4NTMz

- https://webapp.chatgpt4google.com/s/MjQzNjU4

![Share URL Demo](/images/posts/share-demo.jpg)


## YouTube Summary with ChatGPT

> - [主页](https://glasp.co/youtube-summary)
>
> - [Google Chrome 下载地址](https://chrome.google.com/webstore/detail/youtube-summary-with-chat/nmmicjeknamkfloonkhhcjmomieiodli)
>
> - [Safari 下载地址](https://apps.apple.com/us/app/glasp-social-web-highlighter/id1605690124)

当你在 YouTube 上浏览视频时，如果视频的长度特别长的话，为了节省时间，通过单击视频缩略图上的摘要按钮，来快速查看 **视频摘要**。

使用此扩展程序可以节省时间并更快地学习。

![YouTube Summary with ChatGPT](https://pic1.xuehuaimg.com/proxy/https://lh3.googleusercontent.com/mkkL-dX0769Ply3CRSPO1BN6-XLr8wed1biKR3xjnAXOAIoWv9SVsDqp0T1iPH5OtiLFN-w3xOobddpnzG1PcBPC=w640-h400-e365-rj-sc0x00ffffff)


## Summarize

> - [Google Chrome 下载地址](https://chrome.google.com/webstore/detail/summarize/lmhkmibdclhibdooglianggbnhcbcjeh)

要使用 Summarize，你只需打开任何内容，可以是文章、电子邮件或任何其他网站，然后单击扩展程序，它将发送请求到 ChatGPT，可以在几秒钟内提供一个简洁的摘要。

![Summarize](https://pic1.xuehuaimg.com/proxy/https://lh3.googleusercontent.com/B0l-_JbAcYsmMqLSmffyrr0DTpn7I-0-T_0A01uyIU-ofN3YSxt86NNrJecOmxqvMHwaM7igxQBxNEuTn45H6QrwuHY=w640-h400-e365-rj-sc0x00ffffff)


## ChatGPT Writer

> - [主页](https://chatgptwriter.ai)
> 
> - [Google Chrome 下载地址](https://chrome.google.com/webstore/detail/chatgpt-writer-write-mail/pdnenlnelpdomajfejgapbdpmjkfpjkp)

这个插件可以帮我们在网站上，写邮件或是回复信息，该插件登录 OpenAI 后就可以单独使用。

你需要点击扩展程序来打开它，它会问你想要写什么，然后会输出内容，如果需要回复邮件，则把要回复的邮件内容输入进去，它在 Gmail 上的效果更好。

![ChatGPT Writer](https://pic1.xuehuaimg.com/proxy/https://lh3.googleusercontent.com/xYEUxLRUAV9ttGTBpOOsVlYip00dGgNWiFdLVfvfFpX4dZLv8zfABAKG600YX0mx4C6mGUN1m_wTgOfoDYosQO7Y=w640-h400-e365-rj-sc0x00ffffff)

<video src="https://chatgptwriter.ai/videos/chatgpt_writer_small_demo_compressed.mp4" controls="controls" width="100%" height="360"></video>


## LINER

> - [Google Chrome 下载地址](https://chrome.google.com/webstore/detail/liner-chatgpt-google-assi/bmhcbmnbenmcecpmpepghooflbehcack)

**LINER: ChatGPT 搜寻助手 & 网页/Youtube 荧光笔软件**

基于 OpenAI ChatGPT Google 搜索助手，具有 Web 和 YouTube 突出显示功能；

可以在茫茫资讯海之中，帮我们更加迅速地掌握关键内容，让使用者们筛选的网路资源更加方便。

> LINER **需要创建账号**！普通用户只能免费使用 LINER Basic 版本！


## Voice Control for ChatGPT

> - [Google Chrome 下载地址](https://chrome.google.com/webstore/detail/voice-control-for-chatgpt/eollffkcakegifhacjnlnegohfdlidhn)

通过语音控制和大声朗读扩展 ChatGPT。此扩展使您能够在 Chrome 中与来自 OpenAI 的 ChatGPT 进行 **语音对话**。它在输入字段下方注入了一个额外的按钮。

单击后，该扩展程序将录制您的声音并将您的问题提交给 ChatGPT。

![Voice Control for ChatGPT's Demo](https://pic1.xuehuaimg.com/proxy/https://lh3.googleusercontent.com/9rNrSS8-sh-H1UIHivjWfNEYCgXzEt1vBnHEIX08PFMJZvCRMoxRLm0cSMg5UHcy5gjauLqbG12oC2J2ReEoUL-dKg=w640-h400-e365-rj-sc0x00ffffff)

注意：此插件 **暂时只支持 Google Chrome 浏览器**。



## 参考

- 零度解说

  - 博客：[推荐7款非常实用的 ChatGPT 浏览器插件！](https://www.freedidi.com/8670.html)

  - bilibili：[推荐7款 ChatGPT 精品插件！支持Chrome、Edge、火狐等主流浏览器，大大提供我们的学习、工作效率](https://www.bilibili.com/video/BV1vv4y187y1/)


