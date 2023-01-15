---
layout: post
author: bookstall
tags: Google Adsense, Blog
categories: [Google Adsense, Blog]
description: 
keywords: Google Adsense, Blog, Jekyll
title: Google AdSense In Jekyll's Blog
---

本文主要介绍如何使用 Google AdSense 来接入广告，并将广告投放到自己的博客网站中。

## 介绍

Google Adsense 是一个由 Google 公司设置的广告计划，会员可以利用 Youtube 流量和 Blogspot 功能置入广告服务，以赚取佣金。会员可分得搜寻广告收益的 51%、内容广告收益的 68%。

![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/AdSense_Logo.svg/330px-AdSense_Logo.svg.png)



## 获取代码并接入网站

首先，需要有一个 Google 账号，然后在 [Google AdSense - 利用网站创收](https://www.google.com/intl/zh-CN_cn/adsense/start/) 中进行登录。

接着，打开 [Google AdSense](https://www.google.com/adsense/) 网页，会提示你在 `<head></head>` 中加入类似的 AdSense 代码：

```html
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-1120557063402952" crossorigin="anonymous"></script>
```


## 申请 Google AdSense


申请结束之后，Google AdSense 会给出如下的提示：

```text
正在处理中。我们正在对您的网站进行一些检查，通常，此过程几天之内就能完成，但在某些情况下最长可能需要 2 周时间。等到您的网站达到展示广告的要求后，我们会通知您。
```


## 广告投放


### 自动广告

自动广告的代码就是之前嵌入的认证身份的代码，只需要打开自动广告即可：

![](https://last2win.com/images/posts/2020-1-24-google-ads.png)

如果你觉得自动广告的位置不好，可以手动选择广告放的位置：展示广告，信息流广告，文章内嵌广告。



## 添加 ads.txt

> **参考：**
> 
> - 官方文档：[ads.txt 指南](https://support.google.com/adsense/answer/7532444)

谷歌会提醒你需要在域名根目录下放置文件 `ads.txt`，这样可以更好的识别你的网站。

**授权数字卖家 （`ads.txt`）**是一项 IAB Tech Lab 计划，旨在协助您仅通过认定的授权卖家（如 AdSense）销售您的数字广告资源。创建自己的 `ads.txt` 文件后，您可以更好地掌控允许谁在您的网站上销售广告，并可防止向广告客户展示仿冒广告资源。

我们强烈建议您使用 `ads.txt` 文件。它可以帮助买家识别仿冒广告资源，并可以帮助您获得更多广告客户支出，而这些支出原本可能会流向仿冒广告资源。






## 参考

- 维基百科：[Google AdSense](https://zh.wikipedia.org/zh-cn/Google_AdSense)

- 博客：[github建站系列(17) -- 为你的 blog 添加google adsence 广告](https://kebingzao.com/2020/12/07/github-site-17/)

- 博客：[给 Jekyll Blog 添加 AdSense 广告](https://gohalo.me/post/add-google-adsense-to-jekyll-blog.html)

- 官方文档：[ads.txt 指南](https://support.google.com/adsense/answer/7532444)