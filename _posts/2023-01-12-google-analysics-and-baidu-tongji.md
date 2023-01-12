---
layout: post
author: bookstall
tags: Blog
categories: [Blog]
description: Google Analysics And Baidu Tongji in Blog
keywords: Google Analysics, Baidu Tongji, Blog, Jekyll
title: Google Analysics And Baidu Tongji in Blog
---

本文主要介绍两种网站分析工具：Google Analytics（谷歌分析） 和 Baidu Tongji（百度统计）。

## Google Analytics

**Google 分析（Google Analytics，GA）**是一个由 Google 所提供的网站流量统计服务。Google 分析（Analytics）现在是互联网上使用最广泛的网络分析服务。Google Analytics 还提供了一个 SDK，允许从 iOS 和 Android 应用程序收集使用数据，称为 Google Analytics for Mobile Apps。

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Google_Analytics_Logo_2015.png/330px-Google_Analytics_Logo_2015.png)

Google Analytics 原为 Urchin 所营运的付费网站流量统计服务，2005 年 4 月，Google 宣布购并 Urchin 公司，并将原本需要付费的部分服务开放免费使用。此后，基本版可免费使用，但高级版本仍然需要付费。透过在网站中埋入Google Analytics 追踪码，网站主们可以获取进站流量的资料，包括来源、用户、设备、访问路径等，透过 Google Analytics，可以更全面的了解品牌的受众，进而为潜在客户优化购买、访问流程，提高转单意愿，对于网页入门来说是非常推荐使用的工具。

Google 于 2022 年 3 月 16 日宣布，将以新一代的成效评估解决方案 Google Analytics 分析第 4 版取代通用 Analytics 分析，自 2023 年 7 月 1 日起，标准通用 Analytics 分析资源将停止处理新的资料。

> **官网：**
> - [Google Analytics](http://www.google.com/intl/en/analytics/)（英文）
> - [Google 分析](http://www.google.cn/intl/zh-CN/analytics/)（简体中文）
> - [Google 分析](http://www.google.com/intl/zh-TW/analytics/)（繁体中文）
> - Google Analytics 帮助中心：https://support.google.com/analytics/

### 通过 GA 可以获得的统计数据

- 有多少人访问了你的网站，又有多少人在点击登陆网页后，一直浏览你的网站内容？

- 你的网站访客有哪些？基于他们在你的网站上的交互行为，你能从中了解到什么？

- 你的网站能够为移动端和 PC 端用户都提供良好的体验吗？

- 网站上的哪些页面表现最好？哪些工作需要加强？

- 访问者是如何找到你的网站的（搜索引擎，另一个网站，广告等）？

- 是否有人登陆你的网站/点击你的某个营销页面？

- 访客具体是如何与网站进行交互的？从一个页面到另一个页面的浏览顺序？

### GA 的常用词汇

在真正使用 Google Analytics 这个工具之前，我们最好先了解下相关的词汇含义，可以让我们更快地上手这个工具。

- 用户（Users）- 即访问网站的用户；可以查看选定的日期范围内有多少用户访问了网站至少一个页面。

- 报告（Reports）- Google Analytics 提供超过 50 份免费报告，并能够创建自定义报告，帮助你分析网站数据，统计流量和记录访问者行为。数据图是常见的报告形式之一。

- 会话（Sessions）– 网站用户与网站之间的交互行为。

- 流量来源（Traffic Sources）– 显示用户是如何访问到你的网站的，访问渠道包括自然搜索（Organic Search）、社交媒体（Social）、付费广告（Paid Search）、外链引流（Referral）和直接输入网址访问（Direct）。

- 活动（Campaigns）– 跟踪用户发现网站的特定方式。比如，Google Analytics 可以跟踪由 Google Ads 广告活动带来的流量。

- 页面浏览量（Pageviews）– 选定的日期范围内访问者浏览网站的页面总数。

- 跳出率（Bounce Rate）– 选定的日期范围内，用户只访问了网站上的一个页面，没有其他交互操作就退出访问的会话的百分比。

- 受众（Audiences）- 自定义的用户组。创建这些用户组，以帮助在谷歌分析报告、重新营销工作、谷歌广告活动和其他谷歌网站管理员工具中识别特定类型的用户。

- 转化和目标（Conversions & Goals）- 可以把用户某个交互行为定义为目标，以衡量业务价值。比如，用户完成在线购买或查看联系方式页面。转化表示网站用户完成已定义目标的次数。

- 漏斗（Funnels）— 用户完成设定目标经过的路径。

> 以上提到的常用词汇，先有个大概印象即可，没必要把这些都背下来。
> 
> 还可以通过 [Google Analytics 帮助中心](https://support.google.com/analytics/) 进行查询


### 注册、创建 GA 账号

> 请确保已经拥有 Google 账号！

进入 [创建页面](https://analytics.google.com/analytics/web/provision/#/provision/create) 来新建一个 GA 账号。

- 首先，填写 "账号设置" 相关的信息（这里仅选择了 "基准化分析"）。

- 然后，填写 "媒体资源设置" 相关的信息（这里将 "网络媒体资源名称" 填写为 "个人博客"）。

- 最后，填写 "企业信息" 相关的信息，并选择接受 GA 相关的服务条款协议。

经过上面的操作之后，我们成功创建了一个 GA 账号，之后自动跳转到 GA 的主页面。


### 使用

#### 设置数据流

> 要为您的网站或应用设置数据收集，请选择您将要收集数据的来源（网站、Android 应用或 iOS 应用）。接下来，您将获得有关如何向该来源添加数据收集代码的说明。

我们选择 "网站" 数据流，并且使用默认的 "增强型衡量功能"，具体包括：

- 网页浏览量
- 滚动次数
- 出站点击次数
- 网站搜索
- 表单互动次数
- 视频互动度
- 文件下载次数

最终，我们成功创建了一个 "网站" 数据流，并且能够看到该数据流的详细信息，包括：数据流名称、数据流网址、数据流 ID 以及 衡量 ID。

#### 获取代码

创建数据流之后，我们需要获取 GA 代码，并在网页中进行安装。

这里我们选择手动添加 GA 代码，可以得到如下的 GA 代码：

```html
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-S68VCV2RN6"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-S68VCV2RN6');
</script>
```

#### 安装代码

Jekyll




## 百度统计

> 官网：https://tongji.baidu.com/web/welcome/login

![](https://tongji.baidu.com/web5/image/logo.png?__v=1664186255948)

## 参考

- 维基百科：[Google 分析](https://zh.wikipedia.org/zh-cn/Google%E5%88%86%E6%9E%90)
- 知乎：[谷歌流量分析工具Google Analytics使用方法指南教程](https://zhuanlan.zhihu.com/p/136378374)






