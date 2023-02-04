---
layout: fragment
title: 在 Google 搜索结果中展示网站的图标
tags: [Favicon, Blog]
description: some word here
keywords: Favicon, Blog
date: 2023-02-04
---

## 在 Google 搜索结果中展示网站的图标

如果您的网站有网站图标，则它可以显示在与您网站对应的 Google 搜索结果中。

展示的效果如下图所示：

![Google 搜索中展示网站图标](https://developers.google.com/static/search/docs/images/favicon-google-search.png?hl=zh-cn)


### 实现

在 Google 搜索结果中显示网站图标需要满足的条件：

1. 创建网站图标

2. 在网站首页的 `<head></head>` 中添加以下的 `<link>`：
    ~~~html
    <link rel="icon" href="/path/to/favicon.ico">
    ~~~

3. Google 会在抓取首页时查找并更新网站图标。如果您对网站图标做出了更改并想将这些更改告知 Google，您可以请求 [将网站首页编入索引](https://developers.google.com/search/docs/crawling-indexing/ask-google-to-recrawl?hl=zh-cn)。更新内容可能需要几天或更长的时间才会显示在搜索结果中。

## 参考

- [Google 搜索中心：定义要在搜索结果中显示的网站图标](https://developers.google.com/search/docs/appearance/favicon-in-search?hl=zh-cn)
