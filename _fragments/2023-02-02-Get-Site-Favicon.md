---
layout: fragment
title: 获取网站 Favicon 的三种方法
tags: [Favicon]
description: some word here
keywords: Favicon
---

本文记录三种获取网站 favicon 图标（Logo）URL 的方法，包括：

- 在网站首页链接后面添加 `facvion.ico`

- 使用第三方工具

- 使用一些特殊的 API


## 获取网站 Logo 图标的 URL 链接

### 1）在网站首页链接后面添加 facvion.ico

最常用的方法（适用于 90% 的站点）是：直接在访问网址首页链接后加上 `/favicon.ico`，例如：

~~~txt
https://www.baidu.com/favicon.ico
~~~

![百度 Logo](https://www.baidu.com/favicon.ico)

同时，我们也可以进入浏览器的 **开发者模式**，点开 `<head></head>`，找到包含有 `favicon` 或者 `ico` 的 `<link>` 链接，右键点击 "Edit attribute"（编辑属性）以复制该链接。

<a href="https://pic1.xuehuaimg.com/proxy/https://upload-images.jianshu.io/upload_images/26312444-d4a86f8829df7433?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp" data-caption="进入开发者模式"><img src="https://pic1.xuehuaimg.com/proxy/https://upload-images.jianshu.io/upload_images/26312444-d4a86f8829df7433?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp" alt="进入开发者模式"></a>

如下图所示：

<a href="https://pic1.xuehuaimg.com/proxy/https://upload-images.jianshu.io/upload_images/26312444-bc648e387b0e5296?imageMogr2/auto-orient/strip|imageView2/2/w/924/format/webp" data-caption="找到并复制图标 Logo 链接"><img src="https://pic1.xuehuaimg.com/proxy/https://upload-images.jianshu.io/upload_images/26312444-bc648e387b0e5296?imageMogr2/auto-orient/strip|imageView2/2/w/924/format/webp" alt="找到并复制图标 Logo 链接"></a>


### 2）第三方工具

#### The Favicon Finder

![Favicon Finder's Logo](https://favicons.teamtailor-cdn.com/icon.svg)

第三方工具 [Favicon Finder](https://favicons.teamtailor-cdn.com/) 可以在线获取目标网站的 Logo 图标，并且能够获取到不同大小的 Logo 图标。

下面是获取百度 Logo 图标的例子：

![获取百度的 Logo](/images/fragments/get-baidu-favicon.png)



### 3）使用 API

一些网站提供了专门的 API 给其他的开发者或者特殊用途使用，可以更快捷的获取到网站图标。

#### Google API

使用 `https://www.google.com/s2/favicons` 的 Google API，并且可以使用几个不同的参数。

`domain` 参数：

```txt
https://www.google.com/s2/favicons?domain=google.com
```

![Google's Logo](https://pic1.xuehuaimg.com/proxy/https://www.google.com/s2/favicons?domain=google.com)

`sz` 和 `domain_url` 参数：

```txt
https://www.google.com/s2/favicons?sz=64&domain_url=yahoo.com
```

![Yahho's Logo](https://pic1.xuehuaimg.com/proxy/https://www.google.com/s2/favicons?sz=64&domain_url=yahoo.com)


#### AllesEDV API

[Free Favicon-Service by AllesEDV.at](https://favicon.allesedv.com/)

例子：

```txt
https://f1.allesedv.com/stackoverflow.com
```

![AllesEDV-Google's Logo](https://f1.allesedv.com/google.com)

```txt
https://f1.allesedv.com/stackoverflow.com
```

![AllesEDV-StackOverflow's Logo](https://f1.allesedv.com/stackoverflow.com)



## 参考

- 简书：[怎么获取网页logo图标的URL链接](https://www.jianshu.com/p/829fcdd9de8)

- 阿里云：[下载网站 favicon 图标的 3 种方法-阿里云开发者社区](https://developer.aliyun.com/article/849529)

- StackOverflow：[How to get larger favicon from Google's api?](https://stackoverflow.com/questions/38599939/how-to-get-larger-favicon-from-googles-api)