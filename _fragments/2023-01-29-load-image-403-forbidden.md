---
layout: fragment
title: 网络图片访问报 403（防止盗链）
tags: [图片防盗链]
description: some word here
keywords: 图片防盗链
date: 2023-01-29
---

## 访问网络图片报 403

直接引用 CSDN 的一些图片地址时，发现无法正常显示，并且报 403 forbidden 的错误。


### 解决方法

#### 1、使用 no-referrer

Referrer-Policy 头部用来监管哪些访问来源信息——会在 Referer 中发送——应该被包含在生成的请求当中。

![Referrer-Policy 的一些状态](https://pic1.xuehuaimg.com/proxy/https://img-blog.csdnimg.cn/20190924105626276.png)

no-referrer：

1. 整个 Referer 首部包含了当前请求页面的来源页面的地址，即表示当前页面是通过此来源页面里的链接进入的。

2. 服务端一般使用 Referer 首部识别访问来源，可能会以此进行统计分析、日志记录以及缓存优化等。

3. 首部会被移除。访问来源信息不随着请求一起发送。


具体可以参见 [官方 MDN 文档的定义：Referrer-Policy](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Headers/Referrer-Policy)。

---

> 反防盗链的原理：隐藏请求体中标注来源 referrer 字段，referrer 字段只能隐藏，不能定制，这样服务器端的防盗链就无法检测。

在 head 中添加如下代码：

~~~html
<meta name="referrer" content="no-referrer" />
~~~

这种方案不仅针对图片的防盗链,还可以是其他标签：

~~~html
<!-- a 标签的 referrer -->
<a href="http://example.com" referrer="no-referrer|origin|unsafe-url">xxx</a>

<!-- img/image 标签的 referrer -->
<img referrer="no-referrer|origin|unsafe-url" src="{{item.src}}"/>
<image referrer="no-referrer|origin|unsafe-url" src="{{item.src}}"></image>
~~~


#### 2、使用图片镜像缓存服务


**图片镜像缓存服务可以用来做什么？**

- 可以将有防盗链的图片引用到网页，并成功显示；

- 可以将 http 图片引用到 https 页面而不出现证书问题；

- 可以将 xxx 的图片，成功加载；

- 可以将比较慢的图片资源，加快显示；

---

**https://images.weserv.nl 是一款图片镜像缓存服务**，原用于加速图片访问，但有时候有很多妙用。

- 比如 imgur 等国内无法访问图床的图片，使用它就能够访问了；

- 又比如一些防盗链网站的图片是无法直接放在自己的博客上使用的，例如知乎、微信等，但图片地址上加上镜像缓存服务就可以了；

---

以下是网络中收集的一些 **图片镜像缓存服务**：

~~~shell
https://img.noobzone.ru/getimg.php?url=

https://collect34.longsunhd.com/source/plugin/yzs1013_pldr/getimg.php?url=

https://ip.webmasterapi.com/api/imageproxy/

https://images.weserv.nl/?url=

https://pic1.xuehuaimg.com/proxy/ # 亲测有效

https://search.pstatic.net/common?src=
~~~

## 防盗链原理

**防盗链，就是防你盗用我的链接**。你在你的网站上引用了我的资源（图片、音频等），你跑起来倒是没什么事，但是会**浪费我的流量**，资源被引用的多了起来，我这边的服务器可能就扛不住挂了，你说这是多么悲哀的事情！

一般情况下 **图片防盗链** 的场景居多，下面我们来看看图片防盗链是如何做的。

### 图片防盗链

#### 场景

先来看个图，这个图是我在本地启了一个服务后，分别加载了 <u>百度和 360 搜索</u> 两个网站的图片链接，对应防盗链下的样子：

![图片防盗链的场景](https://p1-jj.byteimg.com/tos-cn-i-t2oaga2asx/gold-user-assets/2018/4/22/162ebce706ced357~tplv-t2oaga2asx-zoom-in-crop-mark:4536:0:0:0.image)

---

那么，图片防盗链是如何做到的呢？

![图片的请求头](https://p1-jj.byteimg.com/tos-cn-i-t2oaga2asx/gold-user-assets/2018/4/22/162ebd38ea6041e3~tplv-t2oaga2asx-zoom-in-crop-mark:4536:0:0:0.image)

如上图所示，在请求头中有 **Host（请求的主机）** 和 **Referer（来源）** 两个参数，之所以会形成防盗链，那是因为 **Host 和 Referer 对应的值不相同**。


#### 具体做法

~~~javascript
// js部分
const fs = require('fs');
const path = require('path');
const http = require('http');
const url = require('url');
const getHostName = function (str) {
    let { hostname } = url.parse(str);
    return hostname;
};

http.createServer((req, res) => {
    let refer = req.headers['referer'] || req.headers['referrer'];  // 请求头都是小写的
    // 先看一下refer的值，去和host的值作对比，不相等就需要防盗链了  
    // 要读取文件 返回给客户端
    let { pathname } = url.parse(req.url);
    let src = path.join(__dirname, 'public', '.' + pathname);
    // src代表我要找的文件
    fs.stat(src, err => {   // 先判断文件存不存在
        if (!err) {
            if (refer) {    // 不是所有图片都有来源
                let referHost = getHostName(refer);
                let host = req.headers['host'].split(':')[0];
                
                if (referHost !== host) {
                    // 防盗链
                    fs.createReadStream(path.join(__dirname, 'public', './1.jpg')).pipe(res);
                } else {
                    // 正常显示，如果路径存在，可以正常显示直接返回
                    fs.createReadStream(src).pipe(res);
                }
            } else {
                // 正常显示，如果路径存在，可以正常显示直接返回
                fs.createReadStream(src).pipe(res);
            }
        } else {
            res.end('end');
        }
    });
}).listen(8888);
~~~

以上内容就实现了如何做一个 **图片防盗链**，防止别人使用你的资源，当然不仅仅是 **图片防盗链，音频、视频等** 也可以根据此方法实现，之后大家也可以在工作中尝试尝试。

> 通过请求头做的事情还是比较多的，像 **断点续传** 这样的操作也都是如此。


## 拓展

meta 一览表：

~~~html
<meta charset="utf-8"> <!-- 设置文档字符编码 -->
<meta http-equiv="x-ua-compatible" content="ie=edge"><!-- 告诉IE浏览器，IE8/9及以后的版本都会以最高版本IE来渲染页面。 -->
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no"><!-- 指定页面初始缩放比例。-->

<!-- 上述3个meta标签须放在head标签最前面;其它head内容放在其后面，如link标签-->

<!-- 允许控制加载资源 -->
<meta http-equiv="Content-Security-Policy" content="default-src 'self'">
<!-- 尽可能早的放在文档 -->
<!-- 只适用于下面这个标签的内容 -->

<!-- 使用web应用程序的名称(当网站作为一个应用程序的时候)-->
<meta name="application-name" content="Application Name">

<!-- 页面的简短描述(限150个字符)-->
<!-- 在某些情况下这个描述作为搜索结果中所示的代码片段的一部分。-->
<meta name="description" content="A description of the page">

<!-- 控制搜索引擎爬行和索引的行为 -->
<meta name="robots" content="index,follow,noodp"><!-- 所有搜索引擎 -->
<meta name="googlebot" content="index,follow"><!-- 谷歌 -->

<!-- 告诉谷歌搜索框不显示链接 -->
<meta name="google" content="nositelinkssearchbox">

<!-- 告诉谷歌不要翻译这个页面 -->
<meta name="google" content="notranslate">

<!-- Google网站管理员工具的特定元标记，核实对谷歌搜索控制台所有权 -->
<meta name="google-site-verification" content="verification_token">

<!-- 说明用什么软件构建生成的网站，(例如,WordPress,Dreamweaver) -->
<meta name="generator" content="program">

<!-- 简短描述你的网站的主题 -->
<meta name="subject" content="your website's subject">

<!-- 很短(10个词以内)描述。主要学术论文 -->
<meta name="abstract" content="">

<!-- 完整的域名或网址 -->
<meta name="url" content="https://example.com/">

<meta name="directory" content="submission">

<!-- 对当前页面一个等级衡量，告诉蜘蛛当前页面在整个网站中的权重到底是多少。General是一般页面，Mature是比较成熟的页面，Restricted代表受限制的。 -->
<meta name="rating" content="General">

<!-- 隐藏发送请求时请求头表示来源的referrer字段。 -->
<meta name="referrer" content="no-referrer">

<!-- 禁用自动检测和格式的电话号码 -->
<meta name="format-detection" content="telephone=no">

<!-- 通过设置“off”,完全退出DNS队列 -->
<meta http-equiv="x-dns-prefetch-control" content="off">

<!-- 在客户端存储 cookie，web 浏览器的客户端识别-->
<meta http-equiv="set-cookie" content="name=value; expires=date; path=url">

<!-- 指定要显示在一个特定框架中的页面 -->
<meta http-equiv="Window-Target" content="_value">

<!-- 地理标签 -->
<meta name="ICBM" content="latitude, longitude">
<meta name="geo.position" content="latitude;longitude">
<meta name="geo.region" content="country[-state]"><!-- 国家代码 (ISO 3166-1): 强制性, 州代码 (ISO 3166-2): 可选; 如 content="US" / content="US-NY" -->
<meta name="geo.placename" content="city/town"><!-- 如 content="New York City" -->
~~~


## 参考

- CSDN：[网络图片访问不到，403的解决办法（详解）](https://blog.csdn.net/weixin_43909743/article/details/119137927)

- CSDN：[访问图片出现403的解决办法](https://blog.csdn.net/weixin_45272449/article/details/100896116)

- [图片镜像缓存服务—防盗链图片、imgur 等国内无法访问图片的解决方案](https://funletu.com/10538/.html)

- 掘金：[亲，你的防盗链钥匙，在我手上](https://juejin.cn/post/6844903596937461773)

