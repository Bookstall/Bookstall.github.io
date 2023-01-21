---
layout: post
author: bookstall
tags: Blog, Jekyll
categories: [Blog]
description: Add Music Player In Blog
keywords: Music Player, Blog, Jekyll
title: 在 Jekyll 博客中添加音乐播放器
---

## 网易云音乐插件

<a href="https://picx.zhimg.com/v2-062340e8585bac2a6bf89156e12f0345_1440w.jpg" data-caption="网易云音乐">
<img src="https://picx.zhimg.com/v2-062340e8585bac2a6bf89156e12f0345_1440w.jpg" style="zoom:67%;">
</a>

**网易云音乐** 提供两种类型的插件，分别是：**iframe 插件** 和 **flash 插件**。

### iframe 插件

HTML 内联框架元素（例如 `<iframe>`）表示嵌套的浏览上下文（browsing context），它能够将另一个 HTML 页面嵌入到当前页面中。

- 优点：可以自己调整插件的高度、宽度

- 缺点：很多博客网站不支持嵌入 iframe，请试一下您的网站是否支持

### flash 插件

- 优点：可以适用大部分的博客，如网易、新浪博客等

- 缺点：无法自己调整插件的高度、宽度

### 获取外链

进入网页版的网易云音乐，选择我们想要听的歌曲。

接着点击左侧的 "生成外链播放器"，如下所示：

![生成外链播放器](https://img-blog.csdnimg.cn/5a9bf8efe62a4ff1a7397d7bcf851278.png)

即可跳转到相应的外链播放器页面，如下所示：

![外链播放器页面](https://img-blog.csdnimg.cn/4cc5dd4154e94ec99069151a9b048de0.png)

接着选择 "iframe 插件"，即可得到如应的代码：

```html
<iframe frameborder="no" border="0" marginwidth="0" marginheight="0" width=330 height=86 src="//music.163.com/outchain/player?type=2&id=1366216050&auto=1&height=66">
</iframe>
```

其中，`src` 就是相应的 URL 链接，并且：

- `type` 表示播放器的类型；

- `id` 表示歌曲对应的 id；

- `auto` 表示是否自动播放；

  - `1` 表示自动播放；`0` 表示不自动播放

- `height` 表示具体的歌曲的高度；


### 嵌入博客

将获取到的网易云音乐的 iframe 代码嵌入到博客相应的位置。同时，我们可以将其中的 `src` 链接中的 `id` 提取出来，放到 `_config.yml` 配置文件中。

```yml
# ---------------- #
#      Music       #
# ---------------- #
music_id: 368838
```

然后将 iframe 代码进行修改，如下所示：

```html
<!-- cloud music -->
<!-- auto=1 可以控制自动播放与否，当值为 1 即打开网页就自动播放，值为 0 时需要访客手动点击播放 -->
<iframe frameborder="yes" border="2" marginwidth="0" marginheight="0" width=330 height=86
    src="https://music.163.com/outchain/player?type=2&id={{ site.music_id }}&auto=0&height=66" id="music">
</iframe>
```

### 番外一：iframe 加载完成的事件

可以通过给 iframe 添加一个 id，通过 jquery 来进行监听，如下所示：

```html
<script>
    $("#music").load(function () {
        console.log('iframe 资源加载完成');
    });
</script>
```

### 番外二：添加歌单

上面涉及的仅仅是单首歌曲，如果想添加的是一个歌单，又该如何操作呢？其实也很简单！

**1. 创建歌单并添加自己喜爱的歌曲**

> 如果我们的歌单中添加了受版权保护的歌，在博客中我们的歌单就 GG 了，要避免添加此类歌曲。
> 
> ![添加了受版权保护的歌](https://img-blog.csdnimg.cn/20200409024747754.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NzIwNTk0,size_16,color_FFFFFF,t_70#pic_center)

**2. 获取歌单外链**

打开网易云音乐网页版，登录后点击我的音乐，点击分享。

![分享网易云音乐](https://img-blog.csdnimg.cn/20200409024829980.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NzIwNTk0,size_16,color_FFFFFF,t_70#pic_center)

分享成功后，点击朋友，点击动态。

![查看动态](https://img-blog.csdnimg.cn/20200409024949613.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NzIwNTk0,size_16,color_FFFFFF,t_70#pic_center)

找到我们刚才分享的歌单。

![选择歌单](https://img-blog.csdnimg.cn/20200409025032376.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NzIwNTk0,size_16,color_FFFFFF,t_70#pic_center)

**3. 创建网易云插件**

找到生成外链播放器，点击进入。

![生成外链播放器](https://img-blog.csdnimg.cn/20200409025115529.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NzIwNTk0,size_16,color_FFFFFF,t_70#pic_center)

我们可以设置网易云插件的尺寸，以及是否自动播放。

![外链播放器页面](https://img-blog.csdnimg.cn/20200409025158537.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NzIwNTk0,size_16,color_FFFFFF,t_70#pic_center)

> 此外，我们也可以使用别人的歌单来生成外链播放器！

## 参考

- CSDN：[【前端基础知识】网易云音乐iframe外链的使用](https://blog.csdn.net/weixin_46318413/article/details/127925205)

- CSDN：[创建网易云歌单外链 & Hexo](https://blog.csdn.net/qq_39720594/article/details/105423726)

- CSDN：[jekyll个人博客中添加音乐播放插件](https://blog.csdn.net/z564359805/article/details/100709964)
