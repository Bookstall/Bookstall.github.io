---
layout: post
author: bookstall
tags: Comment, Blog
categories: [Blog, Comment]
description: Some Comment Systems In Static Blog
keywords: Comment, Blog, Jekyll
title: Comment System In Static Blog
---


> - 基于第三方：Disqus；LiveRe（来必力）；Valine；
> - 基于 Github Issue：Gitment；Gitalk；Utterances；Vssue；
> - 基于 Github Discussions：giscus；
> - 纯静态：Staticman；
> - 自建：ISSO；HashOver；

## 基于第三方

### Disqus

> 官网：https://disqus.com

![](https://cdn.sspai.com/2020/12/25/3122c52309255a115249cb8d99e988ed.png?imageMogr2/auto-orient/quality/95/thumbnail/!1420x708r/gravity/Center/crop/1420x708/interlace/1)


**缺点：**

- Disqus 已经无法在大陆使用；


### LiveRe（来必力）

> 官网：https://livere.com

韩国的一家评论系统。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9ydzAtMTI1NzE4MzA2MC5jb3MuYXAtY2hlbmdkdS5teXFjbG91ZC5jb20vYmxvZy1pbWcvMjIuanBn?x-oss-process=image/format,png)

**特点：**

- 支持中文；
- 使用邮箱注册；
- 支持国内 **多家社交平台** 登录；
- 颜值很高；


### Valine

> 官方地址：https://valine.js.org/
> 
> 项目地址：https://github.com/xCss/Valine

Valine 诞生于 2017 年 8 月 7 日，一款基于 LeanCloud 的快速、简洁且高效的无后端评论系统。

理论上，Valine 支持但不限于静态博客，目前已有 Hexo、Jekyll、Typecho、Hugo、Ghost、Docsify 等博客和文档程序在使用 Valine。

<img src="https://valine.js.org/images/valine.png" style="zoom:50%;" />

**特点：**

- 快速
- 安全
- Emoji 😉
- 无后端实现
- MarkDown 全语法支持
- 轻量易用
- 文章阅读量统计（v1.2.0+）

## 基于 Github Issue

### Gitment

> 项目地址：https://github.com/imsun/gitment
> 
> 中文介绍：[Gitment：使用 GitHub Issues 搭建评论系统](https://imsun.net/posts/gitment-introduction/)

Gitment 是一款基于 GitHub Issues 的评论系统。

配置参考：https://imsun.net/posts/gitment-introduction/

**特点：**

- 支持在前端直接引入，不需要任何后端代码；
- 可以在页面进行登录、查看、评论、点赞等操作;
- 有完整的 Markdown / GFM 和代码高亮支持，尤为适合各种基于 GitHub Pages 的静态博客或项目页面；

**缺点：**

- Gitment 只能使用 GitHub 账号进行评论；
- Gitment 项目长期未维护；



### Gitalk

> 官网：https://gitalk.github.io
>
> 项目地址：https://github.com/gitalk/gitalk

Gitalk 是一个基于 Github Issue 和 Preact 开发的评论插件。

配置参考：https://github.com/gitalk/gitalk#install

**特点：**

- 使用 GitHub 登录；

- 支持多语言 [en, zh-CN, zh-TW, es-ES, fr, ru, de, pl, ko, fa, ja]；

- 支持个人或组织；

- 无干扰模式（设置 `distractionFreeMode` 为 `true` 开启）；

- 快捷键提交评论 （`ctrl + enter`）；

### Utterances

> 官方地址：https://utteranc.es/
>
> 项目地址：https://github.com/utterance/utterances

一个基于 GitHub Issue 的轻量级评论小部件。

配置参考：https://utteranc.es/



### Vssue

> 官方地址：https://vssue.js.org/
>
> 项目地址：https://github.com/meteorlxy/vssue

Vssue 是一个 Vue 组件 / 插件，可以为你的静态页面开启评论功能。

Vssue 名字的由来：由 Vue 驱动并基于 Issue 实现。

Demo 演示网址：https://vssue.js.org/demo/

<img src="https://vssue.js.org/logo.png" style="zoom: 50%" />

#### 与其他评论系统的区别

Vssue 的灵感来自于 Gitment 和 Gitalk，但是和它们有些区别：

- Vssue 支持 Github、Gitlab、Bitbucket、Gitee 和 Gitea，并且很容易扩展到其它平台；
  - Gitment 和 Gitalk 仅支持 Github

- Vssue 可以发表、编辑、删除评论；
  - Gitment 和 Gitalk 仅能发表评论

- Vssue 是基于 Vue.js 开发的，可以集成到 Vue 项目中，并且提供了一个 VuePress 插件；
  - Gitment 基于原生 JS，而 Gitalk 基于 Preact


## 基于 GitHub Discussions

> GitHub Discussions 文档：https://docs.github.com/en/discussions

### giscus

> 项目地址：https://github.com/giscus/giscus or https://gitee.com/mirrors/giscus

![](https://avatars.githubusercontent.com/ml/9968?s=200&v=4)

#### 特点

- 开源；

- 无跟踪，无广告，永久免费；

- 无需数据库。全部数据均储存在 GitHub Discussions 中；

- 支持自定义主题；

- 支持多种语言；

- 高度可配置；

- 自动从 GitHub 拉取新评论与编辑；

- 可自建服务；

#### 使用方法

> 配置参考：https://giscus.app/zh-CN
>
> 使用参考：[Use Utterances/Giscus for Jekyll Comments System](https://lazyren.github.io/devlog/use-utterances-for-jekyll-comments.html) 和 [Giscus的基础设置](https://www.michaeltan.org/posts/giscus/)

1. 需要新建一个存储评论的仓库；
2. 为该仓库安装 [giscus app](https://github.com/apps/giscus)；
3. [开启 Github Discussions 功能](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/enabling-or-disabling-github-discussions-for-a-repository)；


## 纯静态

### Staticman

> 官方网址：https://staticman.net/
>
> 项目地址：https://github.com/eduardoboucas/staticman

![](https://github.com/eduardoboucas/staticman/raw/master/logo.png)

纯静态评论系统指用户进行评论生成内容等操作时，将输入内容与其余内容一起重新生成并上传到 GitHub repo 中。

具体的演示 demo 可以参考：http://popcorn.staticman.net/

## 自建

### ISSO

> 官方网址：https://isso-comments.de/
>
> 项目地址：https://github.com/posativ/isso

![](https://isso-comments.de/_static/isso.svg)

- 与 disqus 类似的评论系统；
- 用户可以编辑或删除自己的评论（默认为 15 分钟内）；
- 支持使用 Markdown 编写评论；
- 可以轻松迁移自己的 Disqus / WordPress 评论；



### HashOver

> 官方网址：https://www.barkdull.org/software/hashover
>
> HashOver 1.0 项目地址：https://github.com/jacobwb/hashover
>
> HashOver 2.0 项目地址：https://github.com/jacobwb/hashover-next

<img src="https://cdn.barkdull.org/software/hashover/hashover-logo.png" style="zoom:50%;" />

一个免费开源的 PHP 评论系统，允许 **完全匿名的评论** 和简单的主题化，旨在替代 Disqus 等专有服务。


## 参考

- [关于静态博客的评论系统](https://www.cnblogs.com/nodecat/p/13058292.html)








