---
layout: post
author: bookstall
tags: Blog, Jekyll
categories: [Blog]
description: Add Image Gallery In Blog
keywords: Image Gallery, Blog, Jekyll
title: 在 Jekyll 博客中添加图片查看器
---

在查看博客时，我们有时需要对图片进行放大、查看，而默认的 Jekyll 博客并不具备此功能，因此，本文将介绍两种开源的 **图片查看工具**：**Viewer-JS** 和 **FancyBox（Fancyapps UI）**。

在对这两种工具进行使用之后，最终选择了 FancyBox 这一工具！

## Viewer-JS

> **项目地址：**
> 
> - [viewerjs](https://github.com/fengyuanchen/viewerjs)
> 
> - [jquery-viewer](https://github.com/fengyuanchen/jquery-viewer)



## FancyBox

> 官方地址：https://fancyapps.com/
> 
> FancyBox 文档：https://fancyapps.com/docs/ui/fancybox
> 
> 项目地址：https://github.com/fancyapps/ui

**Fancyapps UI** 是一个 **可复用的 JavaScript UI 组件（Reusable JavaScript UI
Component Library）**，包括三个子组件：

- FancyBOx

- Carousel

- PanZoom

![](https://fancyapps.com/img/logo.svg)



> **注意：**
> 
> 以下内容所使用的是 **`4.0.31` 版本**

### 快速使用


### 配置

#### Image


```javascript
Image: {
  click: "close",
  wheel: "slide",
  zoom: false,
  fit: "cover",
}
```

> 代码参考：https://fancyapps.com/playground/vi

#### Toolbar

**Toolbar 工具栏** 包含多种不同的工具，包括：

- zoom：放大

- slideshow：幻灯片播放

- fullscreen：全屏显示

- thumbnails：缩略图

- close：关闭


默认情况下，这些工具都是显示的，可以使用 `Toolbar: false` 来隐藏工具栏（取消所有工具的显示），也可以使用 `Toolbar: true` 来显示工具栏。

更高级的做法是对某些工具进行显示与隐藏，如下所示：

```javascript
Toolbar: {
  display: [
    {
    id: "counter",
    position: "center",
    },
    "zoom", // 表示显示 zoom 工具
    "slideshow",
    "fullscreen",
    "thumbs",
    "close",
  ]
}
```

> 代码参考：https://fancyapps.com/playground/16W


### Thumbs

用于控制缩略图的位置，

```javascript
Thumbs: {
  Carousel: {
    fill: false, // 
    center: true, // 缩略图居中显示
  }
}
```

> 代码参考：https://fancyapps.com/playground/17g


### 更多实例

更多关于 FancyBox 和 Fancyapps UI 的示例可以参见 [官方示例-Showcase](https://fancyapps.com/showcase/)，并且包括了详细的 JS 代码。





## 参考

- Viewer-JS：
  
  - [viewerjs](https://github.com/fengyuanchen/viewerjs)
  
  - [jquery-viewer](https://github.com/fengyuanchen/jquery-viewer)

- FancyBox：
  
  - [文档](https://fancyapps.com/docs/ui/fancybox)
  
  - [Github](https://github.com/fancyapps/ui)

- 博客：[Hexo 折腾：利用 Fancybox 添加图片放大预览查看功能](https://tianma8023.github.io/post/hexo-material-intergrate-image-display-feature/)

- 博客园：[Jekyll添加FancyBox 插件](https://www.cnblogs.com/Grand-Jon/p/7397652.html)