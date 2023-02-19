---
layout: fragment
title: Python 中变量名后面加『冒号』
tags: [Python]
description: some word here
keywords: Python
---

## Python 中变量名后面加『冒号』

在 [AutoCut：基于 Whisper 的视频剪辑工具]({{ site.url }}/2023/02/06/whisper-and-autocut/) 这篇博文中，对 AutoCut 核心代码进行了简要的描述，其中出现了这样的代码语句：

```python
final_clip: editor.VideoClip = editor.concatenate_videoclips(clips)

final_clip: editor.AudioClip = editor.concatenate_audioclips(clips)
```

这种语法是 **变量注释（Variable Annotations）**，在 **Python 3.6** 才推出。其作用是：注释变量类型,明确指出变量类型，方便帮助复杂案例中的 **类型推断**。

具体的格式如下：

```python
var: type = value
```

其实，本质上就是：

```python
# type 就是 var 期望的类型
var = value
```

但是，**变量注释只是一种提示，并非强制的，Python 解释器不会去校验 `value` 的类型是否真的是 `type`。**

例如：`a: str = 10` 这样是没有错的，Python 解释器在执行时会把 `a` 当作 `int` 来操作（即 `type(a)` 是 `int`）。这不像是 C 语言中 `int a`（`a` 必须是 `int` 类型）。

更详细的信息可以查看 Python 的官方文档：[PEP 526 – Syntax for Variable Annotations](https://peps.python.org/pep-0526/)

## 参考

- CSDN：[Python中变量名后面加冒号是什么意思?](https://blog.csdn.net/qq_43439853/article/details/91491303)

- 百度知道：[Python中变量名后面加冒号是什么意思?](https://zhidao.baidu.com/question/1930988314448559867.html)

- Python 官方文档：[PEP 526 – Syntax for Variable Annotations](https://peps.python.org/pep-0526/)