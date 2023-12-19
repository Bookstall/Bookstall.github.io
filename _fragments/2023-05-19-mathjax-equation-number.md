---
layout: fragment
title: MathJax V2：自动添加公式编号
tags: [MathJax]
excerpt: MathJax V2：自动添加公式编号
keywords: MathJax
mathjax: true
---

## MathJax V2：自动添加公式编号

MathJax v2.0 的新功能是 **能够自动对方程进行编号**。**默认情况下，此功能处于关闭状态**，因此当您从 v1.1 更新到 v2.0 时，页面不会更改，但可以通过添加以下内容轻松配置 MathJax 以生成自动方程编号：

```javascript
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>
```

这段文本需要在引入 `MathJax.js` 文件之前，即：

```javascript
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        TeX: {equationNumbers: {autoNumber: "AMS"}},
    });
</script>

<script type="text/javascript"
    src="https://cdn.jsdelivr.net/gh/mathjax/MathJax@2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
```

并且只有 `\begin{equation}` 和 `\end{equation}` 的公式会被自动编号。

MathJax 非常厉害的一个东西就是 **交叉引用**。在 `\tag{}` 定义编号后面使用 `\label{}` 定义锚点，后面正文中 `\eqref{}` 或者`\ref{}` 就可以引用。区别是前者带括号，后者不带括号。



## 参考

- 文档：[Automatic Equation Numbering](https://docs.mathjax.org/en/v2.7-latest/tex.html#automatic-equation-numbering)

- CSDN：[MathJax 与 Katex 在公式对齐、编号、交叉引用方面的不同](https://blog.csdn.net/weixin_40301746/article/details/123967807)


