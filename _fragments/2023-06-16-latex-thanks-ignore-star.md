---
layout: fragment
title: 去除 LaTex \thanks 脚注的 * 号
tags: [LaTex]
excerpt: 去除 LaTex \thanks 脚注的 * 号
keywords: LaTex
mathjax: true
---

在 `\begin{document}` 之前添加：

```tex
\makeatletter
\def\thanks#1{\protected@xdef\@thanks{\@thanks
        \protect\footnotetext{#1}}}
\makeatother
```


## 参考

- LaTex Stack Exchange：[Remove thanks star from author and affiliation](https://tex.stackexchange.com/questions/428237/remove-thanks-star-from-author-and-affiliation)

- CSDN：[怎么去除latex \thanks 感谢脚注的*号](https://blog.csdn.net/qq_43039472/article/details/114384907)

