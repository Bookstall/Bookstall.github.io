---
layout: fragment
title: 使用 JavaScript 对 URL 进行编解码
tags: [前端]
description: some word here
keywords: 前端
---

## URL 编解码

### 防止歧义

对于 URL 来说，之所以要进行编码，是因为 URL 中有些字符会引起 **歧义**。

例如，URL 参数字符串中使用 `key=value`（键值对）的形式来传参，键值对之间以 `&` 符号分隔（如 `/s?q=abc&ie=utf-8`）。

如果你的 `value` 字符串中包含了`=` 或者 `&`，那么势必会造成接收 URL 的服务器解析错误，因此必须将引起歧义的 `&` 和 `=` 符号进行 **转义**，也就是 **对其进行（重新）编码**。

> URL 的编码格式采用的是 **ASCII 码**，而不是 Unicode，这也就是说你不能在 URL 中包含任何非 ASCII 字符，例如 **中文**。
> 
> 否则如果客户端浏览器和服务端浏览器支持的字符集不同的情况下，<u>中文可能会造成问题（大多数时候被解析成 `%xx%xx` 的形式）</u>。



### URL 的编码原则

URL编码的原则就是使用 **安全的字符**（没有特殊用途或者特殊意义的可打印字符）**去表示那些不安全的字符**。

- 对于 URL 中的合法字符，编码和不编码是等价的。

- 对于不安全字符，如果不经过编码，那么它们有可能会造成 URL 语义的不同。

- 对于 URL 而言，只有普通英文字符和数字，特殊字符 `$-_.+!*'()` 还有保留字符，才能出现在未经编码的 URL 之中，**其他字符均需要经过编码之后才能出现在 URL 中**。

### RFC3986 协议的规定

RFC3986 协议对 URL 的编解码问题做出了详细的建议，指出了哪些字符需要被编码才不会引起 URL 语义的转变，以及对为什么这些字符需要编码做出了相应的解释。

RFC3986 协议规定 URL 只允许包含以下四种字符：

- 英文字母（a-zA-Z）

- 数字（0-9）

- `-`、`_`、`.`、`~` 4个特殊字符

- 所有保留字符，RFC3986 中指定了以下字符为保留字符（英文字符）：`!`、`*`、`'`、`(`、`)`、`;`、`:`、`@`、`&`、`=`、`+`、`$`、`,`、`/`、`?`、`#`、`[`、`]`

### 对 URL 中的非法字符进行编码

**URL 编码** 通常也被称为 **百分号编码**（Url Encoding，also known as percent-encoding）。

它的编码方式非常简单，使用 **`%` 百分号加上两位的字符**（0123456789ABCDEF）代表一个字节的 **十六进制** 形式。

URL 编码默认使用的字符集是 US-ASCII。例如：

- `a` 在 US-ASCII 码中对应的字节是 `0x61`，那么 URL 编码之后得到的就是 `%61`

- `@` 符号在 ASCII 字符集中对应的字节为 `0x40`，经过 URL 编码之后得到的是 `%40`

- 在浏览器地址栏上输入 `http://g.cn/search?q=%61%62%63`，实际上就等同于在 google 上搜索 `abc`

具体的编码规则如下所示：

- 对于非 ASCII 字符，需要使用 ASCII 字符集的超集进行编码得到相应的字节，然后对每个字节执行百分号编码；

- 对于 Unicode 字符，RFC 文档建议使用 UTF-8 对其进行编码得到相应的字节，然后对每个字节执行百分号编码；

    - 例如，对于 `中文` 一词，使用 UTF-8 字符集得到的字节为 `0xE4 0xB8 0xAD 0xE6 0x96 0x87`，经过 URL 编码之后得到 `%E4%B8%AD%E6%96%87`

- 如果某个字节对应着 ASCII 字符集中的某个非保留字符，则此字节 **无需** 使用百分号表示；



## 使用 JavaScript 进行 URL 编解码

在 JavaScript 中，共有三种方式对 URL 进行编解码：

| 方法                         | 说明                                                                                                                                        | 返回值                  |
| :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------- |
| escape(String)               | 使用转义序列替换某些字符来对字符串进行编码，除了 ASCII 字母、数字、标点符号 `@ * _ + - . /` 以外                                            | 返回 Unicode 编码字符串 |
| unescape(String)             | 对使用 `escape()` 编码的字符串进行解码                                                                                                      |                         |
| encodeURI(String)            | 通过转义某些字符对 URI 进行编码，除了常见的符号以外（ASCII 字符），对其他一些在网址中有特殊含义的符号 `; / ? : @ & = + $ , #`，也不进行编码 | 输出 UTF-8 形式字符串   |
| decodeURI(String)            | 对使用 `encodeURI()` 方法编码的字符串进行解码                                                                                               |                         |
| encodeURIComponent(String)   | 通过某些转义字符对 URI 进行编码，会编译所有（包含特殊字符），ASCII 字符不编码，可以将参数中的 <u>中文、特殊字符</u> 进行转义                | 输出 UTF-8 形式字符串   |
| deencodeURIComponent(String) | 对使用 `encodeURIComponent()` 方法编码的字符串进行解码                                                                                      |                         |

- encodeURI 方法不会对 ASCII 字母、数字、~!@#$&*()=:/,;?+' 编码；

- encodeURIComponent 方法不会对 ASCII 字母、数字、~!*()' 编码；

- encodeURIComponent 比 encodeURI 编码的范围大。因此当你需要编码整个 URL，就用 encodeURI；如果只需要编码 URL 中的参数时，就使用 encodeURIComponent；


### escape 和 unescape

`escape()` **不能直接用于 URL 编码**，它的真正作用是 **返回一个字符的 Unicode 编码值**。

它的具体规则是：除了 ASCII 字母、数字、标点符号 `@ * _ + - . /` 以外，对其他所有字符进行编码。在 `u0000` 到 `u00ff` 之间的符号被转成 `%xx` 的形式，其余符号被转成 `%uxxxx` 的形式。

其对应的解码函数是 `unescape()`。

还需要注意：

- 无论网页的原始编码是什么，一旦被 Javascript 编码，就都变为 Unicode 字符。也就是说，Javascipt 函数的输入和输出，默认都是 Unicode 字符。

- `escape()` 不对 `+` 编码。但是我们知道，网页在提交表单的时候，如果有空格，则会被转化为 `+` 字符。服务器处理数据的时候，会把 `+` 号处理成空格。所以，使用的时候要小心。


```javascript
const time = 2022-01-09
const tile = '63元黑糖颗粒固饮'

// escape 编码
let url = "http://localhost:8080/index.html?time="+escape(time)+"&title="+escape(tile)
// 结果：http://localhost:8080/index.html?time=2022-01-09&title=63%u5143%u9ED1%u7CD6%u9897%u7C92%u56FA%u996E


// unescape 解码
let url = "http://localhost:8080/index.html?time="+unescape(2022-01-09)+"&title="+unescape(63%u5143%u9ED1%u7CD6%u9897%u7C92%u56FA%u996E)
// 结果：http://localhost:8080/index.html?time=2022-01-09&title=63元黑糖颗粒固饮
```



### encodeURI 和 decodeURI

`encodeURI()` 是 Javascript 中真正用来对 URL 编码的函数。

它用于对 URL 的组成部分进行个别编码，除了常见的符号以外，对其他一些在网址中有特殊含义的符号 `; / ? : @ & = + $ , #`，也不进行编码。编码后，它输出符号的 UTF-8 形式，并且在每个字节前加上 `%`，，然后用十六进制的转义序列（形式为 `%xx`）对生成的 1 字节、2 字节或 4 字节的字符进行编码。

它对应的解码函数是 `decodeURI()`

```javascript
let url = "http://localhost:8080/index.html?time=2022-01-09&title=63元黑糖颗粒固饮"

// encodeURI 编码
let encodeURI_url = encodeURI(url)
// "http://localhost:8080/index.html?time=2022-01-09&title=63%E5%85%83%E9%BB%91%E7%B3%96%E9%A2%97%E7%B2%92%E5%9B%BA%E9%A5%AE"

// decodeURI 解码
decodeURI(encodeURI_url) = "http://localhost:8080/index.html?time=2022-01-09&title=63元黑糖颗粒固饮"
```



### encodeURIComponent 和 decodeURIComponent

`encodeURIComponent()` 与 `encodeURI()` 的区别是，它用于对整个 URL 进行编码。

`; / ? : @ & = + $ , #` 这些在 `encodeURI()` 中不被编码的符号，在 `encodeURIComponent()` 中统统会被编码。

它对应的解码函数是 `decodeURIComponent()`

```javascript
let url = "http://localhost:8080/index.html?time=2022-01-09&title=63元黑糖颗粒固饮"

// encodeURIComponent 编码
let encodeURIComponent_url = encodeURIComponent(url)
// http%3A%2F%2Flocalhost%3A8080%2Findex.html%3Ftime%3D2022-01-09%26title%3D63%E5%85%83%E9%BB%91%E7%B3%96%E9%A2%97%E7%B2%92%E5%9B%BA%E9%A5%AE

// decodeURIComponent 解码
decodeURIComponent(encodeURIComponent_url) = "http://localhost:8080/index.html?time=2022-01-09&title=63元黑糖颗粒固饮"
```



## 参考

- [js对url进行编码解码的三种方式总结](http://www.codebaoku.com/it-js/it-js-275272.html)

- CSDN：[RFC3986之URL编码与解码、AFPercentEscapedStringFromString](https://blog.csdn.net/lyz0925/article/details/106230095)

- [RFC3986 协议](https://www.rfc-editor.org/rfc/rfc3986)
