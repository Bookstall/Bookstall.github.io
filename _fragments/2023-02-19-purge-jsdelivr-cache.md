---
layout: fragment
title: 清除 JsDelive 缓存
tags: [前端, CDN]
description: some word here
keywords: 前端, CDN
---

JsDelivr 提供的全球 CDN 加速，CDN 的分流作用不仅减少了用户的访问延时，也减少的源站的负载，并且 JsDelivr 是开源的免费 CDN。

但是，当网站更新时，如果 CDN 节点上数据没有及时更新，即便用户在浏览器使用 `Ctrl` + `F5`（windows）或者  `command` + `shift` + `R`（MAC）的 **强制刷新** 方式使 **浏览器端的缓存** 失效，也会因为 **CDN 边缘节点没有同步最新数据而导致用户端未能及时更新**。

也就是说：尽管你已经将更改之后的静态资源（如 js、css 等）推送到 github，仍然会出现 **文件最新版本和 CDN 缓存版本不同步** 的问题，导致无法在第一时间更新。

> - github branch 更新后，JsDelivr 会在手动清理缓存的 **24 小时** 内刷新
> 
> - github release 更新后，JsDelivr 会在手动清理缓存的 **7 天** 内刷新



## 清除 JsDelive 缓存

对于 JsDelivr，缓存刷新的方式也很简单，只需将想刷新的链接的开头的 `cdn` 更改为 `purge`，即将

`https://cdn.jsdelivr.net/` 

切换为 

`https://purge.jsdelivr.net/`。


示例接口：https://purge.jsdelivr.net/gh/Bookstall/Bookstall.github.io@main/assets/vendor/prism/js/prism.js

直接在浏览器访问该实验接口，可以得到如下的结果：

```json
{
  "id": "EHTzLz0bJ4hOM0K3",
  "status": "finished",
  "timestamp": "2023-02-19T06:19:52.546Z",
  "paths": {
    "/gh/Bookstall/Bookstall.github.io@main/assets/vendor/prism/js/prism.js": {
      "throttled": false,
      "providers": {
        "CF": true,
        "FY": true,
        "GC": true
      }
    }
  }
}
```


## 其他工具

- https://github.com/rockswang/ghdelivr

- [GitHub action for the jsDelivr cache purging](https://github.com/gacts/purge-jsdelivr-cache)


## 参考

- [JsDelivr 文档：Purge cache](https://www.jsdelivr.com/documentation#id-purge-cache)

- [jsdelivr CDN 使用和缓存刷新 _](https://www.cnblogs.com/UncleZhao/p/13753723.html)

- [Jsdelivr CDN 缓存清除工具 |缓存刷新|缓存更新|免费CDN](https://www.tgee.cn/jsdelivr-cdn.html)

- [一个帮你实时刷新jsdelivr CDN缓存的小工具](https://segmentfault.com/a/1190000025179613)

