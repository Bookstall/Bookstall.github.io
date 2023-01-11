---
layout: post
author: bookstall
tags: Search, Blog
categories: [Search, Blog]
description: 
keywords: Search, Blog, Jekyll
title: Search In Static Blog
---


## Search In Static Blog

下面主要介绍两个开源的 jekyll 搜索项目：[Simple Jekyll Search](https://github.com/christian-fei/Simple-Jekyll-Search) 和 [Jekyll Tipue Search](https://github.com/jekylltools/jekyll-tipue-search)。

### Simple Jekyll Search

> 在该博客中，目前主要使用的是 Simple Jekyll Search 这一项目。

#### 引入 Simple Jekyll Search

与前端常见的引入 JS 代码的方式一样，我们需要引入 `simple-jekyll-search.js` 或者 `simple-jekyll-search.min.js`（推荐）这已 JS 文件。

这两个文件在 [Simple Jekyll Search 项目的 dest 目录](https://github.com/christian-fei/Simple-Jekyll-Search/tree/master/dest) 中。

```html
<script src="/simple-jekyll-search.min.js"></script>
```

#### 编写 search_data.json

我们还需要设置搜索的逻辑，以及返回的搜索结果的具体形式。因此，我们需要设置一个 `search_data.json` 文件，用于完成这些功能。

下面是 `search_data.json` 的一个例子：

```html
<!--为了防止被编译运行，这里添加了反斜杠，自己手动取掉反斜杠。
---
layout: null
---
\[
  \{\% for post in site.posts \%\}
    \{
      "title"    : "\{\{ post.title | escape \}\}",
      "category" : "\{\{ post.category \}\}",
      "tags"     : "\{\{ post.tags | join: ', ' \}\}",
      "url"      : "\{\{ site.baseurl \}\}\{\{ post.url \}\}",
      "date"     : "\{\{ post.date \}\}"
    \} \{\% unless forloop.last \%\},\{\% endunless \%\}
  \{\% endfor \%\}
\]
-->
```

#### 添加搜索的 HTML 元素

我们还需要编写一个 HTML 的输入框，以供用户输入要搜索的关键字，以及用于展示搜索结果的容器，代码如下所示：

```html
<!-- HTML elements for search -->
<input type="text" id="search-input" placeholder="搜索博客 - 输入标题/相关内容/日期/Tags.." style="width:380px;"/>

<ul id="results-container"></ul>
```


#### 编写搜索的主代码

```html
<script>
    SimpleJekyllSearch({
        searchInput: document.getElementById('search-input'),
        resultsContainer: document.getElementById('results-container'),
        json: '/search.json',
        searchResultTemplate: '<li><a href="{url}" title="{desc}">{title}</a></li>',
        noResultsText: '没有搜索到文章',
        limit: 20,
        fuzzy: false
    })
</script>
```

#### 效果演示

最终的演示效果如下图所示：

![](https://github.com/ZoharAndroid/MarkdownImages/blob/master/2019-08/%E6%95%88%E6%9E%9C2.png?raw=true)

> 图片来源：[Jekyll个人博客实现搜索功能](https://zoharandroid.github.io/2019-08-01-jekyll%E4%B8%AA%E4%BA%BA%E5%8D%9A%E5%AE%A2%E5%AE%9E%E7%8E%B0%E6%90%9C%E7%B4%A2%E5%8A%9F%E8%83%BD/)


### Jekyll Tipue Search

更加高级的搜索组件，可以显示搜索的结果、某次搜索所用的时间、关键词对应的文本内容。

#### 效果演示

演示地址：https://jekylltools.github.io/jekyll-tipue-search/search/



### 参考

- Simple-Jekyll-Search：https://github.com/christian-fei/Simple-Jekyll-Search
- Jekyll Tipue Search：https://github.com/jekylltools/jekyll-tipue-search
- 博客：[Jekyll个人博客实现搜索功能](https://zoharandroid.github.io/2019-08-01-jekyll%E4%B8%AA%E4%BA%BA%E5%8D%9A%E5%AE%A2%E5%AE%9E%E7%8E%B0%E6%90%9C%E7%B4%A2%E5%8A%9F%E8%83%BD/)


