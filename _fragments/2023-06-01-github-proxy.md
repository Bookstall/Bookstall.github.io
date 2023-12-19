---
layout: fragment
title: GitHub 文件加速及代理
tags: [github]
excerpt: 
keywords: github
mathjax: true
---


## GitHub 文件加速及代理



### 常用的代理

#### 1）Github Proxy

> GitHub 文件、Releases 、archive、gist、raw.githubusercontent.com 文件代理加速下载服务

- 支持终端命令行 `git clone`、`wget`、`curl` 等工具下载

- 支持 raw.githubusercontent.com、gist.github.com、gist.githubusercontent.com 文件下载

- 不支持 SSH Key 方式 git clone 下载

使用示例如下：

```shell
# git clone
$ git clone https://ghproxy.com/https://github.com/stilleshan/ServerStatus


# git clone 私有仓库
# Clone 私有仓库需要用户在 Personal access tokens 申请 Token 配合使用.
$ git clone https://user:your_token@ghproxy.com/https://github.com/your_name/your_private_repo


# wget & curl
$ wget https://ghproxy.com/https://github.com/stilleshan/ServerStatus/archive/master.zip
$ wget https://ghproxy.com/https://raw.githubusercontent.com/stilleshan/ServerStatus/master/Dockerfile
$ curl -O https://ghproxy.com/https://github.com/stilleshan/ServerStatus/archive/master.zip
$ curl -O https://ghproxy.com/https://raw.githubusercontent.com/stilleshan/ServerStatus/master/Dockerfile
```

#### 2）GitHub 文件加速

GitHub 文件链接带不带协议头都可以，支持 release、archive 以及文件，右键复制出来的链接都是符合标准的

但是，**不支持项目文件夹**。

使用示例如下所示：

- 分支源码：
  
  `https://github.moeyy.xyz/https://github.com/moeyy/project/archive/master.zip`

- release源码：
  
  `https://github.moeyy.xyz/https://github.com/moeyy/project/archive/v0.1.0.tar.gz`

- release文件：

  `https://github.moeyy.xyz/https://github.com/moeyy/project/releases/download/v0.1.0/example.zip`

- 分支文件：
  
  `https://github.moeyy.xyz/https://github.com/moeyy/project/blob/master/filename`

- Raw：
  
  `https://github.moeyy.xyz/https://raw.githubusercontent.com/moeyy/project/archive/master.zip`

- 使用 Git: 
  
  `git clone https://github.moeyy.xyz/https://github.com/moeyy/project`



### 其他

其他方式可以参考 [GitHub代下载[文件加速]网站及反代列表](https://cjh0613.com/githubproxy)，暂时没有进行尝试。


## 参考

- Github Proxy
  
  - [网址](https://ghproxy.com/)
  
  - [Github Repo](https://github.com/hunshcn/gh-proxy)

- 博客：[GitHub代下载[文件加速]网站及反代列表](https://cjh0613.com/githubproxy)

