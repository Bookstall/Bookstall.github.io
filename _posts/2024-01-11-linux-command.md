---
layout: post
author: bookstall
tags: Linux
categories: [Linux]
excerpt: 记录了常用的一些 Linux 命令，包括 chmod、ps、df 等。
keywords: Linux
title: Linux 常用命令
mathjax: true
---

## Linux 常用命令

文件和目录操作：

- `ls`：列出当前目录中的文件和子目录

- `cd`：切换工作目录

- `pwd`：显示当前工作目录的路径

- `find`：

- `mkdir`：创建新目录

- `rm`

- `vi` / `vim`

- `cat`

文件权限管理：

- `chmod`

网络相关命令：

- `ifconifg`

- `netstat`

系统资源监控命令：

- `top`

- `df`

`df` 以磁盘分区为单位查看文件系统，可以获取硬盘被占用了多少空间，目前还剩下多少空间等信息：

```shell
$ df -h 
```

![df -h 结果示意图](https://img-blog.csdnimg.cn/20210111134855846.png)

- `du`：查看当前目录占用的内存（disk usage）

```shell
#查看当前目录大小
$ du -sh
#返回该目录/文件的大小
$ du -sh [目录/文件]


#查看当前文件夹下的所有文件大小（包含子文件夹）
$ du -h
#查看指定文件夹下的所有文件大小（包含子文件夹）
$ du -h [目录/文件]

#返回当前文件夹的总M数
$ du -sm
#返回指定文件夹/文件总M数
$ du -sm [文件夹/文件]
```

- `free`：内存使用

- `at`：定时任务

进程与端口管理：

- `ps`

```shell
$ ps -aux | grep tensorboard
```

- `kill`

```shell
kill -9 进程号
```

- `nohup`：不挂起（No Haug Up）

```shell
nohuo python -m test.py > out.log 2&1 &
```

压缩与解压缩：

- `tar`

- `unzip`

线上查询及帮助命令：

- `man`

- `help`



## 参考

- CSDN
  
  - [Linux 查看磁盘空间命令（df、du）](https://blog.csdn.net/jadeandplum/article/details/112466387)

- [Linux 命令完全手册](https://www.freecodecamp.org/chinese/news/the-linux-commands-handbook/)
