---
layout: fragment
title: OSError——No space left on device
tags: [PyTorch]
description: some word here
keywords: PyTorch
---

使用 `nohup` 后台运行 python 程序：

```shell
CUDA_VISIBLE_DEVICES=2 nohup python -u train.py > train.log 2>&1 &
```

在运行了两个 epoch 之后，报下面的错误：

```shell
OSError: [Errno 28] No space left on device: '/tmp/pymp-jdj0i3mt'
```

大致意思就是说系统的空间不足。

使用以下命令查看磁盘的存储情况：

```shell
df -h
```

```shell
$ df -h
Filesystem                      Size  Used Avail Use% Mounted on
devtmpfs                        189G     0  189G   0% /dev
tmpfs                           189G  843M  188G   1% /dev/shm
tmpfs                           189G  1.5G  187G   1% /run
tmpfs                           189G     0  189G   0% /sys/fs/cgroup
/dev/mapper/thin--volumes-root   50G   47G     0 100% /
/dev/sda2                       488M  338M  115M  75% /boot
/dev/mapper/thin--volumes-data  5.6T  2.9T  2.5T  55% /kolla
tmpfs                            38G     0   38G   0% /run/user/1000
```

可以看到其中的 `/dev/mapper/thin--volumes-root` 已经占用 100% 了，所以提示空间不足。

进入 `/tmp` 目录，使用 `du -h -x --max-depth=1`，发现有一个文件占了 2.2 G，进一步了解到是上述 python 运行时留下的缓存文件，将其进行删除。

---

更进一步，由于在使用 `nohup` 命令时没有启用 python 缓存（即 `-u`），导致了根目录 `/` 空间爆满。

因此，保险起见，在根目录空间不足的情况下，还是 **要开启 python 缓存**（虽然存储程序输出的文件不能实时刷新）：

```shell
CUDA_VISIBLE_DEVICES=2 nohup python train.py > train.log 2>&1 &
```



## 参考

- CSDN：[OSError: [Errno 28] No space left on device以及查看系统分区情况](https://blog.csdn.net/qq_51570094/article/details/124582940)

- CSDN：[Python后台运行 -- nohup python xxx.py &](https://blog.csdn.net/m0_38024592/article/details/103336210)
