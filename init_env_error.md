# Docker 环境初始化问题总结

## 问题描述
在 WSL2 环境中运行 `sudo docker-compose up flappy-bird-dqn` 时遇到 Docker 守护进程未启动的错误：

```
FileNotFoundError: [Errno 2] No such file or directory
docker.errors.DockerException: Error while fetching server API version
```

## 根本原因
- WSL2 环境中 Docker 守护进程默认未启动
- 与传统 Linux 系统不同，WSL2 不使用 systemd 作为初始化系统
- 需要手动启动 Docker 守护进程

## 解决方案

### 方案1: 手动启动 Docker 守护进程
```bash
# 启动 Docker 守护进程（后台运行）
sudo dockerd > /dev/null 2>&1 &

# 等待几秒让守护进程完全启动
sleep 3

# 验证 Docker 是否正常工作
docker ps
```

### 方案2: 使用 Docker Desktop（推荐）
1. 在 Windows 主机上安装并启动 Docker Desktop
2. 确保 Docker Desktop 设置中启用了 WSL2 集成
3. 在 WSL2 中直接使用 docker 命令

## 验证步骤
1. 检查 Docker 安装状态：
   ```bash
   docker --version
   which docker
   ```

2. 测试 Docker 守护进程：
   ```bash
   docker ps
   ```

3. 如果成功显示容器列表（即使为空），说明 Docker 已正常启动

## 注意事项
- WSL2 中不支持 `systemctl` 和 `service` 命令启动 Docker
- 每次重启 WSL2 后可能需要重新启动 Docker 守护进程
- 建议使用 Docker Desktop 以获得更稳定的体验

## 相关命令汇总
```bash
# 检查 Docker 状态
docker --version
docker ps

# 启动 Docker 守护进程
sudo dockerd > /dev/null 2>&1 &

# 运行项目
docker-compose up flappy-bird-dqn
```