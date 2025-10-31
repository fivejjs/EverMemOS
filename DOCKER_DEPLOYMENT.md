# Memsys Docker 部署指南

这个文档说明如何使用 Docker Compose 部署 Memsys 记忆系统及其依赖服务。

## 服务架构

本 Docker Compose 配置包含以下服务：

- **memsys-app**: Memsys 主应用 (端口: 1995)
- **mongodb**: MongoDB 数据库 (端口: 27017)
- **elasticsearch**: Elasticsearch 搜索引擎 (端口: 9200)
- **milvus-standalone**: Milvus 向量数据库 (端口: 19530)
- **milvus-etcd**: Milvus 依赖的 etcd 服务
- **milvus-minio**: Milvus 依赖的 MinIO 对象存储 (端口: 9000, 9001)
- **redis**: Redis 缓存 (端口: 6379)
- **postgresql**: PostgreSQL 数据库，用于 LangGraph 检查点 (端口: 5432)

## 快速开始

### 1. 环境准备

确保您的系统已安装：
- Docker (版本 20.10+)
- Docker Compose (版本 2.0+)

### 2. 配置环境变量

复制环境配置文件并修改相关配置：

```bash
cp docker.env .env
```

编辑 `.env` 文件，配置您的 API 密钥：

```bash
# 必须配置的 API 密钥
CONV_MEMCELL_LLM_API_KEY=your-openai-api-key-here
EPISODE_MEMORY_LLM_API_KEY=your-openai-api-key-here

# 可选配置
DEEPINFRA_API_KEY=your-deepinfra-api-key-here
SILICONFLOW_API_KEY=your-siliconflow-api-key-here
```

### 3. 启动服务

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f memsys-app
```

### 4. 验证部署

等待所有服务启动完成后，访问以下端点验证：

- **Memsys 应用**: http://localhost:1995
- **Elasticsearch**: http://localhost:9200
- **Milvus MinIO 控制台**: http://localhost:9001 (用户名/密码: minioadmin/minioadmin)

## 详细配置

### 数据库配置

#### MongoDB
- 用户名: `admin`
- 密码: `memsys123`
- 数据库: `memsys`
- 端口: `27017`

#### PostgreSQL
- 用户名: `memsys`
- 密码: `memsys123`
- 数据库: `memsys`
- 端口: `5432`

#### Redis
- 端口: `6379`
- 数据库: `8`

### 搜索引擎配置

#### Elasticsearch
- 端口: `9200`
- 安全模式: 已禁用 (开发环境)
- 内存配置: 1GB

#### Milvus
- 端口: `19530`
- Web 端口: `9091`
- MinIO 控制台: `9001`

## 常用命令

### 服务管理

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 重启特定服务
docker-compose restart memsys-app
```

### 日志查看

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f memsys-app
docker-compose logs -f mongodb
docker-compose logs -f elasticsearch
```

### 数据管理

```bash
# 备份数据
docker-compose exec mongodb mongodump --out /backup
docker-compose exec postgresql pg_dump -U memsys memsys > backup.sql

# 清理数据 (谨慎使用)
docker-compose down -v
```

### 进入容器

```bash
# 进入应用容器
docker-compose exec memsys-app bash

# 进入数据库容器
docker-compose exec mongodb mongosh
docker-compose exec postgresql psql -U memsys -d memsys
```

## 故障排除

### 常见问题

1. **服务启动失败**
   ```bash
   # 检查服务状态
   docker-compose ps
   
   # 查看详细日志
   docker-compose logs memsys-app
   ```

2. **端口冲突**
   - 检查端口是否被占用
   - 修改 `docker-compose.yaml` 中的端口映射

3. **内存不足**
   - 调整 Elasticsearch 内存配置
   - 确保系统有足够内存 (建议 8GB+)

4. **API 密钥错误**
   - 检查 `.env` 文件中的 API 密钥配置
   - 确保 API 密钥有效且有足够额度

### 健康检查

所有服务都配置了健康检查，可以通过以下命令查看：

```bash
# 查看服务健康状态
docker-compose ps
```

绿色状态表示服务正常运行。

## 生产环境部署

### 安全配置

1. **修改默认密码**
   - 修改 MongoDB、PostgreSQL、Redis 的默认密码
   - 启用 Elasticsearch 安全模式

2. **网络配置**
   - 使用 Docker 网络隔离
   - 配置防火墙规则

3. **数据持久化**
   - 确保数据卷正确挂载
   - 定期备份数据

### 性能优化

1. **资源限制**
   ```yaml
   services:
     memsys-app:
       deploy:
         resources:
           limits:
             memory: 2G
             cpus: '1.0'
   ```

2. **日志管理**
   - 配置日志轮转
   - 设置日志级别

## 监控和维护

### 监控指标

- 服务健康状态
- 资源使用情况
- 应用性能指标

### 定期维护

- 清理日志文件
- 更新镜像版本
- 备份重要数据

## 支持

如果遇到问题，请：

1. 查看服务日志
2. 检查配置文件
3. 参考项目文档
4. 提交 Issue

---

**注意**: 这是一个开发环境的配置，生产环境部署时请根据实际需求调整安全配置和性能参数。
