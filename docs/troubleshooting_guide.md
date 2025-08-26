# Qwen3-4B-Thinking 模型服务故障排除指南

## 概述

本指南提供了 Qwen3-4B-Thinking 模型服务常见问题的诊断和解决方案，包括部署问题、运行时错误、性能问题和监控告警的处理方法。

## 快速诊断工具

### 自动诊断脚本

```bash
# 运行完整诊断
uv run python scripts/diagnose_issues.py

# 检查特定问题
uv run python scripts/diagnose_issues.py --check service
uv run python scripts/diagnose_issues.py --check gpu
uv run python scripts/diagnose_issues.py --check memory
```

### 手动检查清单

```bash
# 1. 服务状态检查
curl http://localhost:8000/health

# 2. 容器状态检查（Docker部署）
docker ps | grep qwen
docker logs qwen-thinking-service --tail=50

# 3. 系统资源检查
free -h
nvidia-smi
df -h

# 4. 端口检查
netstat -tlnp | grep :8000
```

## 部署问题

### 1. Docker 构建失败

**症状**:
- Docker build 命令失败
- 依赖安装错误
- 镜像构建超时

**诊断步骤**:
```bash
# 检查Docker版本
docker --version

# 检查磁盘空间
df -h

# 清理Docker缓存
docker system prune -f

# 查看详细构建日志
docker build -t qwen-thinking-service . --no-cache --progress=plain
```

**解决方案**:

1. **依赖安装失败**:
```bash
# 更新包索引
docker build --build-arg DEBIAN_FRONTEND=noninteractive -t qwen-thinking-service .

# 使用国内镜像源
docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t qwen-thinking-service .
```

2. **内存不足**:
```bash
# 增加Docker内存限制
# 在Docker Desktop中调整内存设置到8GB+

# 或使用多阶段构建减少内存使用
# 修改Dockerfile使用多阶段构建
```

3. **网络问题**:
```bash
# 使用代理构建
docker build --build-arg HTTP_PROXY=http://proxy:port -t qwen-thinking-service .

# 或使用离线安装包
# 预先下载依赖包到本地
```

### 2. 容器启动失败

**症状**:
- 容器启动后立即退出
- 健康检查失败
- 端口绑定错误

**诊断步骤**:
```bash
# 查看容器日志
docker logs qwen-thinking-service

# 检查容器状态
docker inspect qwen-thinking-service

# 进入容器调试
docker run -it --entrypoint /bin/bash qwen-thinking-service

# 检查端口占用
netstat -tlnp | grep :8000
```

**解决方案**:

1. **端口冲突**:
```bash
# 修改端口映射
docker run -p 8001:8000 qwen-thinking-service

# 或停止占用端口的进程
sudo lsof -ti:8000 | xargs sudo kill -9
```

2. **权限问题**:
```bash
# 检查文件权限
ls -la ./models/
chmod -R 755 ./models/

# 使用正确的用户ID
docker run --user $(id -u):$(id -g) qwen-thinking-service
```

3. **环境变量错误**:
```bash
# 检查环境变量
docker run --env-file .env qwen-thinking-service

# 验证环境变量
docker exec qwen-thinking-service env | grep MODEL
```

### 3. 模型加载失败

**症状**:
- 模型文件找不到
- 内存不足错误
- 格式不兼容错误

**诊断步骤**:
```bash
# 检查模型文件
ls -la ./models/qwen3-4b-thinking/
du -sh ./models/qwen3-4b-thinking/

# 检查模型文件完整性
python -c "
import torch
try:
    model = torch.load('./models/qwen3-4b-thinking/pytorch_model.bin', map_location='cpu')
    print('模型文件正常')
except Exception as e:
    print(f'模型文件错误: {e}')
"

# 检查可用内存
free -h
nvidia-smi
```

**解决方案**:

1. **模型文件缺失**:
```bash
# 重新下载模型
huggingface-cli download Qwen/Qwen3-4B-Thinking-2507 --local-dir ./models/qwen3-4b-thinking

# 或从备份恢复
cp -r /backup/models/qwen3-4b-thinking ./models/
```

2. **内存不足**:
```bash
# 启用CPU卸载
export ENABLE_CPU_OFFLOAD=true

# 使用更激进的量化
export QUANTIZATION_FORMAT=int4

# 减少批次大小
export BATCH_SIZE=1
```

3. **格式不兼容**:
```bash
# 转换模型格式
python scripts/convert_model_format.py --input ./models/qwen3-4b-thinking --output ./models/qwen3-4b-thinking-converted

# 或使用兼容的模型版本
```

## 运行时问题

### 1. 服务无响应

**症状**:
- API请求超时
- 健康检查失败
- 服务进程存在但不响应

**诊断步骤**:
```bash
# 检查进程状态
ps aux | grep uvicorn
ps aux | grep python

# 检查网络连接
netstat -tlnp | grep :8000
curl -v http://localhost:8000/health

# 检查系统负载
top
htop
```

**解决方案**:

1. **进程死锁**:
```bash
# 发送信号检查进程状态
kill -USR1 $(cat service.pid)

# 强制重启服务
kill -9 $(cat service.pid)
./scripts/start_service.sh
```

2. **资源耗尽**:
```bash
# 检查内存使用
free -h
# 如果内存不足，重启服务
sudo systemctl restart qwen-thinking-service

# 检查文件描述符
ulimit -n
# 增加文件描述符限制
ulimit -n 65536
```

3. **网络问题**:
```bash
# 检查防火墙
sudo ufw status
sudo iptables -L

# 重新绑定端口
sudo netstat -tlnp | grep :8000
sudo lsof -ti:8000 | xargs sudo kill -9
```

### 2. 内存溢出 (OOM)

**症状**:
- 进程被系统杀死
- CUDA out of memory 错误
- 系统变慢或无响应

**诊断步骤**:
```bash
# 检查系统日志
dmesg | grep -i "killed process"
journalctl -u qwen-thinking-service | grep -i oom

# 检查内存使用历史
sar -r 1 10

# 检查GPU内存
nvidia-smi
watch -n 1 nvidia-smi
```

**解决方案**:

1. **系统内存不足**:
```bash
# 增加交换空间
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 调整内存参数
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

2. **GPU内存不足**:
```bash
# 启用梯度检查点
export ENABLE_GRADIENT_CHECKPOINTING=true

# 减少批次大小
export BATCH_SIZE=1
export PER_DEVICE_BATCH_SIZE=1

# 启用CPU卸载
export ENABLE_CPU_OFFLOAD=true
```

3. **内存泄漏**:
```bash
# 重启服务释放内存
docker restart qwen-thinking-service

# 监控内存使用
python scripts/monitor_memory.py

# 启用内存分析
export PYTHONMALLOC=debug
```

### 3. 推理速度慢

**症状**:
- 响应时间超过预期
- GPU利用率低
- 队列积压

**诊断步骤**:
```bash
# 检查GPU利用率
nvidia-smi dmon -s pucvmet -d 1

# 检查CPU使用率
top -p $(pgrep -f uvicorn)

# 分析请求性能
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# 检查网络延迟
ping localhost
```

**解决方案**:

1. **GPU未充分利用**:
```bash
# 检查CUDA版本兼容性
python -c "import torch; print(torch.version.cuda)"
nvidia-smi

# 启用混合精度
export ENABLE_MIXED_PRECISION=true

# 优化批处理
export ENABLE_DYNAMIC_BATCHING=true
export MAX_BATCH_SIZE=8
```

2. **模型量化问题**:
```bash
# 尝试不同量化格式
export QUANTIZATION_FORMAT=int8  # 或 int4, gptq

# 禁用量化测试性能
export QUANTIZATION_FORMAT=none
```

3. **系统瓶颈**:
```bash
# 检查磁盘I/O
iostat -x 1 10

# 优化系统参数
echo 'net.core.somaxconn = 65535' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## 网络和API问题

### 1. API请求失败

**症状**:
- HTTP 500 错误
- 连接被拒绝
- 请求超时

**诊断步骤**:
```bash
# 测试API端点
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "max_length": 10}' \
  -v

# 检查API文档
curl http://localhost:8000/docs

# 查看详细错误日志
docker logs qwen-thinking-service --tail=100
tail -f logs/service.log
```

**解决方案**:

1. **请求格式错误**:
```bash
# 验证JSON格式
echo '{"prompt": "test"}' | jq .

# 检查必需字段
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "max_length": 100, "temperature": 0.7}'
```

2. **认证问题**:
```bash
# 如果启用了API密钥
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/generate

# 检查CORS设置
curl -H "Origin: http://localhost:3000" http://localhost:8000/health
```

3. **服务器错误**:
```bash
# 检查服务器日志
grep -i error logs/service.log
grep -i exception logs/service.log

# 重启服务
docker restart qwen-thinking-service
```

### 2. 并发问题

**症状**:
- 高并发时响应慢
- 连接被拒绝
- 服务不稳定

**诊断步骤**:
```bash
# 并发测试
ab -n 100 -c 10 http://localhost:8000/health

# 检查连接数
netstat -an | grep :8000 | wc -l

# 监控系统负载
vmstat 1 10
```

**解决方案**:

1. **增加工作进程**:
```bash
# 修改worker数量
export MAX_WORKERS=4

# 或在docker-compose.yml中修改
environment:
  - MAX_WORKERS=4
```

2. **优化连接池**:
```bash
# 调整系统参数
echo 'net.core.somaxconn = 65535' | sudo tee -a /etc/sysctl.conf
echo 'fs.file-max = 65535' | sudo tee -a /etc/sysctl.conf

# 增加文件描述符限制
ulimit -n 65535
```

3. **负载均衡**:
```bash
# 使用nginx负载均衡
# 配置多个服务实例
docker-compose up --scale qwen-model-service=3
```

## 监控和告警问题

### 1. 监控数据缺失

**症状**:
- Prometheus无数据
- Grafana图表空白
- 告警不触发

**诊断步骤**:
```bash
# 检查Prometheus目标
curl http://localhost:9090/api/v1/targets

# 检查指标端点
curl http://localhost:8000/metrics

# 验证Grafana数据源
curl http://localhost:3000/api/datasources
```

**解决方案**:

1. **指标端点问题**:
```bash
# 启用指标收集
export ENABLE_METRICS=true

# 检查指标格式
curl http://localhost:8000/metrics | head -20
```

2. **网络连接问题**:
```bash
# 检查容器网络
docker network ls
docker network inspect qwen-network

# 测试容器间连接
docker exec prometheus ping qwen-model-service
```

3. **配置错误**:
```bash
# 验证Prometheus配置
docker exec prometheus promtool check config /etc/prometheus/prometheus.yml

# 重新加载配置
curl -X POST http://localhost:9090/-/reload
```

### 2. 告警误报

**症状**:
- 频繁的假阳性告警
- 告警阈值不合理
- 告警通知失败

**诊断步骤**:
```bash
# 检查告警规则
curl http://localhost:9090/api/v1/rules

# 查看告警历史
curl http://localhost:9090/api/v1/alerts

# 测试告警通知
curl -X POST http://localhost:9093/api/v1/alerts
```

**解决方案**:

1. **调整告警阈值**:
```yaml
# 修改 monitoring/alerts.yml
- alert: HighMemoryUsage
  expr: memory_usage_percent > 85  # 从90调整到85
  for: 10m  # 增加持续时间
```

2. **优化告警规则**:
```yaml
# 添加标签过滤
- alert: ServiceDown
  expr: up{job="qwen-service", instance!~"test.*"} == 0
```

3. **配置通知渠道**:
```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
```

## 数据和模型问题

### 1. 模型输出质量差

**症状**:
- 生成文本不相关
- 中文处理错误
- 专业术语错误

**诊断步骤**:
```bash
# 测试基础功能
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "什么是AES加密？", "max_length": 100}'

# 检查模型版本
curl http://localhost:8000/model_info

# 验证tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./models/qwen3-4b-thinking')
print(tokenizer.encode('测试中文'))
"
```

**解决方案**:

1. **模型版本问题**:
```bash
# 验证模型完整性
python scripts/verify_model.py --model-path ./models/qwen3-4b-thinking

# 重新下载模型
rm -rf ./models/qwen3-4b-thinking
huggingface-cli download Qwen/Qwen3-4B-Thinking-2507 --local-dir ./models/qwen3-4b-thinking
```

2. **量化影响质量**:
```bash
# 尝试不同量化设置
export QUANTIZATION_FORMAT=int8  # 或禁用量化

# 调整推理参数
export DEFAULT_TEMPERATURE=0.7
export DEFAULT_TOP_P=0.9
```

3. **中文处理问题**:
```bash
# 检查中文tokenizer
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./models/qwen3-4b-thinking')
text = '中文测试'
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
print(f'原文: {text}')
print(f'解码: {decoded}')
"
```

### 2. 推理结果不一致

**症状**:
- 相同输入产生不同输出
- 随机性过高
- 结果不可复现

**诊断步骤**:
```bash
# 多次测试相同输入
for i in {1..5}; do
  curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "测试", "temperature": 0.1, "do_sample": false}'
done

# 检查随机种子设置
grep -r "seed" src/
```

**解决方案**:

1. **设置固定种子**:
```python
# 在模型服务中添加
import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

2. **调整采样参数**:
```bash
# 降低随机性
export DEFAULT_TEMPERATURE=0.1
export DEFAULT_TOP_P=0.95
export DEFAULT_DO_SAMPLE=false
```

3. **启用确定性模式**:
```python
# 在服务启动时设置
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 性能优化

### 1. 内存优化

```bash
# 启用内存优化选项
export ENABLE_MEMORY_OPTIMIZATION=true
export ENABLE_GRADIENT_CHECKPOINTING=true
export ENABLE_CPU_OFFLOAD=true

# 调整批次大小
export BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=4

# 使用更激进的量化
export QUANTIZATION_FORMAT=int4
```

### 2. GPU优化

```bash
# 优化CUDA设置
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0

# 启用混合精度
export ENABLE_MIXED_PRECISION=true

# 优化GPU内存分配
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 3. 网络优化

```bash
# 调整系统网络参数
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 87380 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# 启用连接池
export ENABLE_CONNECTION_POOLING=true
export MAX_CONNECTIONS=100
```

## 日志分析

### 常见错误模式

1. **CUDA错误**:
```bash
grep -i "cuda" logs/service.log
grep -i "out of memory" logs/service.log
```

2. **模型加载错误**:
```bash
grep -i "model" logs/service.log | grep -i "error"
grep -i "load" logs/service.log | grep -i "fail"
```

3. **API错误**:
```bash
grep -E "HTTP/[0-9.]+ [45][0-9][0-9]" logs/access.log
grep -i "exception" logs/service.log
```

### 日志级别调整

```bash
# 启用调试日志
export LOG_LEVEL=DEBUG

# 或在配置文件中修改
logging:
  level: DEBUG
  handlers:
    - console
    - file
```

## 紧急恢复程序

### 1. 服务完全无响应

```bash
#!/bin/bash
# emergency_recovery.sh

echo "开始紧急恢复..."

# 1. 强制停止所有相关进程
pkill -f uvicorn
pkill -f python.*model_service
docker stop qwen-thinking-service

# 2. 清理资源
docker system prune -f
sync && echo 3 > /proc/sys/vm/drop_caches

# 3. 重启服务
docker-compose up -d qwen-model-service

# 4. 等待服务启动
sleep 30

# 5. 验证服务
curl http://localhost:8000/health || echo "服务仍未恢复，需要人工干预"
```

### 2. 数据损坏恢复

```bash
#!/bin/bash
# data_recovery.sh

echo "开始数据恢复..."

# 1. 停止服务
docker stop qwen-thinking-service

# 2. 备份当前数据
cp -r ./models ./models.backup.$(date +%Y%m%d_%H%M%S)

# 3. 从备份恢复
if [ -d "./models.backup" ]; then
    rm -rf ./models
    cp -r ./models.backup ./models
    echo "从备份恢复完成"
else
    echo "未找到备份，重新下载模型..."
    rm -rf ./models/qwen3-4b-thinking
    huggingface-cli download Qwen/Qwen3-4B-Thinking-2507 --local-dir ./models/qwen3-4b-thinking
fi

# 4. 重启服务
docker-compose up -d qwen-model-service
```

## 联系支持

如果以上解决方案都无法解决问题，请收集以下信息并联系技术支持：

### 必需信息

1. **系统信息**:
```bash
uname -a
docker --version
nvidia-smi
free -h
df -h
```

2. **服务日志**:
```bash
docker logs qwen-thinking-service --tail=100 > service_logs.txt
tail -100 logs/service.log > application_logs.txt
```

3. **配置信息**:
```bash
cat docker-compose.yml > config_info.txt
env | grep -E "(MODEL|CUDA|QUANTIZATION)" >> config_info.txt
```

4. **错误复现步骤**:
- 详细描述问题发生的步骤
- 提供具体的错误信息
- 说明问题发生的频率和条件

### 支持渠道

- 技术支持邮箱: support@example.com
- GitHub Issues: https://github.com/your-repo/issues
- 文档反馈: docs@example.com

---

*本故障排除指南会持续更新，请定期检查最新版本。*