# Qwen3-4B-Thinking 模型服务运维指南

## 概述

本文档提供 Qwen3-4B-Thinking 模型服务的完整运维指南，包括服务启动、监控、故障排除和性能调优的详细说明。

## 服务启动

### 1. 启动前检查

#### 1.1 系统资源检查

```bash
#!/bin/bash
# scripts/pre_start_check.sh

echo "=== 系统资源检查 ==="

# 检查CPU
echo "CPU信息:"
lscpu | grep -E "Model name|CPU\(s\)|Thread"

# 检查内存
echo -e "\n内存信息:"
free -h

# 检查磁盘空间
echo -e "\n磁盘空间:"
df -h | grep -E "/$|/app"

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "\nGPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader
else
    echo -e "\n警告: 未检测到NVIDIA GPU"
fi

# 检查Docker
echo -e "\nDocker版本:"
docker --version
docker-compose --version

# 检查端口占用
echo -e "\n端口检查:"
netstat -tlnp | grep :8000 || echo "端口8000可用"
```

#### 1.2 模型文件检查

```bash
#!/bin/bash
# scripts/check_model_files.sh

MODEL_PATH=${MODEL_PATH:-"./models/qwen3-4b-thinking"}

echo "=== 模型文件检查 ==="
echo "模型路径: $MODEL_PATH"

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型目录不存在"
    exit 1
fi

# 检查必需文件
REQUIRED_FILES=(
    "config.json"
    "pytorch_model.bin"
    "tokenizer.json"
    "tokenizer_config.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$MODEL_PATH/$file" ]; then
        echo "✓ $file 存在"
        ls -lh "$MODEL_PATH/$file"
    else
        echo "✗ $file 缺失"
    fi
done

# 检查模型大小
echo -e "\n模型目录大小:"
du -sh "$MODEL_PATH"
```

### 2. 服务启动流程

#### 2.1 Docker 启动

```bash
#!/bin/bash
# scripts/start_docker_service.sh

set -e

echo "=== 启动 Qwen3-4B-Thinking 服务 ==="

# 检查Docker服务
if ! systemctl is-active --quiet docker; then
    echo "启动Docker服务..."
    sudo systemctl start docker
fi

# 检查NVIDIA Docker
if command -v nvidia-docker &> /dev/null; then
    echo "NVIDIA Docker 可用"
else
    echo "警告: NVIDIA Docker 不可用，将使用CPU模式"
fi

# 构建镜像（如果需要）
if [ "$1" = "--build" ]; then
    echo "构建Docker镜像..."
    docker build -t qwen-thinking-service:latest .
fi

# 启动服务
echo "启动服务容器..."
docker-compose up -d qwen-model-service

# 等待服务就绪
echo "等待服务启动..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "服务启动成功！"
        break
    fi
    echo "等待中... ($i/30)"
    sleep 2
done

# 显示服务状态
docker-compose ps
docker-compose logs --tail=20 qwen-model-service
```

#### 2.2 本地启动

```bash
#!/bin/bash
# scripts/start_local_service.sh

set -e

echo "=== 本地启动 Qwen3-4B-Thinking 服务 ==="

# 检查Python环境
if ! command -v uv &> /dev/null; then
    echo "错误: uv 未安装"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
uv sync --check

# 设置环境变量
export PYTHONPATH="$(pwd):$PYTHONPATH"
export MODEL_PATH=${MODEL_PATH:-"./models/qwen3-4b-thinking"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# 启动服务
echo "启动服务..."
uv run python -m uvicorn src.model_service:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --access-log &

SERVICE_PID=$!
echo "服务PID: $SERVICE_PID"
echo $SERVICE_PID > service.pid

# 等待服务就绪
echo "等待服务启动..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "服务启动成功！"
        break
    fi
    echo "等待中... ($i/30)"
    sleep 2
done
```

### 3. 服务验证

```bash
#!/bin/bash
# scripts/verify_service.sh

echo "=== 服务验证 ==="

# 健康检查
echo "1. 健康检查:"
curl -s http://localhost:8000/health | jq '.'

# 模型信息
echo -e "\n2. 模型信息:"
curl -s http://localhost:8000/model_info | jq '.'

# 简单生成测试
echo -e "\n3. 生成测试:"
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "什么是AES加密？",
    "max_length": 100,
    "temperature": 0.7
  }' | jq '.generated_text'

# 思考推理测试
echo -e "\n4. 思考推理测试:"
curl -X POST "http://localhost:8000/thinking" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "RSA和AES的区别是什么？",
    "thinking_depth": 2
  }' | jq '.final_answer'

echo -e "\n服务验证完成！"
```

## 监控

### 1. 实时监控

#### 1.1 系统监控脚本

```bash
#!/bin/bash
# scripts/monitor_system.sh

while true; do
    clear
    echo "=== Qwen3-4B-Thinking 服务监控 ==="
    echo "时间: $(date)"
    echo

    # 服务状态
    echo "1. 服务状态:"
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✓ 服务运行正常"
    else
        echo "✗ 服务异常"
    fi

    # 系统资源
    echo -e "\n2. 系统资源:"
    echo "CPU使用率: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
    echo "内存使用: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
    
    # GPU状态
    if command -v nvidia-smi &> /dev/null; then
        echo -e "\n3. GPU状态:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        awk -F, '{printf "GPU利用率: %s%%, 内存: %sMB/%sMB, 温度: %s°C\n", $1, $2, $3, $4}'
    fi

    # 服务统计
    echo -e "\n4. 服务统计:"
    curl -s http://localhost:8000/stats | jq -r '
        "总请求数: \(.total_requests)",
        "成功请求: \(.successful_requests)",
        "失败请求: \(.failed_requests)",
        "平均响应时间: \(.average_response_time)s",
        "运行时间: \(.uptime_hours)小时"
    ' 2>/dev/null || echo "无法获取服务统计"

    # 容器状态（如果使用Docker）
    if command -v docker &> /dev/null; then
        echo -e "\n5. 容器状态:"
        docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" | grep qwen || echo "容器未运行"
    fi

    sleep 5
done
```

#### 1.2 日志监控

```bash
#!/bin/bash
# scripts/monitor_logs.sh

LOG_DIR=${LOG_DIR:-"./logs"}

echo "=== 日志监控 ==="

# 实时日志
echo "1. 实时服务日志:"
if [ -f "$LOG_DIR/service.log" ]; then
    tail -f "$LOG_DIR/service.log" &
    LOG_PID=$!
fi

# 错误日志监控
echo -e "\n2. 错误监控:"
if [ -f "$LOG_DIR/error.log" ]; then
    tail -f "$LOG_DIR/error.log" | while read line; do
        echo "[ERROR] $(date): $line"
        # 可以添加告警逻辑
    done &
    ERROR_PID=$!
fi

# 清理函数
cleanup() {
    echo "停止日志监控..."
    [ ! -z "$LOG_PID" ] && kill $LOG_PID 2>/dev/null
    [ ! -z "$ERROR_PID" ] && kill $ERROR_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

wait
```

### 2. 性能监控

#### 2.1 性能指标收集

```python
# scripts/collect_metrics.py
"""
性能指标收集脚本
"""

import time
import json
import requests
import psutil
import subprocess
from datetime import datetime
from typing import Dict, Any

class MetricsCollector:
    def __init__(self, service_url: str = "http://localhost:8000"):
        self.service_url = service_url
        self.metrics_history = []
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / (1024**3),
            "disk_total_gb": disk.total / (1024**3)
        }
        
        # GPU指标
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(',')
                metrics.update({
                    "gpu_utilization": float(gpu_data[0]),
                    "gpu_memory_used_mb": float(gpu_data[1]),
                    "gpu_memory_total_mb": float(gpu_data[2]),
                    "gpu_temperature": float(gpu_data[3])
                })
        except Exception as e:
            print(f"GPU指标收集失败: {e}")
        
        return metrics
    
    def collect_service_metrics(self) -> Dict[str, Any]:
        """收集服务指标"""
        try:
            # 健康检查
            health_response = requests.get(f"{self.service_url}/health", timeout=5)
            health_data = health_response.json()
            
            # 服务统计
            stats_response = requests.get(f"{self.service_url}/stats", timeout=5)
            stats_data = stats_response.json()
            
            return {
                "service_status": health_data.get("status"),
                "model_loaded": health_data.get("model_loaded"),
                "uptime_seconds": health_data.get("uptime_seconds"),
                "total_requests": stats_data.get("total_requests"),
                "successful_requests": stats_data.get("successful_requests"),
                "failed_requests": stats_data.get("failed_requests"),
                "average_response_time": stats_data.get("average_response_time"),
                "model_memory_usage": stats_data.get("model_memory_usage")
            }
        except Exception as e:
            print(f"服务指标收集失败: {e}")
            return {"service_status": "error", "error": str(e)}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """收集所有指标"""
        system_metrics = self.collect_system_metrics()
        service_metrics = self.collect_service_metrics()
        
        all_metrics = {**system_metrics, **service_metrics}
        self.metrics_history.append(all_metrics)
        
        # 保持最近1000条记录
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return all_metrics
    
    def save_metrics(self, filename: str = None):
        """保存指标到文件"""
        if not filename:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        
        print(f"指标已保存到: {filename}")
    
    def run_continuous_monitoring(self, interval: int = 60):
        """持续监控"""
        print(f"开始持续监控，间隔: {interval}秒")
        
        try:
            while True:
                metrics = self.collect_all_metrics()
                print(f"[{metrics['timestamp']}] "
                      f"CPU: {metrics.get('cpu_percent', 0):.1f}% "
                      f"内存: {metrics.get('memory_percent', 0):.1f}% "
                      f"服务: {metrics.get('service_status', 'unknown')}")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n停止监控")
            self.save_metrics()

if __name__ == "__main__":
    collector = MetricsCollector()
    collector.run_continuous_monitoring()
```

### 3. 告警系统

#### 3.1 告警规则配置

```python
# scripts/alert_system.py
"""
告警系统
"""

import time
import smtplib
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class AlertRule:
    name: str
    condition: str
    threshold: float
    severity: str
    message: str

@dataclass
class Alert:
    rule_name: str
    severity: str
    message: str
    timestamp: str
    value: float

class AlertSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = self._load_alert_rules()
        self.active_alerts = {}
    
    def _load_alert_rules(self) -> List[AlertRule]:
        """加载告警规则"""
        return [
            AlertRule(
                name="high_cpu_usage",
                condition="cpu_percent > threshold",
                threshold=80.0,
                severity="warning",
                message="CPU使用率过高: {value}%"
            ),
            AlertRule(
                name="high_memory_usage",
                condition="memory_percent > threshold",
                threshold=85.0,
                severity="warning",
                message="内存使用率过高: {value}%"
            ),
            AlertRule(
                name="gpu_temperature_high",
                condition="gpu_temperature > threshold",
                threshold=80.0,
                severity="critical",
                message="GPU温度过高: {value}°C"
            ),
            AlertRule(
                name="service_down",
                condition="service_status != 'healthy'",
                threshold=0,
                severity="critical",
                message="服务状态异常: {value}"
            ),
            AlertRule(
                name="high_error_rate",
                condition="error_rate > threshold",
                threshold=0.1,
                severity="warning",
                message="错误率过高: {value}%"
            )
        ]
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Alert]:
        """检查告警条件"""
        alerts = []
        
        for rule in self.alert_rules:
            try:
                if self._evaluate_condition(rule, metrics):
                    alert = Alert(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=rule.message.format(
                            value=self._get_metric_value(rule, metrics)
                        ),
                        timestamp=metrics.get("timestamp", ""),
                        value=self._get_metric_value(rule, metrics)
                    )
                    alerts.append(alert)
            except Exception as e:
                print(f"告警规则 {rule.name} 评估失败: {e}")
        
        return alerts
    
    def _evaluate_condition(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """评估告警条件"""
        if rule.name == "high_cpu_usage":
            return metrics.get("cpu_percent", 0) > rule.threshold
        elif rule.name == "high_memory_usage":
            return metrics.get("memory_percent", 0) > rule.threshold
        elif rule.name == "gpu_temperature_high":
            return metrics.get("gpu_temperature", 0) > rule.threshold
        elif rule.name == "service_down":
            return metrics.get("service_status") != "healthy"
        elif rule.name == "high_error_rate":
            total = metrics.get("total_requests", 1)
            failed = metrics.get("failed_requests", 0)
            error_rate = failed / total if total > 0 else 0
            return error_rate > rule.threshold
        
        return False
    
    def _get_metric_value(self, rule: AlertRule, metrics: Dict[str, Any]) -> Any:
        """获取指标值"""
        if rule.name == "high_cpu_usage":
            return metrics.get("cpu_percent", 0)
        elif rule.name == "high_memory_usage":
            return metrics.get("memory_percent", 0)
        elif rule.name == "gpu_temperature_high":
            return metrics.get("gpu_temperature", 0)
        elif rule.name == "service_down":
            return metrics.get("service_status", "unknown")
        elif rule.name == "high_error_rate":
            total = metrics.get("total_requests", 1)
            failed = metrics.get("failed_requests", 0)
            return (failed / total * 100) if total > 0 else 0
        
        return 0
    
    def send_alert(self, alert: Alert):
        """发送告警"""
        print(f"[ALERT] {alert.severity.upper()}: {alert.message}")
        
        # 发送邮件告警
        if self.config.get("email", {}).get("enabled"):
            self._send_email_alert(alert)
        
        # 发送Webhook告警
        if self.config.get("webhook", {}).get("enabled"):
            self._send_webhook_alert(alert)
    
    def _send_email_alert(self, alert: Alert):
        """发送邮件告警"""
        try:
            email_config = self.config["email"]
            
            msg = MimeMultipart()
            msg['From'] = email_config["from"]
            msg['To'] = email_config["to"]
            msg['Subject'] = f"[{alert.severity.upper()}] Qwen服务告警"
            
            body = f"""
            告警规则: {alert.rule_name}
            严重级别: {alert.severity}
            告警信息: {alert.message}
            发生时间: {alert.timestamp}
            当前值: {alert.value}
            """
            
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            server.send_message(msg)
            server.quit()
            
            print(f"邮件告警已发送: {alert.message}")
        except Exception as e:
            print(f"邮件告警发送失败: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """发送Webhook告警"""
        try:
            webhook_config = self.config["webhook"]
            
            payload = {
                "rule_name": alert.rule_name,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "value": alert.value
            }
            
            response = requests.post(
                webhook_config["url"],
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Webhook告警已发送: {alert.message}")
            else:
                print(f"Webhook告警发送失败: {response.status_code}")
        except Exception as e:
            print(f"Webhook告警发送失败: {e}")

# 告警配置示例
ALERT_CONFIG = {
    "email": {
        "enabled": False,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your-email@gmail.com",
        "password": "your-password",
        "from": "your-email@gmail.com",
        "to": "admin@example.com"
    },
    "webhook": {
        "enabled": False,
        "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    }
}

if __name__ == "__main__":
    alert_system = AlertSystem(ALERT_CONFIG)
    
    # 示例指标
    test_metrics = {
        "timestamp": "2025-01-26T10:30:00",
        "cpu_percent": 85.0,
        "memory_percent": 90.0,
        "gpu_temperature": 82.0,
        "service_status": "healthy",
        "total_requests": 1000,
        "failed_requests": 50
    }
    
    alerts = alert_system.check_alerts(test_metrics)
    for alert in alerts:
        alert_system.send_alert(alert)
```

## 故障排除

### 1. 常见故障诊断

#### 1.1 服务启动失败

```bash
#!/bin/bash
# scripts/diagnose_startup_failure.sh

echo "=== 服务启动失败诊断 ==="

# 检查端口占用
echo "1. 检查端口占用:"
netstat -tlnp | grep :8000

# 检查Docker状态
echo -e "\n2. Docker状态:"
systemctl status docker
docker ps -a | grep qwen

# 检查容器日志
echo -e "\n3. 容器日志:"
docker logs qwen-thinking-service --tail=50

# 检查模型文件
echo -e "\n4. 模型文件检查:"
ls -la ./models/qwen3-4b-thinking/ || echo "模型目录不存在"

# 检查磁盘空间
echo -e "\n5. 磁盘空间:"
df -h

# 检查内存
echo -e "\n6. 内存状态:"
free -h

# 检查GPU
echo -e "\n7. GPU状态:"
nvidia-smi || echo "GPU不可用"
```

#### 1.2 内存不足问题

```bash
#!/bin/bash
# scripts/fix_memory_issues.sh

echo "=== 内存不足问题修复 ==="

# 检查内存使用
echo "1. 当前内存使用:"
free -h

# 清理系统缓存
echo -e "\n2. 清理系统缓存:"
sudo sync
sudo echo 3 > /proc/sys/vm/drop_caches

# 检查大内存进程
echo -e "\n3. 大内存进程:"
ps aux --sort=-%mem | head -10

# 调整服务配置
echo -e "\n4. 调整服务配置:"
cat > configs/memory_optimized.yaml << EOF
model:
  quantization:
    format: "int4"  # 使用更激进的量化
    enable: true
  
inference:
  batch_size: 1     # 减少批次大小
  max_length: 512   # 减少最大长度
  
memory:
  enable_optimization: true
  gradient_checkpointing: true
  cpu_offload: true  # 启用CPU卸载
EOF

echo "内存优化配置已生成: configs/memory_optimized.yaml"
echo "请重启服务以应用配置"
```

#### 1.3 GPU问题诊断

```bash
#!/bin/bash
# scripts/diagnose_gpu_issues.sh

echo "=== GPU问题诊断 ==="

# 检查NVIDIA驱动
echo "1. NVIDIA驱动版本:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader || echo "NVIDIA驱动未安装"

# 检查CUDA版本
echo -e "\n2. CUDA版本:"
nvcc --version || echo "CUDA未安装"

# 检查Docker GPU支持
echo -e "\n3. Docker GPU支持:"
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi || echo "Docker GPU支持异常"

# 检查GPU内存
echo -e "\n4. GPU内存使用:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# 检查GPU进程
echo -e "\n5. GPU进程:"
nvidia-smi pmon -c 1

# 重置GPU
echo -e "\n6. 重置GPU (如果需要):"
echo "sudo nvidia-smi --gpu-reset"
echo "注意: 这将终止所有GPU进程"
```

### 2. 性能问题排查

#### 2.1 响应时间过长

```python
# scripts/diagnose_performance.py
"""
性能问题诊断脚本
"""

import time
import requests
import statistics
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

class PerformanceDiagnostic:
    def __init__(self, service_url: str = "http://localhost:8000"):
        self.service_url = service_url
    
    def test_response_time(self, num_requests: int = 10) -> dict:
        """测试响应时间"""
        print(f"测试响应时间 ({num_requests} 次请求)...")
        
        response_times = []
        errors = 0
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.service_url}/generate",
                    json={
                        "prompt": f"测试请求 {i+1}: 什么是加密算法？",
                        "max_length": 100,
                        "temperature": 0.7
                    },
                    timeout=30
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
                else:
                    errors += 1
                    print(f"请求 {i+1} 失败: {response.status_code}")
            
            except Exception as e:
                errors += 1
                print(f"请求 {i+1} 异常: {e}")
        
        if response_times:
            return {
                "total_requests": num_requests,
                "successful_requests": len(response_times),
                "failed_requests": errors,
                "min_time": min(response_times),
                "max_time": max(response_times),
                "avg_time": statistics.mean(response_times),
                "median_time": statistics.median(response_times),
                "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
        else:
            return {"error": "所有请求都失败了"}
    
    def test_concurrent_requests(self, num_concurrent: int = 5, num_requests: int = 20) -> dict:
        """测试并发请求"""
        print(f"测试并发请求 ({num_concurrent} 并发, {num_requests} 总请求)...")
        
        def make_request(request_id):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.service_url}/generate",
                    json={
                        "prompt": f"并发测试 {request_id}: 解释对称加密",
                        "max_length": 50,
                        "temperature": 0.7
                    },
                    timeout=30
                )
                end_time = time.time()
                
                return {
                    "request_id": request_id,
                    "success": response.status_code == 200,
                    "response_time": end_time - start_time,
                    "status_code": response.status_code
                }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e),
                    "response_time": None
                }
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            results = list(executor.map(make_request, range(num_requests)))
        
        end_time = time.time()
        
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        if successful_results:
            response_times = [r["response_time"] for r in successful_results]
            return {
                "total_time": end_time - start_time,
                "total_requests": num_requests,
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "requests_per_second": num_requests / (end_time - start_time),
                "avg_response_time": statistics.mean(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times)
            }
        else:
            return {"error": "所有并发请求都失败了"}
    
    def analyze_bottlenecks(self):
        """分析性能瓶颈"""
        print("分析性能瓶颈...")
        
        # 获取系统指标
        try:
            health_response = requests.get(f"{self.service_url}/health")
            health_data = health_response.json()
            
            stats_response = requests.get(f"{self.service_url}/stats")
            stats_data = stats_response.json()
            
            print("\n系统状态:")
            print(f"  内存使用: {health_data.get('memory_usage', {}).get('percent', 0):.1f}%")
            print(f"  GPU内存: {health_data.get('memory_usage', {}).get('gpu_used_gb', 0):.1f}GB")
            print(f"  平均响应时间: {stats_data.get('average_response_time', 0):.2f}s")
            print(f"  成功率: {stats_data.get('successful_requests', 0) / max(stats_data.get('total_requests', 1), 1) * 100:.1f}%")
            
            # 性能建议
            print("\n性能优化建议:")
            
            if health_data.get('memory_usage', {}).get('percent', 0) > 80:
                print("  - 内存使用率过高，建议启用CPU卸载或减少批次大小")
            
            if stats_data.get('average_response_time', 0) > 5:
                print("  - 响应时间过长，建议检查模型量化设置或GPU性能")
            
            success_rate = stats_data.get('successful_requests', 0) / max(stats_data.get('total_requests', 1), 1)
            if success_rate < 0.95:
                print("  - 成功率偏低，建议检查错误日志和资源限制")
            
        except Exception as e:
            print(f"无法获取系统指标: {e}")
    
    def generate_performance_report(self):
        """生成性能报告"""
        print("生成性能报告...")
        
        # 单请求测试
        single_test = self.test_response_time(10)
        
        # 并发测试
        concurrent_test = self.test_concurrent_requests(3, 15)
        
        # 生成报告
        report = f"""
# 性能测试报告

## 单请求测试
- 总请求数: {single_test.get('total_requests', 0)}
- 成功请求: {single_test.get('successful_requests', 0)}
- 失败请求: {single_test.get('failed_requests', 0)}
- 平均响应时间: {single_test.get('avg_time', 0):.2f}s
- 最小响应时间: {single_test.get('min_time', 0):.2f}s
- 最大响应时间: {single_test.get('max_time', 0):.2f}s
- 标准差: {single_test.get('std_dev', 0):.2f}s

## 并发测试
- 总请求数: {concurrent_test.get('total_requests', 0)}
- 成功请求: {concurrent_test.get('successful_requests', 0)}
- 失败请求: {concurrent_test.get('failed_requests', 0)}
- 总耗时: {concurrent_test.get('total_time', 0):.2f}s
- 请求/秒: {concurrent_test.get('requests_per_second', 0):.2f}
- 平均响应时间: {concurrent_test.get('avg_response_time', 0):.2f}s

## 建议
"""
        
        # 添加建议
        if single_test.get('avg_time', 0) > 3:
            report += "- 单请求响应时间较长，建议优化模型或硬件配置\n"
        
        if concurrent_test.get('requests_per_second', 0) < 1:
            report += "- 并发处理能力较低，建议增加GPU资源或优化批处理\n"
        
        # 保存报告
        with open(f"performance_report_{int(time.time())}.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("性能报告已生成")
        return report

if __name__ == "__main__":
    diagnostic = PerformanceDiagnostic()
    diagnostic.generate_performance_report()
    diagnostic.analyze_bottlenecks()
```

### 3. 自动修复脚本

```bash
#!/bin/bash
# scripts/auto_recovery.sh

echo "=== 自动故障恢复 ==="

# 检查服务状态
check_service_health() {
    curl -s http://localhost:8000/health > /dev/null
    return $?
}

# 重启服务
restart_service() {
    echo "重启服务..."
    docker-compose restart qwen-model-service
    sleep 30
}

# 清理资源
cleanup_resources() {
    echo "清理系统资源..."
    
    # 清理内存
    sudo sync
    sudo echo 3 > /proc/sys/vm/drop_caches
    
    # 清理GPU内存
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --gpu-reset || true
    fi
    
    # 清理Docker资源
    docker system prune -f
}

# 主恢复流程
main() {
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "尝试 $attempt/$max_attempts"
        
        if check_service_health; then
            echo "服务运行正常"
            exit 0
        fi
        
        echo "服务异常，开始恢复..."
        
        # 清理资源
        cleanup_resources
        
        # 重启服务
        restart_service
        
        # 等待服务启动
        sleep 60
        
        if check_service_health; then
            echo "服务恢复成功"
            exit 0
        fi
        
        attempt=$((attempt + 1))
    done
    
    echo "自动恢复失败，需要人工干预"
    exit 1
}

main "$@"
```

## 性能调优

### 1. 模型优化

```python
# scripts/optimize_model.py
"""
模型性能优化脚本
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelOptimizer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """加载模型"""
        print("加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
    
    def optimize_for_inference(self):
        """推理优化"""
        print("优化模型用于推理...")
        
        # 设置为评估模式
        self.model.eval()
        
        # 禁用梯度计算
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 编译模型（PyTorch 2.0+）
        if hasattr(torch, 'compile'):
            print("编译模型...")
            self.model = torch.compile(self.model)
        
        # 融合操作
        if hasattr(self.model, 'fuse_modules'):
            print("融合模块...")
            self.model.fuse_modules()
    
    def quantize_model(self, quantization_type: str = "dynamic"):
        """量化模型"""
        print(f"量化模型: {quantization_type}")
        
        if quantization_type == "dynamic":
            # 动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            return quantized_model
        
        elif quantization_type == "static":
            # 静态量化（需要校准数据）
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            
            # 这里需要运行校准数据
            # calibrate_model(self.model, calibration_data)
            
            quantized_model = torch.quantization.convert(self.model, inplace=False)
            return quantized_model
    
    def benchmark_model(self, num_runs: int = 10):
        """基准测试"""
        print(f"基准测试 ({num_runs} 次运行)...")
        
        test_input = "什么是AES加密算法？"
        inputs = self.tokenizer.encode(test_input, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = self.model.generate(inputs, max_length=inputs.shape[1] + 50)
        
        # 基准测试
        times = []
        
        for i in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    do_sample=True,
                    temperature=0.7
                )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            times.append(end_time - start_time)
            print(f"运行 {i+1}: {end_time - start_time:.3f}s")
        
        avg_time = sum(times) / len(times)
        print(f"平均时间: {avg_time:.3f}s")
        print(f"吞吐量: {100 / avg_time:.1f} tokens/s")
        
        return {
            "average_time": avg_time,
            "min_time": min(times),
            "max_time": max(times),
            "throughput": 100 / avg_time
        }

if __name__ == "__main__":
    optimizer = ModelOptimizer("./models/qwen3-4b-thinking")
    optimizer.load_model()
    optimizer.optimize_for_inference()
    
    # 基准测试原始模型
    print("=== 原始模型基准测试 ===")
    original_benchmark = optimizer.benchmark_model()
    
    # 量化模型
    quantized_model = optimizer.quantize_model("dynamic")
    optimizer.model = quantized_model
    
    # 基准测试量化模型
    print("=== 量化模型基准测试 ===")
    quantized_benchmark = optimizer.benchmark_model()
    
    # 比较结果
    print("=== 性能对比 ===")
    print(f"原始模型平均时间: {original_benchmark['average_time']:.3f}s")
    print(f"量化模型平均时间: {quantized_benchmark['average_time']:.3f}s")
    print(f"性能提升: {original_benchmark['average_time'] / quantized_benchmark['average_time']:.2f}x")
```

### 2. 系统优化

```bash
#!/bin/bash
# scripts/system_optimization.sh

echo "=== 系统性能优化 ==="

# 1. 内核参数优化
echo "1. 优化内核参数..."
sudo tee -a /etc/sysctl.conf << EOF
# 网络优化
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 65536 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728

# 内存优化
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# 文件系统优化
fs.file-max = 2097152
EOF

sudo sysctl -p

# 2. GPU优化
echo "2. 优化GPU设置..."
if command -v nvidia-smi &> /dev/null; then
    # 设置GPU性能模式
    sudo nvidia-smi -pm 1
    
    # 设置最大时钟频率
    sudo nvidia-smi -ac 877,1215  # 根据具体GPU调整
    
    # 设置功耗限制
    sudo nvidia-smi -pl 300  # 根据具体GPU调整
fi

# 3. CPU优化
echo "3. 优化CPU设置..."
# 设置CPU调度器为性能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 4. Docker优化
echo "4. 优化Docker配置..."
sudo tee /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

sudo systemctl restart docker

# 5. 文件系统优化
echo "5. 优化文件系统..."
# 挂载tmpfs用于临时文件
sudo mkdir -p /tmp/qwen-cache
sudo mount -t tmpfs -o size=2G tmpfs /tmp/qwen-cache

echo "系统优化完成！"
echo "建议重启系统以确保所有优化生效。"
```

这个运维指南提供了完整的服务管理、监控、故障排除和性能调优方案。现在让我创建测试脚本来验证服务接口的功能完整性。

<function_calls>
<invoke name="fsWrite">
<parameter name="path">tests/test_model_service.py