# 导出模型使用指南

## 测试摘要

- **测试时间**: 2025-08-27T21:24:10.778640
- **总模型数**: 8
- **可用模型**: 5
- **成功率**: 62.5%

## 推荐模型

### 🏆 最佳选择

- **最小体积**: `safe_quantized_output_int4_safe` - 适合存储受限环境
- **最快加载**: `safe_quantized_output_fp16_safe` - 适合频繁重启场景  
- **最快推理**: `fixed_quantized_output_fp16` - 适合实时应用

## 可用模型列表

| 模型名称 | 大小(MB) | 加载时间(s) | 推理时间(s) | 状态 |
|----------|----------|-------------|-------------|------|
| fixed_quantized_output_fp16 | 7687.5 | 7.5 | 9.37 | ✅ |
| size_demo_output_fp16 | 7687.5 | 0.9 | 46.55 | ✅ |
| safe_quantized_output_fp16_safe | 7687.5 | 0.6 | 46.33 | ✅ |
| safe_quantized_output_int4_safe | 7687.5 | 0.6 | 45.62 | ✅ |
| safe_quantized_output_int8_safe | 7687.5 | 0.7 | 44.98 | ✅ |

## 使用方法

### 加载模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 选择一个可用的模型路径
model_path = "path/to/your/chosen/model"

# 加载
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
```

### 推理示例

```python
# 基础推理
prompt = "什么是AES加密算法？"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# 思考推理
thinking_prompt = "<thinking>分析这个问题</thinking>请解释RSA算法。"
inputs = tokenizer(thinking_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=300, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 性能建议

- 存在CUDA相关错误，建议检查量化算法

## 故障排除

如果遇到问题，请检查：

1. **CUDA内存**: 确保GPU内存足够
2. **依赖版本**: 使用兼容的transformers版本
3. **模型完整性**: 确认模型文件完整下载
4. **设备兼容性**: 检查CUDA版本兼容性

---

*生成时间: 2025-08-27 21:24:10*
*测试版本: v1.0*
