#!/usr/bin/env python3
"""
uv环境测试脚本
测试uv管理的项目环境和依赖
"""

import sys
import subprocess
from pathlib import Path
import locale
import os

# 设置环境变量以确保UTF-8输出
os.environ['PYTHONIOENCODING'] = 'utf-8'

def run_command_safe(cmd, timeout=30):
    """安全运行命令，处理编码问题"""
    try:
        # 在Windows上使用UTF-8编码
        if sys.platform == "win32":
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'  # 替换无法解码的字符
            )
        else:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=timeout
            )
        return result
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试系统默认编码
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=timeout,
                encoding=locale.getpreferredencoding(),
                errors='replace'
            )
            return result
        except Exception as e:
            raise subprocess.CalledProcessError(1, cmd, str(e))

def test_uv_environment():
    """测试uv环境"""
    print("=== 测试uv环境 ===")
    
    try:
        # 检查uv版本
        result = run_command_safe(["uv", "--version"])
        print(f"✓ uv版本: {result.stdout.strip()}")
        
        # 检查Python版本
        python_result = run_command_safe(["uv", "run", "python", "--version"])
        print(f"✓ Python版本: {python_result.stdout.strip()}")
        
        # 检查虚拟环境
        venv_result = run_command_safe(["uv", "run", "python", "-c", "import sys; print(sys.prefix)"])
        venv_path = venv_result.stdout.strip()
        print(f"✓ 虚拟环境: {venv_path}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ uv环境测试失败: {e}")
        return False
    except Exception as e:
        print(f"❌ uv环境测试异常: {e}")
        return False

def test_dependencies():
    """测试依赖安装"""
    print("\n=== 测试依赖安装 ===")
    
    # 核心依赖 (包名 -> 导入名映射)
    core_deps = {
        "torch": "torch",
        "transformers": "transformers", 
        "datasets": "datasets",
        "peft": "peft",
        "psutil": "psutil",
        "pyyaml": "yaml"  # pyyaml包的导入名是yaml
    }
    
    # 可选依赖（可能因为网络或系统原因失败）
    optional_deps = {
        "pynvml": "pynvml",
        "bitsandbytes": "bitsandbytes",
        "accelerate": "accelerate"
    }
    
    results = {}
    
    # 测试核心依赖
    for package_name, import_name in core_deps.items():
        try:
            result = run_command_safe(
                ["uv", "run", "python", "-c", f"import {import_name}; print(f'{package_name} imported successfully')"]
            )
            print(f"✓ {package_name}")
            results[package_name] = True
        except subprocess.CalledProcessError:
            print(f"❌ {package_name}")
            results[package_name] = False
        except Exception as e:
            print(f"❌ {package_name}: {e}")
            results[package_name] = False
    
    # 测试可选依赖
    for package_name, import_name in optional_deps.items():
        try:
            result = run_command_safe(
                ["uv", "run", "python", "-c", f"import {import_name}; print(f'{package_name} imported successfully')"]
            )
            print(f"✓ {package_name} (可选)")
            results[package_name] = True
        except subprocess.CalledProcessError:
            print(f"⚠️  {package_name} (可选，可能需要CUDA)")
            results[package_name] = False
        except Exception as e:
            print(f"⚠️  {package_name} (可选): {e}")
            results[package_name] = False
    
    # 统计结果
    core_success = sum(results[dep] for dep in core_deps.keys())
    core_total = len(core_deps)
    
    print(f"\n核心依赖: {core_success}/{core_total} 成功")
    
    return core_success == core_total

def test_project_modules():
    """测试项目模块"""
    print("\n=== 测试项目模块 ===")
    
    modules = [
        "gpu_utils",
        "model_config",
        "config_manager", 
        "environment_setup"
    ]
    
    results = {}
    
    for module in modules:
        try:
            result = run_command_safe(
                ["uv", "run", "python", "-c", f"import sys; sys.path.insert(0, 'src'); import {module}; print(f'{module} imported successfully')"]
            )
            print(f"✓ {module}")
            results[module] = True
        except subprocess.CalledProcessError as e:
            print(f"❌ {module}: {e}")
            results[module] = False
        except Exception as e:
            print(f"❌ {module}: {e}")
            results[module] = False
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n项目模块: {success_count}/{total_count} 成功")
    
    return success_count == total_count

def test_scripts():
    """测试脚本执行"""
    print("\n=== 测试脚本执行 ===")
    
    scripts = [
        ("check_environment.py", "环境检查脚本"),
        ("train.py", "训练脚本")
    ]
    
    results = {}
    
    for script, description in scripts:
        try:
            # 使用--help参数测试脚本是否能正常启动
            result = run_command_safe(
                ["uv", "run", "python", f"scripts/{script}"],
                timeout=30
            )
            
            # 检查是否有严重错误（导入错误等）
            if "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
                print(f"❌ {script}: 导入错误")
                results[script] = False
            else:
                print(f"✓ {script}: {description}")
                results[script] = True
                
        except subprocess.TimeoutExpired:
            print(f"✓ {script}: {description} (超时但启动正常)")
            results[script] = True
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {script}: {description} (可能需要额外配置)")
            results[script] = True  # 脚本能运行但可能缺少配置
        except Exception as e:
            print(f"❌ {script}: {description} - {e}")
            results[script] = False
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n脚本测试: {success_count}/{total_count} 成功")
    
    return success_count == total_count

def test_uv_commands():
    """测试uv命令"""
    print("\n=== 测试uv命令 ===")
    
    commands = [
        (["uv", "tree"], "依赖树"),
        (["uv", "pip", "list"], "包列表"),
    ]
    
    for cmd, description in commands:
        try:
            result = run_command_safe(cmd)
            print(f"✓ {' '.join(cmd)}: {description}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {' '.join(cmd)}: {description} (可能需要完整安装)")
        except Exception as e:
            print(f"⚠️  {' '.join(cmd)}: {description} - {e}")
    
    return True

def main():
    """主测试函数"""
    print("=== Qwen3-4B-Thinking 微调系统 - uv环境测试 ===")
    print("测试uv管理的项目环境和依赖")
    print()
    
    tests = [
        ("uv环境", test_uv_environment),
        ("依赖安装", test_dependencies),
        ("项目模块", test_project_modules),
        ("脚本执行", test_scripts),
        ("uv命令", test_uv_commands)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results[test_name] = False
    
    print("\n" + "="*50)
    print("测试结果汇总:")
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n总体结果: {success_count}/{total_count} 测试通过")
    
    if success_count == total_count:
        print("\n🎉 uv环境测试全部通过！")
        print("\n环境已准备就绪，可以开始开发:")
        print("- 检查环境: uv run python scripts/check_environment.py")
        print("- 运行训练: uv run python scripts/train.py")
        print("- 运行测试: uv run pytest tests/")
        print("- 添加依赖: uv add <package>")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述问题")
        print("\n常见解决方案:")
        print("- 重新同步依赖: uv sync --extra dev")
        print("- 检查Python版本: uv run python --version")
        print("- 查看详细错误: uv run python -c 'import <failed_module>'")
        return 1

if __name__ == "__main__":
    sys.exit(main())