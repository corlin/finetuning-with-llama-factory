#!/usr/bin/env python3
"""
环境检查脚本
检查系统环境、GPU状态和配置有效性

使用方法: uv run python scripts/check_environment.py
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def main():
    """环境检查主函数"""
    try:
        from environment_setup import EnvironmentSetup
        
        print("=== Qwen3-4B-Thinking 环境检查 ===")
        print()
        
        # 创建环境设置实例
        setup = EnvironmentSetup(project_root)
        
        # 生成GPU报告
        print(setup.gpu_detector.generate_system_report())
        print()
        
        # 验证配置
        validation = setup.config_manager.validate_configs()
        print("配置验证结果:")
        for key, value in validation.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}")
        
        print()
        
        # 获取优化建议
        recommendations = setup.gpu_detector.get_optimization_recommendations()
        if recommendations:
            print("优化建议:")
            for rec in recommendations:
                print(f"  {rec}")
        
        # 检查是否所有验证都通过
        all_passed = all(validation.values())
        if all_passed:
            print("\n🎉 环境检查全部通过！系统已准备就绪。")
            return 0
        else:
            print("\n⚠️  发现一些问题，请根据上述信息进行修复。")
            return 1
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请先运行 python setup.py 初始化环境")
        return 1
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())