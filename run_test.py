#!/usr/bin/env python3
"""
运行AI游戏测试的启动脚本
确保从正确的目录运行测试
"""

import os
import sys
import subprocess

def main():
    """主函数"""
    print("🎮 Flappy Bird AI 测试启动器")
    print("=" * 50)
    
    # 确保在项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"📂 当前工作目录: {os.getcwd()}")
    
    # 检查必要的文件是否存在
    required_files = [
        "game/wrapped_flappy_bird_fast.py",
        "test/test_ai_gameplay.py",
        "saved_networks"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 请确保在正确的项目目录中运行此脚本")
        return 1
    
    # 检查是否有保存的模型
    saved_networks_dir = "saved_networks"
    if os.path.exists(saved_networks_dir):
        model_files = [f for f in os.listdir(saved_networks_dir) if f.endswith('.pth')]
        if model_files:
            print(f"✅ 找到 {len(model_files)} 个模型文件")
            # 显示最新的几个模型
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(saved_networks_dir, x)), reverse=True)
            print("   最新模型:")
            for model in model_files[:3]:
                print(f"   - {model}")
        else:
            print("⚠️  saved_networks目录存在但没有模型文件")
            print("💡 请先运行训练: python deep_Q_oneStep.py")
    
    print("\n🚀 启动AI游戏测试...")
    
    # 运行测试脚本，指定加载bird-dqn-oneStep-200.pth模型
    try:
        result = subprocess.run([
            sys.executable, 
            "test/test_ai_gameplay.py",
            "--model", "saved_networks/bird-dqn-oneStep-200.pth",
            "--fps", "60"
        ], cwd=script_dir)
        return result.returncode
    except Exception as e:
        print(f"❌ 启动测试失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        input("\n按 Enter 键退出...")
    sys.exit(exit_code)