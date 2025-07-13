#!/usr/bin/env python3
"""
网络维度测试脚本
验证FixedOptimizedDQN网络架构的tensor维度匹配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从主文件导入网络定义
try:
    from deep_Q_oneStep import FixedOptimizedDQN, BATCH, ACTIONS
    print(f"✅ 成功导入网络定义")
    print(f"   BATCH = {BATCH}")
    print(f"   ACTIONS = {ACTIONS}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_network_dimensions():
    """测试网络各层的tensor维度"""
    print("\n🔍 开始网络维度测试...")
    
    # 创建网络实例
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")
    
    network = FixedOptimizedDQN(ACTIONS).to(device)
    network.eval()  # 设置为评估模式
    
    # 创建测试输入 (BATCH, 4, 80, 80)
    test_input = torch.randn(BATCH, 4, 80, 80).to(device)
    print(f"   输入维度: {test_input.shape}")
    
    try:
        # 逐层测试
        print("\n📊 逐层维度跟踪:")
        
        # 卷积层1
        x = network.conv1(test_input)
        print(f"   Conv1 输出: {x.shape}")
        x = network.bn1(x)
        x = F.relu(x)
        print(f"   Conv1+BN+ReLU: {x.shape}")
        
        # 卷积层2  
        x = network.conv2(x)
        print(f"   Conv2 输出: {x.shape}")
        x = network.bn2(x)
        x = F.relu(x)
        print(f"   Conv2+BN+ReLU: {x.shape}")
        
        # 自适应池化
        x = network.adaptive_pool(x)
        print(f"   AdaptivePool: {x.shape}")
        
        # 展平
        x = x.reshape(x.size(0), -1)
        print(f"   Flatten: {x.shape}")
        
        # 全连接层1
        x = network.fc1(x)
        print(f"   FC1 输出: {x.shape}")
        x = F.relu(x)
        
        # 全连接层2
        x = network.fc2(x)
        print(f"   FC2 输出: {x.shape}")
        x = F.relu(x)
        
        # 全连接层3 (输出层)
        x = network.fc3(x)
        print(f"   FC3 (最终输出): {x.shape}")
        
        # 完整前向传播测试
        print("\n🚀 完整前向传播测试:")
        output = network(test_input)
        print(f"   网络输出维度: {output.shape}")
        print(f"   预期输出维度: ({BATCH}, {ACTIONS})")
        
        # 验证输出维度
        expected_shape = (BATCH, ACTIONS)
        if output.shape == expected_shape:
            print("   ✅ 输出维度匹配!")
        else:
            print(f"   ❌ 输出维度不匹配! 期望: {expected_shape}, 实际: {output.shape}")
            return False
            
        # 验证输出数值范围
        print(f"   输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"   输出数值类型: {output.dtype}")
        
        # 内存使用检查
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(device) / 1024**2  # MB
            memory_cached = torch.cuda.memory_reserved(device) / 1024**2  # MB
            print(f"\n💾 GPU内存使用:")
            print(f"   已用内存: {memory_used:.1f}MB")
            print(f"   缓存内存: {memory_cached:.1f}MB")
            
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False

def test_gradient_flow():
    """测试梯度流动"""
    print("\n🔄 梯度流动测试...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = FixedOptimizedDQN(ACTIONS).to(device)
    network.train()  # 设置为训练模式
    
    # 创建测试数据
    test_input = torch.randn(BATCH, 4, 80, 80, requires_grad=True).to(device)
    target = torch.randint(0, ACTIONS, (BATCH,)).to(device)
    
    try:
        # 前向传播
        output = network(test_input)
        
        # 计算损失 (使用交叉熵)
        loss = F.cross_entropy(output, target)
        print(f"   损失值: {loss.item():.4f}")
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        total_grad_norm = 0
        param_count = 0
        for name, param in network.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm**2
                param_count += 1
                if 'fc' in name and 'weight' in name:  # 只显示全连接层权重梯度
                    print(f"   {name} 梯度范数: {grad_norm:.6f}")
        
        total_grad_norm = total_grad_norm**0.5
        print(f"   总梯度范数: {total_grad_norm:.6f}")
        print(f"   有梯度的参数数量: {param_count}")
        
        if total_grad_norm > 0:
            print("   ✅ 梯度流动正常!")
            return True
        else:
            print("   ❌ 梯度为零，可能存在梯度消失问题!")
            return False
            
    except Exception as e:
        print(f"❌ 梯度测试失败: {e}")
        return False

def test_memory_efficiency():
    """测试内存效率"""
    print("\n💾 内存效率测试...")
    
    if not torch.cuda.is_available():
        print("   ⚠️  CPU模式，跳过GPU内存测试")
        return True
        
    device = torch.device("cuda")
    
    # 清空GPU缓存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(device)
    
    try:
        # 创建网络
        network = FixedOptimizedDQN(ACTIONS).to(device)
        network_memory = torch.cuda.memory_allocated(device) - initial_memory
        
        # 创建批次数据
        test_input = torch.randn(BATCH, 4, 80, 80).to(device)
        data_memory = torch.cuda.memory_allocated(device) - network_memory - initial_memory
        
        # 前向传播
        output = network(test_input)
        forward_memory = torch.cuda.memory_allocated(device) - data_memory - network_memory - initial_memory
        
        print(f"   网络参数内存: {network_memory / 1024**2:.1f}MB")
        print(f"   批次数据内存: {data_memory / 1024**2:.1f}MB") 
        print(f"   前向传播额外内存: {forward_memory / 1024**2:.1f}MB")
        print(f"   总内存使用: {torch.cuda.memory_allocated(device) / 1024**2:.1f}MB")
        
        # 计算理论内存需求
        theoretical_data = BATCH * 4 * 80 * 80 * 4 / 1024**2  # float32
        print(f"   理论数据内存: {theoretical_data:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 内存测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 FixedOptimizedDQN 网络架构测试")
    print("=" * 50)
    
    # 执行所有测试
    tests = [
        ("网络维度测试", test_network_dimensions),
        ("梯度流动测试", test_gradient_flow), 
        ("内存效率测试", test_memory_efficiency)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 执行失败: {e}")
            results.append((test_name, False))
    
    # 测试结果总结
    print(f"\n{'='*50}")
    print("🎯 测试结果总结:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print(f"\n🏆 总体结果: {'✅ 所有测试通过!' if all_passed else '❌ 存在测试失败!'}")
    
    if all_passed:
        print("\n🚀 网络架构验证完成，可以安全开始训练!")
    else:
        print("\n⚠️  请修复失败的测试项再开始训练!")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)