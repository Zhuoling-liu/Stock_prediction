import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_system_diagram():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # --- 样式设置 ---
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#e1f5fe', edgecolor='#01579b', linewidth=2)
    model_props = dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=2)
    innovation_props = dict(boxstyle='round,pad=0.5', facecolor='#fff3e0', edgecolor='#ef6c00', linewidth=3) # 橙色强调创新
    optimize_props = dict(boxstyle='round,pad=0.5', facecolor='#fce4ec', edgecolor='#c2185b', linewidth=3) # 红色强调结果

    # --- 1. 数据输入层 ---
    ax.text(1, 3, "Market Data\n(Price, Vol)", ha='center', va='center', fontsize=12, bbox=box_props)
    
    # 箭头 1 -> 2
    ax.annotate('', xy=(2.5, 3), xytext=(1.8, 3), arrowprops=dict(arrowstyle='->', lw=2))

    # --- 2. 特征工程层 ---
    ax.text(3.5, 3, "Feature\nEngineering\n(RSI, Mom, Vol)", ha='center', va='center', fontsize=12, bbox=box_props)

    # 分流箭头 (Dual Stream)
    ax.annotate('', xy=(5, 4.5), xytext=(4.3, 3.2), arrowprops=dict(arrowstyle='->', lw=2, connectionstyle="arc3,rad=0.2"))
    ax.annotate('', xy=(5, 1.5), xytext=(4.3, 2.8), arrowprops=dict(arrowstyle='->', lw=2, connectionstyle="arc3,rad=-0.2"))

    # --- 3. 模型层 (The Hybrid Brain) ---
    # Path A: Linear
    ax.text(6, 4.5, "Linear Stream\n(Ridge Regression)\n[Trend Signal]", ha='center', va='center', fontsize=11, bbox=model_props)
    
    # Path B: Non-Linear
    ax.text(6, 1.5, "Non-Linear Stream\n(Gradient Boosting)\n[Complex Pattern]", ha='center', va='center', fontsize=11, bbox=model_props)

    # --- 4. 核心创新 (Risk Monitor) ---
    # 这个放在中间，连接特征和优化器，作为并行的监控模块
    ax.text(6, 3, "⚠️ Risk Monitor\n(Rolling Validation)\n[Gap=20d]", ha='center', va='center', fontsize=11, fontweight='bold', bbox=innovation_props)
    # 输入给 Risk Monitor
    ax.annotate('', xy=(5.2, 3), xytext=(4.3, 3), arrowprops=dict(arrowstyle='->', lw=1.5, ls='--')) # 虚线表示监控流

    # --- 5. 信号汇聚 ---
    # 箭头汇聚到 Optimizer
    ax.annotate('', xy=(8, 3.2), xytext=(6.9, 4.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(8, 2.8), xytext=(6.9, 1.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(8, 3), xytext=(6.9, 3), arrowprops=dict(arrowstyle='->', lw=2, color='#ef6c00')) # 橙色箭头

    # 信号标签 (数学符号显专业)
    ax.text(7.5, 4, r"$\mu$ (Exp. Return)", fontsize=10, color='green')
    ax.text(7.2, 3.2, r"$\sigma_{error}$ (Uncertainty)", fontsize=10, color='#ef6c00', fontweight='bold')

    # --- 6. 决策层 (Optimizer) ---
    ax.text(9, 3, "MVF Optimizer\n(NSGA-II Logic)\n[Max Utility]", ha='center', va='center', fontsize=12, fontweight='bold', bbox=optimize_props)

    # --- 7. 输出 ---
    ax.annotate('', xy=(10, 3), xytext=(9.8, 3), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(10, 2.5, "Dynamic\nPortfolio\nWeights", ha='center', va='top', fontsize=11)

    # --- 标题 ---
    plt.title("System Architecture: Dual-Stream Hybrid Model with Uncertainty Penalization", fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()

# 运行绘图
draw_system_diagram()