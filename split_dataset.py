import os
import shutil
import random

# ==================== 🛠️ 路径与参数配置 ====================
# 1. 你的源数据文件夹（堆满 .h5 的地方）
SRC_DIR = r"E:\\Projects\\Mamba\\WSL4MIS-main\\data\\ACDC\\ACDC_training_slices"

# 2. 目标根目录（新项目 mamba_skip 的路径）
DEST_BASE = r"E:\\Projects\\Mamba\\mamba_skip\\data\\acdc"

# 3. 划分比例（推荐 0.8 或 0.85，根据需要自行修改）
TRAIN_RATIO = 0.85  # 80% 用于训练，剩下的 20% 用于验证
# ==========================================================

def secure_split():
    # 确定具体的 train 和 val 目标路径
    train_dir = os.path.join(DEST_BASE, "train")
    val_dir = os.path.join(DEST_BASE, "val")
    
    # 创建目标文件夹
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 检查源目录
    if not os.path.exists(SRC_DIR):
        print(f"❌ 错误：找不到源文件夹，请确认路径是否正确：\n   {SRC_DIR}")
        return

    # 获取所有 .h5 切片文件
    all_files = [f for f in os.listdir(SRC_DIR) if f.endswith('.h5')]
    if not all_files:
        print(f"❌ 错误：在源目录中没有找到任何 .h5 文件！")
        return

    # --- 核心逻辑：按 Patient ID 聚类，防止数据泄露 ---
    patients = set()
    for f in all_files:
        # 假设文件名格式为 "patient001_frame01_slice_0.h5"
        patient_id = f.split('_')[0]
        patients.add(patient_id)
    
    patients = sorted(list(patients))
    total_patients = len(patients)
    print(f"🔍 扫描完毕：共发现 {total_patients} 个独立病人编号。")

    # 固定随机种子，确保实验可复现
    random.seed(42)
    random.shuffle(patients)

    # 计算切分边界
    split_idx = int(total_patients * TRAIN_RATIO)
    train_patients = set(patients[:split_idx])
    val_patients = set(patients[split_idx:])

    print(f"📊 划分方案：")
    print(f"   - 训练集 (Train): {len(train_patients)} 个病人 (占比 {TRAIN_RATIO*100:.1f}%)")
    print(f"   - 验证集 (Val):   {len(val_patients)} 个病人 (占比 {(1-TRAIN_RATIO)*100:.1f}%)")
    print("-" * 50)

    # --- 开始物理复制 ---
    train_slices_count = 0
    val_slices_count = 0

    print("🚀 正在跨项目复制切片文件，请稍候...")
    for f in all_files:
        patient_id = f.split('_')[0]
        src_file_path = os.path.join(SRC_DIR, f)
        
        if patient_id in train_patients:
            shutil.copy(src_file_path, os.path.join(train_dir, f))
            train_slices_count += 1
        elif patient_id in val_patients:
            shutil.copy(src_file_path, os.path.join(val_dir, f))
            val_slices_count += 1

    print("-" * 50)
    print("🎉 数据集安全划分并迁移成功！")
    print(f"📁 训练集路径: {train_dir} (总计 {train_slices_count} 个 .h5 切片)")
    print(f"📁 验证集路径: {val_dir} (总计 {val_slices_count} 个 .h5 切片)")

if __name__ == "__main__":
    secure_split()