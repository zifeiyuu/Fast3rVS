import pandas as pd

# 示例：加载某个帧的关联文件
frame_id = "14183710428479823719_3140_000_3160_000"  # 根据实际文件名修改
assoc_file = f"/high_perf_store/l3_deep/open-datasets/waymo/v2.0.1/training/camera_to_lidar_box_association/{frame_id}.parquet"
assoc_df = pd.read_parquet(assoc_file)

print(assoc_df.head()) 