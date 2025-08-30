from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

# 视频文件夹路径
video_folder = "/high_perf_store/l3_deep/xiaoziyu/FreeVS/diffusers/eval_output_newtraj/eval/psuedo"  # 替换为你的视频文件夹路径
output_path = "output.mp4"  # 输出文件路径

# 获取文件夹中的所有视频文件，并按文件名排序
video_files = sorted([os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".mp4")])

# 加载视频
clips = [VideoFileClip(video) for video in video_files]

# 拼接视频
final_clip = concatenate_videoclips(clips, method="compose")

# 调整色调（如果视频偏蓝）
def adjust_tone(clip):
    # 定义一个函数来处理每一帧图像
    def adjust_frame(frame):
        # 交换红色和蓝色通道
        return frame[:, :, [1, 2, 0]]
    
    # 对每一帧应用调整
    return clip.fl_image(adjust_frame)

# 应用色调调整
# final_clip = adjust_tone(final_clip)

# 导出视频
final_clip.write_videofile(output_path, codec="libx264", fps=24)

print(f"视频已成功导出到: {output_path}")