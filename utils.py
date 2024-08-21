import subprocess
import os
import re
import shutil
import cv2
import numpy as np
import dashscope
import warnings
import glob

from http import HTTPStatus
from PIL import Image
from transformers import BlipProcessor,BlipForConditionalGeneration
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from moviepy.editor import VideoFileClip, AudioFileClip, AudioClip, concatenate_audioclips


warnings.filterwarnings('ignore')

# 路径设置
root_dict = os.getcwd()
keyframe_output_dir = os.path.join(root_dict, 'Video', 'img')
Keyframe_output_filename = os.path.join(root_dict, 'video', 'img', 'output_%06d.jpg')
os.makedirs(keyframe_output_dir, exist_ok=True)


# 预编码器 与 预训练模型
model_url = os.path.join(root_dict, 'models', 'Image2text')
processor = BlipProcessor.from_pretrained(model_url)
model = BlipForConditionalGeneration.from_pretrained(model_url)



# 1、查看视频 frames & fps
def Video_fps_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_round = np.round(fps, 0)
    return [frame_count, fps_round]
    

# 2、提取视频帧
def Video_to_images(import_url,export_url):
    # 若文件夹不存在，就新建一个
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 保存当前帧为图片
        frame_filename = os.path.join(output_dir, f'frame_{frame_index:04d}.png')
        cv2.imwrite(frame_filename, frame)

        print(f'Saved {frame_filename}')

        frame_index += 1

    cap.release()
    print("Done extracting frames.")


# 3、基于图片差异阈值，提取关键帧，并导出时间戳
def extract_keyframes_with_timestamps(input_file, output_pattern, diff=0.8):
    command = [
        "ffmpeg",
        "-i", input_file,                                          # 指定输入
        "-vf", "select='eq(pict_type\\,I)*gt(scene\\,%.3f)',showinfo" % diff, # 选择I帧并基于场景变化，diff为关键帧判定阈值
        "-vsync", "vfr",                                           # vfr -- 提取的帧，与原视频的帧变化一致
        output_pattern                                             # 指定输出（%06d 表示占数字6位数，不足则补0）
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # 解析输出以提取时间戳
    timestamps = []
    for line in result.stderr.split('\n'):
        match = re.search(r'pts_time:(\d+\.\d+)', line)
        if match:
            timestamps.append(float(match.group(1)))

    return timestamps



# 4、基于始末时间提取音频
def exact_audio_timestemp(input_file,start_time,end_time,output_file):
    # 构建 FFmpeg 命令
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-vn',  # 禁用视频
        '-acodec', 'pcm_s16le',  # 使用pcm_s16le编码器
        '-ar','16000',           # 音频采样率为44100Hz
        '-ac', '1',  # 设置音频质量，范围0-9，2是中等质量
        output_file ]
    # 使用 subprocess 运行 FFmpeg 命令
    subprocess.run(ffmpeg_cmd, check=True)


# 5、语音识别
def audio_to_text(audio_file):
    root_dict = os.getcwd()
    model_dict = os.path.join(root_dict, 'models', 'audio2txt')
    model = AutoModel(model=model_dict, model_revision="v2.0.4")

    # 读取音频文件
    speech, sample_rate = soundfile.read(audio_file)

    # 模型处理
    res = model.generate(input=speech, 
                         cache=None, 
                         is_final=True, 
                         chunk_size=None, 
                         encoder_chunk_look_back=None, 
                         decoder_chunk_look_back=None)
    return res


# 6、提取所有视频帧
def extract_all_frames(input_file, output_pattern):
    command = [
        "ffmpeg",
        "-i", input_file,
        "-vf", "select='not(mod(n,1))'",  # 提取所有帧
        "-vsync", "vfr",
        output_pattern
    ]
    subprocess.run(command, capture_output=True, text=True)
    

# 7、各关键帧打包至{interval_i}中【核心逻辑】  
def split_frames_into_intervals(input_file, timestamps, temp_dir, output_base_dir, fps):
    frame_pattern = os.path.join(temp_dir, 'frame_%06d.jpg')
    
    # 提取所有帧
    extract_all_frames(input_file, frame_pattern)
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_') and f.endswith('.jpg')])

    for i in range(len(timestamps)):
        # 为各关键帧创建文件夹
        interval_dir = os.path.join(output_base_dir, f'interval_{i}')
        os.makedirs(interval_dir, exist_ok=True)
        
        # 确定该间隔的帧范围
        start_time = timestamps[i]
        end_time = timestamps[i + 1] if i + 1 < len(timestamps) else None
        
        # 命名从1开始
        frame_counter = 1 
        
        for frame_file in frame_files:
            frame_index = int(re.search(r'frame_(\d+)', frame_file).group(1))
            frame_time = frame_index / fps  
            
            # 调试输出：打印当前帧的时间戳和对应的区间
            print(f"Frame index: {frame_index}, Frame time: {frame_time}, Interval: [{start_time}, {end_time})")
            
            # 基于时间阈值，确定关键帧范围内的所有图片
            if (end_time is None and frame_time >= start_time) or (start_time <= frame_time < end_time):
                new_frame_file = f'frame_{frame_counter:06d}.jpg'
                shutil.move(os.path.join(temp_dir, frame_file), os.path.join(interval_dir, new_frame_file))
                frame_counter += 1

                
# 8、基于MSE来量化图片差异
def calculate_image_difference(image1_path, image2_path):
    image1 = Image.open(image1_path).convert('RGB')
    image2 = Image.open(image2_path).convert('RGB')
    
    image1 = image1.resize((256, 256))
    image2 = image2.resize((256, 256))
    
    np_image1 = np.array(image1)
    np_image2 = np.array(image2)

    err = np.sum((np_image1.astype("float") - np_image2.astype("float")) ** 2)
    err /= float(np_image1.shape[0] * np_image1.shape[1])
    
    return err


# 9、图片解析
def image_to_text(image_url,text_express=''):
    raw_image = Image.open(image_url).convert('RGB')
    inputs = processor(raw_image, text_express, return_tensors="pt")
    out = model.generate(**inputs)
    conditional_out = processor.decode(out[0], skip_special_tokens=True)

    return conditional_out


# 10、【优化】图片"差异度" > 10000时，再进行解析
def extract_text_from_images_with_difference(directory, threshold=100000):
    combined_text = ""
    image_files = [f for f in os.listdir(directory)]

    previous_image = None
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(directory, image_file)
        
        if i == 0:  # 默认解析第一张图片
            print(image_file)
            text = image_to_text(image_path)
            combined_text += text
            previous_image = image_path
        else:
            difference = calculate_image_difference(previous_image, image_path)
            if difference > threshold:
                print(image_file)   # 打印解析图片名
                text = image_to_text(image_path)
                combined_text += text 
                previous_image = image_path

    return combined_text


# 11、文本转解说词
def text_to_commentary(Question):
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': Question}]

    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=messages,
        result_format='message',)  # 将返回结果格式设置为 message
    
    answer = response.output.choices[0].message.content
    return answer


# 12、视频帧转视频
def images_to_video(image_folder, output_video, frame_rate=30):
    # 构建FFmpeg命令
    command = [
        'ffmpeg',
        '-framerate', str(frame_rate),             # 帧率
        '-i', os.path.join(image_folder, 'frame_%06d.jpg'),  # 输入图片路径格式
        '-c:v', 'libx264',                         # 编码器
        '-pix_fmt', 'yuv420p',                     # 像素格式
        output_video                               # 输出视频文件
    ]
    
    # 运行FFmpeg命令
    subprocess.run(command)


# 13、摘要
def Summarization(text):
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

    # 摘要
    article_text = combined_text

    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(
        [WHITESPACE_HANDLER(article_text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4,
        length_penalty=1.0  # 增加生成内容的长度惩罚，生成更长的内容

    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return summary



# 14、LLM功能
def text_to_commentary(Question):
    dashscope.api_key="sk-8ee136df3d1c461bb85c60a31418d43a"
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': Question}]

    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=messages,
        result_format='message',)  # 将返回结果格式设置为 message
    
    answer = response.output.choices[0].message.content
    return answer
# answer = text_to_commentary(Question).replace('\n','')


# 15、语音合成
def text_to_audio(text,model_id,output_audio_file):
    sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech, model=model_id,cache_dir='/AI/cacheOfmodelscope')
    output = sambert_hifigan_tts(input=text, voice='zhitian_emo')
    wav = output[OutputKeys.OUTPUT_WAV]
    with open(output_audio_file, 'wb') as f:
        f.write(wav)


# 16、视频音频合成
def audio_video_concat(video_path,audio_path,output_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    # 确保音频长度与视频一致
    if audio_clip.duration < video_clip.duration:
        # 计算需要填充的静音时长，创建静音音频剪辑，并与原音频合并
        silence_duration = video_clip.duration - audio_clip.duration
        silence = AudioClip(lambda t: 0, duration=silence_duration)
        final_audio = concatenate_audioclips([audio_clip, silence])
    else:
        # 如果音频长于或等于视频长度，只截取音频的前部分
        final_audio = audio_clip.subclip(0, video_clip.duration)

    # 合并
    final_clip = video_clip.set_audio(final_audio)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # 释放资源
    final_clip.close()
    video_clip.close()
    final_audio.close()
    
# 17、指定文件夹，合并其中所有视频文件
def create_file_list(folder_path, extension='mp4'):
    file_list = [f"file '{os.path.join(folder_path, file)}'\n"
                 for file in os.listdir(folder_path)
                 if file.endswith(extension)]
    with open('filelist.txt', 'w') as f:
        f.writelines(file_list)
    
def concatenate_videos(output_file):
    command = [
        'ffmpeg', 
        '-f', 'concat', 
        '-safe', '0', 
        '-i', 'filelist.txt', 
        '-c', 'copy', output_file]
    subprocess.run(command)
    
    
# 18、清理产生的中间音频、图片、视频
def file_folder(directory_path):
    if os.path.exists(directory_path):
        files = glob.glob(os.path.join(directory_path, '*'))
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
                print(f"已删除文件：{file}")
            else:
                print(f"跳过目录：{file}")
        print(f"{directory_path} 下的所有文件已被删除。")
    else:
        print(f"路径 {directory_path} 不存在。")



# 19、AI视频剪辑
def AI_Video_Clip(video_path,output_file):
    # 1、帧提取并分类存储
    fps = Video_fps_frames(video_path)[1]

    # 运行提取函数并处理帧
    timestamps = extract_keyframes_with_timestamps(video_path, os.path.join(keyframe_output_dir, 'output_%06d.jpg'))

    # 临时目录存储所有帧，至各自的路径
    temp_dir = os.path.join(root_dict, 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    split_frames_into_intervals(video_path, timestamps, temp_dir, keyframe_output_dir, fps)
    shutil.rmtree(temp_dir)

    print("完成帧提取并分类存储。")



    # 2、合并文字信息 & 重合成视频
    interval_num = len(timestamps) 
    interval_combined_text_list = []            
    interval_combined_video_time_list = []

    for num in range(interval_num):
        image_folder = os.path.join(os.getcwd(), 'Video', 'img','interval_%d'%num)
        output_video = os.path.join(os.getcwd(), 'Video', 'key_frame_video','interval_%d_output.mp4'%num)
        #if os.path.exists(output_video):
        combined_text = extract_text_from_images_with_difference(image_folder)     # 提取并合并文字信息
        images_to_video(image_folder, output_video,fps)                            # 提取并重合成视频

        frames,fps_out = Video_fps_frames(output_video)                            # 提取视频时长，用于后续语音&视频合成
        if fps_out==0:
            fps_out = 30
            output_video_time =  round(frames/fps_out)
        else:
            output_video_time =  round(frames/fps_out)

        interval_combined_text_list.append(combined_text)                           # 解析文本列表
        interval_combined_video_time_list.append(output_video_time)                 # 视频时长列表


    # 3、解说词生成
    commentary_list = []
    for combined_text,time in zip(interval_combined_text_list,interval_combined_video_time_list):
        combined_text = '(' + combined_text + ')'
        prompt = '将上述内容转换成中文视频解说词，要求语言精炼、有逻辑、以第三人称输出解说词部分、不要加入"我们"的元素，仅描述一段内容即可，解说词的内容需要远小于%s内，说完注意控制字数'%time
        Question = combined_text + prompt
        commentary = text_to_commentary(Question).replace('\n','')
        
        commentary_list.append(commentary)


    # 4、解说词_语音生成
    # model_id = 'damo/speech_sambert-hifigan_tts_zh-cn_16k'
    model_id = os.path.join(os.getcwd(), 'models', 'text2audio')

    # 根据，批量生成解说词.wav文件
    for id_,commentary in enumerate(commentary_list):
        audio_folder = os.path.join(os.getcwd(), 'Video', 'commentary_audio','interval_%s.wav'%id_) 
        text_to_audio(commentary,model_id,audio_folder)


    # 5、合并所有关键帧下的视频
    video_name_list = os.listdir(os.path.join(os.getcwd(), 'Video', 'key_frame_video'))
    audio_name_list = os.listdir(os.path.join(os.getcwd(), 'Video', 'commentary_audio'))


    # 6、生成所有关键帧，对应带有语音的子视频
    for i,j in zip(video_name_list,audio_name_list):
        video_name = os.path.join(os.getcwd(), 'Video', 'key_frame_video',i) 
        audio_name = os.path.join(os.getcwd(), 'Video', 'commentary_audio',j)
        output_path = os.path.join(os.getcwd(), 'Video', 'concat_video_audio','con_%s'%i) 
        audio_video_concat(video_name,audio_name,output_path)


    # 7、将concat_video_audio路径下，所有子视频串联成一个完整视频
    folder_path = os.path.join(os.getcwd(), 'Video', 'concat_video_audio')
    # folder_path = os.path.join(os.getcwd(), 'Video')
    # output_file = os.path.join(os.getcwd(), 'Video','output.mp4')

    create_file_list(folder_path)
    concatenate_videos(output_file)


    # 8、每次清空临时文件
    clear_folder = ['commentary_audio','concat_video_audio','img','key_frame_video']
    for i in clear_folder:
        directory_path = os.path.join(os.getcwd(), 'Video',i)
        
        # 开始清空
        file_folder(directory_path)