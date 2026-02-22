"""
乒乓球智能计分识别系统
结合姿态识别和语音识别技术，实现自动化得分统计、实时比分显示及发球方提示功能
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import logging
from datetime import datetime
import sys
import os

# 设置系统编码为UTF-8
if sys.platform.startswith('win'):
    # Windows平台设置控制台编码
    os.system('chcp 65001')

# 尝试导入语音识别模块，如果失败则标记为不可用
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    print("警告: 未找到speech_recognition或pyaudio模块，语音识别功能不可用".encode('utf-8').decode('utf-8'))
    SPEECH_RECOGNITION_AVAILABLE = False
except AttributeError as e:
    if "Could not find PyAudio" in str(e):
        print("警告: 未找到PyAudio，语音识别功能不可用".encode('utf-8').decode('utf-8'))
        SPEECH_RECOGNITION_AVAILABLE = False
    else:
        raise e

from config import *


class TableTennisScorer:
    def __init__(self):
        # 初始化变量
        self.score_a = 0
        self.score_b = 0
        self.game_active = True
        self.last_score_time = 0
        self.cooldown_period = COOLDOWN_PERIOD
        self.serve_side = 'A'  # 当前发球方
        self.score_source = ''  # 得分来源：'pose' 或 'voice'
        self.pose_detected_recently = False
        self.pose_detection_start_time = 0
        self.pose_detection_threshold = 1.0  # 需要持续检测1秒才确认得分
        
        # 设置日志 - 使用FileHandler显式设置编码
        if LOG_TO_FILE:
            import logging.handlers
            handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
        else:
            # 如果不需要记录到文件，则只使用控制台输出
            logging.basicConfig(level=logging.INFO)
        
        # 初始化MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE
        )
        
        # 初始化语音识别（如果可用）
        self.voice_recognition_available = SPEECH_RECOGNITION_AVAILABLE
        if self.voice_recognition_available:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = VOICE_ENERGY_THRESHOLD
                self.microphone = sr.Microphone()
                
                # 设置语音关键词
                self.score_keywords = [keyword.lower() for keyword in VOICE_KEYWORDS]
                
                # 启动语音识别线程
                self.voice_thread = threading.Thread(target=self.listen_for_voice_commands)
                self.voice_thread.daemon = True
                self.voice_thread.start()
                
                print("语音识别功能已启用".encode('utf-8').decode('utf-8'))
            except Exception as e:
                print(f"语音识别初始化失败: {str(e).encode('utf-8').decode('utf-8')}")
                self.voice_recognition_available = False
        else:
            self.voice_recognition_available = False
            print("语音识别功能不可用".encode('utf-8').decode('utf-8'))

    def start_listening(self):
        """开始监听语音命令"""
        if not self.voice_recognition_available:
            return
            
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def listen_for_voice_commands(self):
        """持续监听语音命令的后台线程"""
        if not self.voice_recognition_available:
            return
            
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        while True:
            try:
                audio = self.recognizer.listen(self.microphone, timeout=1, phrase_time_limit=VOICE_PHRASE_TIME_LIMIT)
                try:
                    # 使用Google语音识别，支持中文
                    text = self.recognizer.recognize_google(audio, language='zh-CN')
                    print(f"识别到语音: {text}".encode('utf-8').decode('utf-8'))
                    
                    if DEBUG_MODE:
                        print(f"语音文本: {text}".encode('utf-8').decode('utf-8'))
                    
                    # 检查是否包含得分关键词（转换为小写进行比较）
                    text_lower = text.lower()
                    for keyword in self.score_keywords:
                        if keyword in text_lower:
                            self.process_score('voice')
                            break
                except sr.UnknownValueError:
                    # 无法识别的语音，继续监听
                    continue
                except sr.RequestError as e:
                    print(f"语音识别服务错误: {str(e).encode('utf-8').decode('utf-8')}")
                    if LOG_TO_FILE:
                        logging.error(f"语音识别服务错误: {e}")
                    continue
            except sr.WaitTimeoutError:
                # 没有检测到语音，继续监听
                continue
    
    def detect_touching_table_pose(self, landmarks):
        """
        检测"摸球桌"动作
        这里我们检测食指尖是否靠近手掌区域，模拟"触摸桌面"的动作
        """
        # 获取关键点
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # 计算食指尖到手腕的距离
        index_to_wrist_dist = np.sqrt((index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2)
        
        # 计算中指尖到手腕的距离
        middle_to_wrist_dist = np.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)
        
        # 检查是否手指靠近手腕（模拟触摸动作）
        touching_threshold = 0.2
        is_touching = (index_to_wrist_dist < touching_threshold and 
                       middle_to_wrist_dist < touching_threshold)
        
        return is_touching
    
    def detect_scissors_pose(self, landmarks):
        """
        检测剪刀手（胜利）手势
        食指和中指伸直，其他手指弯曲
        """
        # 获取关键点
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = landmarks[self.mp_hands.HandLandmark.PINKY_MCP]
        
        # 检查食指和中指是否伸直（指尖高于指关节）
        index_extended = index_tip.y < index_mcp.y
        middle_extended = middle_tip.y < middle_mcp.y
        
        # 检查无名指和小指是否弯曲（指尖低于指关节）
        ring_curled = ring_tip.y > ring_mcp.y
        pinky_curled = pinky_tip.y > pinky_mcp.y
        
        # 检查拇指是否弯曲或位置合适
        thumb_position = abs(thumb_tip.x - index_mcp.x) > 0.1  # 拇指不要太靠近食指
        
        return (index_extended and middle_extended and 
                ring_curled and pinky_curled and thumb_position)
    
    def detect_pose(self, image):
        """检测姿态，识别特定得分动作"""
        # 将BGR图像转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 检测是否是得分手势
                is_touching = self.detect_touching_table_pose(hand_landmarks.landmark)
                is_scissors = self.detect_scissors_pose(hand_landmarks.landmark)
                
                if is_touching or is_scissors:
                    # 获取手的边界框中心来判断是哪一侧
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    
                    center_x = (min(x_coords) + max(x_coords)) / 2
                    
                    if center_x < 0.5:
                        return 'A', True  # 左侧选手得分
                    else:
                        return 'B', True  # 右侧选手得分
        
        return None, False
    
    def process_score(self, source):
        """处理得分事件"""
        current_time = time.time()
        
        # 检查是否在冷却期内
        if current_time - self.last_score_time < self.cooldown_period:
            if DEBUG_MODE:
                print("仍在冷却期内，忽略本次得分".encode('utf-8').decode('utf-8'))
            return False
        
        # 更新分数
        if self.serve_side == 'A':
            self.score_a += 1
        else:
            self.score_b += 1
            
        self.score_source = source
        self.last_score_time = current_time
        
        # 更新发球方（每SERVE_CHANGE_INTERVAL分换发）
        total_score = self.score_a + self.score_b
        if total_score % SERVE_CHANGE_INTERVAL == 0:
            self.serve_side = 'B' if self.serve_side == 'A' else 'A'
        
        log_msg = f"得分！选手{'A' if self.score_a > self.score_b else 'B'}得分，来源：{source}，当前比分 A:{self.score_a} - B:{self.score_b}"
        print(log_msg.encode('utf-8').decode('utf-8'))  # 确保正确显示中文
        
        if LOG_TO_FILE:
            logging.info(log_msg)
        
        # 检查是否达到胜利条件（WINNING_SCORE分且领先MINIMUM_WINNING_DIFFERENCE分）
        score_diff = abs(self.score_a - self.score_b)
        winning_score_reached = self.score_a >= WINNING_SCORE or self.score_b >= WINNING_SCORE
        winning_diff_reached = score_diff >= MINIMUM_WINNING_DIFFERENCE
        
        if winning_score_reached and winning_diff_reached:
            self.game_active = False
            winner = 'A' if self.score_a > self.score_b else 'B'
            log_msg = f"比赛结束！{winner}选手获胜！最终比分 A:{self.score_a} - B:{self.score_b}"
            print(log_msg.encode('utf-8').decode('utf-8'))  # 确保正确显示中文
            
            if LOG_TO_FILE:
                logging.info(log_msg)
        
        return True
    
    def draw_ui(self, frame):
        """绘制用户界面"""
        h, w, c = frame.shape
        
        # 绘制中央分割线
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        
        # 绘制左右区域标识
        cv2.putText(frame, 'A区', (w//4, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        cv2.putText(frame, 'B区', (w//2 + w//4, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        # 显示比分
        cv2.putText(frame, f'A: {self.score_a}', (w//4, h//4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, SCORE_COLOR, 3)
        cv2.putText(frame, f'B: {self.score_b}', (w//2 + w//4, h//4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, SCORE_COLOR, 3)
        
        # 显示发球方指示
        serve_text = f'发球方: {self.serve_side}'
        text_size = cv2.getTextSize(serve_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, serve_text, (text_x, h//4 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, SERVE_ARROW_COLOR, FONT_THICKNESS)
        
        # 显示得分来源
        if self.score_source:
            source_text = f'上次得分来源: {self.score_source}'
            cv2.putText(frame, source_text, (w//2 - 120, h//4 + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8, (255, 255, 0), FONT_THICKNESS)
        
        # 显示语音识别状态
        voice_status = "语音: 已启用" if self.voice_recognition_available else "语音: 不可用"
        cv2.putText(frame, voice_status, (w//2 - 80, h//4 + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8, 
                   SCORE_COLOR if self.voice_recognition_available else ERROR_COLOR, FONT_THICKNESS)
        
        # 显示冷却时间
        remaining_time = max(0, self.cooldown_period - int(time.time() - self.last_score_time))
        if remaining_time > 0:
            cooldown_text = f'冷却中: {remaining_time}s'
            cv2.putText(frame, cooldown_text, (w//2 - 80, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, ERROR_COLOR, FONT_THICKNESS)
        
        # 显示操作说明
        cv2.putText(frame, "按键: 'q'-退出 | 'r'-重置 | 's'-切换手势类型", (20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.6, (200, 200, 200), 1)
        
        # 显示系统说明
        gesture_desc = "手势: 剪刀手 或 触摸动作"
        voice_desc = "语音: 说 '得分/好球/goal'" if self.voice_recognition_available else "语音: 不可用"
        cv2.putText(frame, gesture_desc, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.6, (200, 200, 200), 1)
        cv2.putText(frame, voice_desc, (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.6, (200, 200, 200), 1)
        
        # 显示比赛状态
        if not self.game_active:
            winner = 'A' if self.score_a > self.score_b else 'B'
            win_text = f"{winner}选手获胜!"
            text_size = cv2.getTextSize(win_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, win_text, (text_x, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, ERROR_COLOR, 3)
        
        return frame
    
    def reset_game(self):
        """重置游戏"""
        self.score_a = 0
        self.score_b = 0
        self.game_active = True
        self.last_score_time = 0
        self.serve_side = 'A'
        self.score_source = ''
        print("游戏已重置".encode('utf-8').decode('utf-8'))
        
        if LOG_TO_FILE:
            logging.info("游戏已重置")
    
    def run(self):
        """运行主循环"""
        print("启动乒乓球智能计分系统...".encode('utf-8').decode('utf-8'))
        print(f"语音识别: {'已启用' if self.voice_recognition_available else '不可用'}".encode('utf-8').decode('utf-8'))
        if self.voice_recognition_available:
            print(f"请使用'{VOICE_KEYWORDS}'等关键词或做出得分手势来得分".encode('utf-8').decode('utf-8'))
        else:
            print("请使用手势来得分".encode('utf-8').decode('utf-8'))
        print("按 'q' 退出，按 'r' 重置比分".encode('utf-8').decode('utf-8'))
        
        # 打开摄像头
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        while cap.isOpened() and self.game_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 水平翻转图像（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 检测姿态
            side, pose_detected = self.detect_pose(frame)
            if pose_detected:
                # 如果检测到姿态，处理得分
                self.process_score('pose')
            
            # 绘制UI
            frame = self.draw_ui(frame)
            
            # 显示帧
            cv2.imshow(WINDOW_NAME, frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_game()
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    scorer = TableTennisScorer()
    scorer.run()