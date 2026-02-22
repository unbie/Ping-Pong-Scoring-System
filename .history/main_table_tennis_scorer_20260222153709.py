"""
Ping Pong Smart Scoring System
Combining pose recognition and voice recognition technologies to achieve automated scoring statistics, 
real-time score display, and serving side indication
"""
# encoding: utf-8
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

# 添加语音播报功能
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("Warning: pyttsx3 module not found, text-to-speech feature unavailable")
    TTS_AVAILABLE = False

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
        
        # 添加总局比分追踪
        self.total_games_a = 0
        self.total_games_b = 0
        
        # 初始化语音播报引擎
        self.tts_available = TTS_AVAILABLE
        if self.tts_available:
            try:
                self.tts_engine = pyttsx3.init()
                # 设置语速
                self.tts_engine.setProperty('rate', 150)
                # 设置音量
                self.tts_engine.setProperty('volume', 0.9)
            except:
                self.tts_available = False
                print("Failed to initialize text-to-speech engine")
        
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
        
        # 移除语音识别部分
        # 保留语音播报功能
        self.voice_recognition_available = False
        print("Voice recognition disabled, text-to-speech enabled")

    def start_listening(self):
        """语音识别功能已移除"""
        pass
    
    # 移除 listen_for_voice_commands 方法，因为它属于语音识别功能
    # 保留语音播报功能
    
    def speak_score(self, player, current_score_a, current_score_b):
        """播报得分"""
        if not self.tts_available:
            print("Text-to-speech not available")
            return
            
        # 创建播报文本
        text = f"Player {player} scores. Current score: A {current_score_a}, B {current_score_b}"
        print(f"TTS: {text}")  # 添加调试信息
        
        try:
            # 每次都创建新的TTS引擎实例，避免状态问题
            temp_tts_engine = pyttsx3.init()
            
            # 设置语速和音量
            temp_tts_engine.setProperty('rate', 150)
            temp_tts_engine.setProperty('volume', 0.9)
            
            # 执行语音播报
            temp_tts_engine.say(text)
            # 同步等待播报完成
            temp_tts_engine.runAndWait()
            
            # 清理引擎实例
            del temp_tts_engine
        except Exception as e:
            print(f"Text-to-speech playback failed: {e}")

    # 移除备用音频播放方法
    
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
    
    def detect_pose(self, image):
        """检测姿态，识别特定得分动作"""
        # 将BGR图像转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 只检测触摸动作，去除剪刀手势
                is_touching = self.detect_touching_table_pose(hand_landmarks.landmark)
                
                if is_touching:
                    # 获取手的边界框中心来判断是哪一侧
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    
                    center_x = (min(x_coords) + max(x_coords)) / 2
                    
                    if center_x < 0.5:
                        return 'A', True  # 左侧选手得分
                    else:
                        return 'B', True  # 右侧选手得分
        
        return None, False
    
    def process_score(self, source, target_side=None):
        """处理得分事件"""
        current_time = time.time()
        
        # 检查是否在冷却期内
        if current_time - self.last_score_time < self.cooldown_period:
            if DEBUG_MODE:
                print(f"Still in cooldown period, {self.cooldown_period - (current_time - self.last_score_time):.1f}s remaining")
            return False
        
        # 如果指定了目标方，则给指定方加分，否则给当前发球方加分
        if target_side == 'A':
            self.score_a += 1
        elif target_side == 'B':
            self.score_b += 1
        else:
            # 默认给当前发球方加分
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
        
        # 确定得分选手
        scoring_player = 'A' if target_side == 'A' or (target_side is None and self.serve_side == 'A') else 'B'
        
        log_msg = f"Score! Player {scoring_player} scored, Source: {source}, Current Score A:{self.score_a} - B:{self.score_b}"
        print(log_msg)  # 确保正确显示
        
        # 播报得分
        self.speak_score(scoring_player, self.score_a, self.score_b)
        
        if LOG_TO_FILE:
            logging.info(log_msg)
        
        # 检查是否达到胜利条件（WINNING_SCORE分且领先MINIMUM_WINNING_DIFFERENCE分）
        score_diff = abs(self.score_a - self.score_b)
        winning_score_reached = self.score_a >= WINNING_SCORE or self.score_b >= WINNING_SCORE
        winning_diff_reached = score_diff >= MINIMUM_WINNING_DIFFERENCE
        
        if winning_score_reached and winning_diff_reached:
            winner = 'A' if self.score_a > self.score_b else 'B'
            
            # 更新总局比分
            if winner == 'A':
                self.total_games_a += 1
            else:
                self.total_games_b += 1
                
            # 检查总局比分是否达到4局胜利
            match_winner = None
            if self.total_games_a >= 4:
                match_winner = 'A'
                self.game_active = False  # 整场比赛结束
                print(f"Match Over! Player {match_winner} wins the match! Final Score: A:{self.total_games_a} - B:{self.total_games_b}")
                if LOG_TO_FILE:
                    logging.info(f"Match Over! Player {match_winner} wins the match! Final Score: A:{self.total_games_a} - B:{self.total_games_b}")
            elif self.total_games_b >= 4:
                match_winner = 'B'
                self.game_active = False  # 整场比赛结束
                print(f"Match Over! Player {match_winner} wins the match! Final Score: A:{self.total_games_a} - B:{self.total_games_b}")
                if LOG_TO_FILE:
                    logging.info(f"Match Over! Player {match_winner} wins the match! Final Score: A:{self.total_games_a} - B:{self.total_games_b}")
            else:
                # 仅当前局结束，重置当前局比分，继续比赛
                print(f"Game Over! Player {winner} wins this game! Set Score A:{self.score_a} - B:{self.score_b}, Total Games A:{self.total_games_a} - B:{self.total_games_b}")
                self.score_a = 0  # 重置当前局比分
                self.score_b = 0  # 重置当前局比分
                self.serve_side = 'A'  # 重新开始时A发球
                # 注意：这里不改变game_active状态，继续比赛
                
                # 播报新游戏开始
                if self.tts_available:
                    def speak_new_game():
                        try:
                            self.tts_engine.say(f"New game, starting from zero")
                            self.tts_engine.runAndWait()
                        except:
                            print("Text-to-speech playback failed")
                    
                    speak_thread = threading.Thread(target=speak_new_game)
                    speak_thread.daemon = True
                    speak_thread.start()
                
            log_msg = f"Game Over! Player {winner} wins! Set Score A:{self.score_a} - B:{self.score_b}, Total Games A:{self.total_games_a} - B:{self.total_games_b}"
            print(log_msg)  # 确保正确显示
            
            # 播报比赛结束
            if self.tts_available:
                final_text = f"Game over! Player {winner} wins this game! Match score: A {self.total_games_a}, B {self.total_games_b}"
                if match_winner:
                    final_text = f"Match over! Player {match_winner} wins the match! Final score: A {self.total_games_a}, B {self.total_games_b}"
                
                def speak_final():
                    try:
                        self.tts_engine.say(final_text)
                        self.tts_engine.runAndWait()
                    except:
                        print("Text-to-speech playback failed")
                
                speak_thread = threading.Thread(target=speak_final)
                speak_thread.daemon = True
                speak_thread.start()
            
            if LOG_TO_FILE:
                logging.info(log_msg)
        
        return True
    
    def draw_ui(self, frame):
        """绘制用户界面"""
        h, w, c = frame.shape
        
        # 绘制中央分割线
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        
        # 绘制左右区域标识
        cv2.putText(frame, 'Player A', (w//4, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        cv2.putText(frame, 'Player B', (w//2 + w//4, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
        
        # 显示小局比分（大字体，红色）
        cv2.putText(frame, f'A: {self.score_a}', (w//4, h//4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)  # 更大的字体和线条粗细
        cv2.putText(frame, f'B: {self.score_b}', (w//2 + w//4, h//4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)  # 更大的字体和线条粗细
        
        # 显示总局比分（稍小一点的字体，黄色）
        cv2.putText(frame, f'Total A: {self.total_games_a}', (w//4 - 100, h//4 + 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)  # 黄色，较大字体
        cv2.putText(frame, f'Total B: {self.total_games_b}', (w//2 + w//4 - 100, h//4 + 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)  # 黄色，较大字体
        
        # 显示发球方指示
        serve_text = f'Serving: {self.serve_side}'
        text_size = cv2.getTextSize(serve_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, serve_text, (text_x, h//4 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, SERVE_ARROW_COLOR, FONT_THICKNESS)
        
        # 显示得分来源
        if self.score_source:
            source_text = f'Last Score Source: {self.score_source}'
            cv2.putText(frame, source_text, (w//2 - 120, h//4 + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8, (255, 255, 0), FONT_THICKNESS)
        
        # 显示语音识别状态（现在是语音播报状态）
        tts_status = "TTS: Enabled" if self.tts_available else "TTS: Unavailable"
        cv2.putText(frame, tts_status, (w//2 - 80, h//4 + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.8, 
                   SCORE_COLOR if self.tts_available else ERROR_COLOR, FONT_THICKNESS)
        
        # 显示冷却时间
        remaining_time = max(0, self.cooldown_period - int(time.time() - self.last_score_time))
        if remaining_time > 0:
            cooldown_text = f'Cooldown: {remaining_time}s'
            cv2.putText(frame, cooldown_text, (w//2 - 80, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, ERROR_COLOR, FONT_THICKNESS)
        
        # 显示操作说明
        cv2.putText(frame, "Keys: 'q'-Quit ", (20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.6, (200, 200, 200), 1)
        
        # 显示系统说明
        gesture_desc = "Gesture: Touch Action Only"
        # 移除语音识别说明，改为语音播报说明
        tts_desc = "TTS: Enabled" if self.tts_available else "TTS: Unavailable"
        cv2.putText(frame, gesture_desc, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.6, (200, 200, 200), 1)
        cv2.putText(frame, tts_desc, (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.6, (200, 200, 200), 1)
        
        # 显示比赛状态
        if not self.game_active:
            winner = 'A' if self.total_games_a > self.total_games_b else 'B'
            win_text = f"Player {winner} Wins Match!"
            text_size = cv2.getTextSize(win_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, win_text, (text_x, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, ERROR_COLOR, 3)
        
        return frame
    
    def reset_game(self):
        """重置游戏（保留总局比分）"""
        self.score_a = 0
        self.score_b = 0
        self.game_active = True
        self.last_score_time = 0
        self.serve_side = 'A'
        self.score_source = ''
        print("Current game reset, totals kept")
        
        if LOG_TO_FILE:
            logging.info("Current game reset, totals kept")
    
    def run(self):
        """运行主循环"""
        print("Starting Ping Pong Smart Scoring System...")
        print(f"Text-to-Speech: {'Enabled' if self.tts_available else 'Unavailable'}")
        print(f"Use touch gestures to score")
        print("Press 'q' to quit, press 'r' to reset score")
        
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
                self.process_score('pose', side)
            
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