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
import queue
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
        
        # 得分闪烁动画
        self.last_score_change_time = 0
        self.last_scoring_player = None
        self.score_flash_duration = 1.5  # 闪烁持续秒数
        
        # 初始化语音播报引擎
        self.tts_available = TTS_AVAILABLE
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        self.tts_queue = queue.Queue(maxsize=50)
        self.tts_thread = None
        self.tts_stop_event = threading.Event()
        if self.tts_available:
            try:
                # 在独立线程中初始化并驱动TTS，避免跨线程调用导致后续无声
                self._start_tts_worker()
            except Exception:
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
        self._speak_async(text)

    def _speak_async(self, text):
        """异步播报文本，避免阻塞主线程"""
        if not self.tts_available:
            return
        try:
            self.tts_queue.put_nowait(text)
        except queue.Full:
            # 队列满时丢弃最旧一条，优先播报最新比分
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.put_nowait(text)
            except Exception:
                pass

    def _start_tts_worker(self):
        """启动TTS工作线程（线程内初始化引擎）"""
        if self.tts_thread and self.tts_thread.is_alive():
            return
        self.tts_stop_event.clear()
        self.tts_thread = threading.Thread(target=self._tts_worker_loop, daemon=True)
        self.tts_thread.start()

    def _tts_worker_loop(self):
        """TTS线程主循环：每次播报都新建引擎实例，规避 pyttsx3 runAndWait 内部状态bug"""
        while not self.tts_stop_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break

            engine = None
            try:
                # 每次播报都新建引擎，彻底避免 _inLoop / endLoop 状态残留
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Text-to-speech playback failed: {e}")
            finally:
                # 确保引擎被彻底释放
                if engine is not None:
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    del engine

    def _stop_tts_worker(self):
        """停止TTS工作线程"""
        if not self.tts_available:
            return
        self.tts_stop_event.set()
        try:
            self.tts_queue.put_nowait(None)
        except Exception:
            pass
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=1.0)

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
        
        # 记录得分方，用于界面闪烁动画
        self.last_scoring_player = scoring_player
        self.last_score_change_time = current_time
        
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
                    self._speak_async("New game, starting from zero")
                
            log_msg = f"Game Over! Player {winner} wins! Set Score A:{self.score_a} - B:{self.score_b}, Total Games A:{self.total_games_a} - B:{self.total_games_b}"
            print(log_msg)  # 确保正确显示
            
            # 播报比赛结束
            if self.tts_available:
                final_text = f"Game over! Player {winner} wins this game! Match score: A {self.total_games_a}, B {self.total_games_b}"
                if match_winner:
                    final_text = f"Match over! Player {match_winner} wins the match! Final score: A {self.total_games_a}, B {self.total_games_b}"

                self._speak_async(final_text)
            
            if LOG_TO_FILE:
                logging.info(log_msg)
        
        return True
    
    def _draw_translucent_rect(self, frame, x1, y1, x2, y2, color, alpha=0.6):
        """在 frame 上绘制半透明矩形"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _put_text_centered(self, frame, text, cx, cy, scale, color, thickness):
        """以 (cx, cy) 为中心绘制文字"""
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        tx = cx - text_size[0] // 2
        ty = cy + text_size[1] // 2
        # 文字描边（黑色轮廓），提升对比度
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 4)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def draw_ui(self, frame):
        """绘制美化后的用户界面 —— 超大比分、高对比度、一眼可见"""
        h, w, c = frame.shape
        now = time.time()

        # ── 颜色定义 ──
        COLOR_A = (200, 120, 50)     # 选手A 蓝橙色
        COLOR_B = (50, 50, 200)      # 选手B 红色
        COLOR_A_LIGHT = (230, 170, 80)
        COLOR_B_LIGHT = (80, 80, 240)
        COLOR_WHITE = (255, 255, 255)
        COLOR_YELLOW = (0, 255, 255)
        COLOR_GREEN = (0, 230, 118)
        COLOR_DARK = (30, 30, 30)

        # ── 计分面板高度（占画面上方 ~35%） ──
        panel_h = int(h * 0.35)
        mid_x = w // 2

        # ── 得分闪烁效果 ──
        flash_a = False
        flash_b = False
        if self.last_scoring_player and (now - self.last_score_change_time) < self.score_flash_duration:
            # 0.25秒闪一次
            blink = int((now - self.last_score_change_time) / 0.25) % 2 == 0
            if self.last_scoring_player == 'A':
                flash_a = blink
            else:
                flash_b = blink

        # ── 绘制左右半透明背景面板 ──
        alpha_a = 0.75 if flash_a else 0.55
        alpha_b = 0.75 if flash_b else 0.55
        bg_a = COLOR_A_LIGHT if flash_a else COLOR_A
        bg_b = COLOR_B_LIGHT if flash_b else COLOR_B
        self._draw_translucent_rect(frame, 0, 0, mid_x - 1, panel_h, bg_a, alpha_a)
        self._draw_translucent_rect(frame, mid_x + 1, 0, w, panel_h, bg_b, alpha_b)

        # ── 中央分割线 ──
        cv2.line(frame, (mid_x, 0), (mid_x, panel_h), COLOR_WHITE, 3)

        # ── 选手名称 ──
        name_y = 40
        self._put_text_centered(frame, 'A', mid_x // 2, name_y, 1.8, COLOR_WHITE, 3)
        self._put_text_centered(frame, 'B', mid_x + mid_x // 2, name_y, 1.8, COLOR_WHITE, 3)

        # ── 超大比分数字（核心：不戴眼镜也能看清） ──
        score_y = panel_h // 2 + 20
        score_scale = min(w, h) / 160.0  # 根据分辨率自适应，1280x720时约8.0
        score_scale = max(score_scale, 4.0)
        score_thick = max(int(score_scale * 1.8), 6)

        self._put_text_centered(frame, str(self.score_a), mid_x // 2, score_y,
                                score_scale, COLOR_WHITE, score_thick)
        self._put_text_centered(frame, str(self.score_b), mid_x + mid_x // 2, score_y,
                                score_scale, COLOR_WHITE, score_thick)

        # ── 中间 VS / 冒号 ──
        self._put_text_centered(frame, ':', mid_x, score_y, score_scale * 0.6, COLOR_YELLOW, score_thick - 2)

        # ── 总局比分（面板底部） ──
        game_y = panel_h - 15
        game_text = f'Games  {self.total_games_a} : {self.total_games_b}'
        self._put_text_centered(frame, game_text, mid_x, game_y, 1.2, COLOR_YELLOW, 3)

        # ── 发球方指示（小球图标 + 文字） ──
        serve_y = panel_h + 40
        serve_label = f'Serve >>  {self.serve_side}'
        # 在发球方一侧画一个小圆球
        if self.serve_side == 'A':
            ball_cx = mid_x // 2
        else:
            ball_cx = mid_x + mid_x // 2
        cv2.circle(frame, (ball_cx, serve_y - 5), 14, COLOR_YELLOW, -1)
        cv2.circle(frame, (ball_cx, serve_y - 5), 14, COLOR_DARK, 2)
        self._put_text_centered(frame, serve_label, mid_x, serve_y, 1.0, COLOR_GREEN, 2)

        # ── 冷却倒计时（面板下方居中，醒目红色） ──
        remaining = self.cooldown_period - (now - self.last_score_time)
        if 0 < remaining < self.cooldown_period:
            cd_text = f'Cooldown {remaining:.1f}s'
            cd_y = serve_y + 40
            self._put_text_centered(frame, cd_text, mid_x, cd_y, 0.9, (0, 0, 255), 2)

        # ── 底部状态栏（半透明黑条） ──
        bar_h = 36
        self._draw_translucent_rect(frame, 0, h - bar_h, w, h, COLOR_DARK, 0.65)
        # 左侧：按键提示
        cv2.putText(frame, "Q-Quit  R-Reset", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        # 右侧：TTS状态
        tts_label = "TTS ON" if self.tts_available else "TTS OFF"
        tts_color = COLOR_GREEN if self.tts_available else (0, 0, 200)
        cv2.putText(frame, tts_label, (w - 130, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tts_color, 2)

        # ── 比赛结束大字幕 ──
        if not self.game_active:
            # 全屏半透明遮罩
            self._draw_translucent_rect(frame, 0, 0, w, h, COLOR_DARK, 0.6)
            winner = 'A' if self.total_games_a > self.total_games_b else 'B'
            win_color = COLOR_A_LIGHT if winner == 'A' else COLOR_B_LIGHT
            self._put_text_centered(frame, f'Player {winner} Wins!', mid_x, h // 2 - 30, 3.0, win_color, 6)
            final_score = f'Match  {self.total_games_a} : {self.total_games_b}'
            self._put_text_centered(frame, final_score, mid_x, h // 2 + 60, 2.0, COLOR_YELLOW, 4)

        return frame
    
    def reset_game(self):
        """重置游戏（保留总局比分）"""
        self.score_a = 0
        self.score_b = 0
        self.game_active = True
        self.last_score_time = 0
        self.serve_side = 'A'
        self.score_source = ''
        self.last_scoring_player = None
        self.last_score_change_time = 0
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
        
        # 创建窗口（支持全屏）
        if WINDOW_FULLSCREEN:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        
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
        self._stop_tts_worker()


if __name__ == "__main__":
    scorer = TableTennisScorer()
    scorer.run()