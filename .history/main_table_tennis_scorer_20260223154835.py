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
    TTS_AVAILABLE = False  # 彻底禁用语音播报
except ImportError:
    print("Warning: pyttsx3 module not found, text-to-speech feature unavailable")
    TTS_AVAILABLE = False

# 添加语音识别功能
try:
    SR_AVAILABLE = False  # 彻底禁用语音识别
except ImportError:
    print("Warning: speech_recognition module not found, voice scoring unavailable")
    SR_AVAILABLE = False

from config import *


class TableTennisScorer:
        class PlayerGestureState:
            def __init__(self):
                self.is_holding_high = False
                self.start_hold_time = 0
                self.cooldown_until = 0

        def _init_gesture_states(self):
            self.left_gesture = self.PlayerGestureState()
            self.right_gesture = self.PlayerGestureState()

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
        
        # 彻底禁用语音播报相关变量
        self.tts_available = False
        
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
        
        # 彻底禁用语音识别相关变量
        self.voice_recognition_available = False
        self._init_gesture_states()


    # 语音识别相关方法全部禁用
    def start_listening(self):
        pass


    def _voice_listen_loop(self):
        pass


    def _recognize_and_score(self, audio):
        pass


    def _match_voice_command(self, text):
        pass


    def stop_listening(self):
        pass
    

    def speak_score(self, player, current_score_a, current_score_b):
        pass


    def _speak_async(self, text):
        pass


    def _start_tts_worker(self):
        pass


    def _tts_worker_loop(self):
        pass


    def _stop_tts_worker(self):
        pass

    # 移除备用音频播放方法
    
    def detect_raise_and_hold_gesture(self, landmarks, image_shape):
        """
        检测举手悬停手势：
        - 手腕关键点在画面上1/3区域（高举）
        - 判断左右分区（x<0.5为左，x>=0.5为右）
        返回: (side, is_high)
        """
        h, w = image_shape[:2]
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        # 归一化坐标转像素
        wrist_x, wrist_y = wrist.x, wrist.y
        midtip_x, midtip_y = middle_tip.x, middle_tip.y
        # 只要手腕或中指指尖有一个在画面上1/3区域就算举高
        high_threshold = 0.33
        is_high = (wrist_y < high_threshold) or (midtip_y < high_threshold)
        # 左右分区
        side = None
        if is_high:
            if wrist_x < 0.5:
                side = 'A'
            else:
                side = 'B'
        return side, is_high
    
    def detect_pose(self, image):
        """检测举手悬停手势，返回(side, is_high)"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                side, is_high = self.detect_raise_and_hold_gesture(hand_landmarks.landmark, image.shape)
                if is_high:
                    return side, True
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
        #self.speak_score(scoring_player, self.score_a, self.score_b)
        
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
                    def draw_ui(self, frame):
                        """绘制美化后的用户界面 —— 超大比分、高对比度、一眼可见，并为举手悬停加分手势显示进度反馈"""
                        h, w, c = frame.shape
                        now = time.time()

                        # 颜色定义
                        COLOR_A = (200, 120, 50)
                        COLOR_B = (50, 50, 200)
                        COLOR_A_LIGHT = (230, 170, 80)
                        COLOR_B_LIGHT = (80, 80, 240)
                        COLOR_WHITE = (255, 255, 255)
                        COLOR_YELLOW = (0, 255, 255)
                        COLOR_GREEN = (0, 230, 118)
                        COLOR_DARK = (30, 30, 30)
                        COLOR_PROGRESS = (0, 180, 255)
                        COLOR_PROGRESS_OK = (0, 220, 0)

                        # 计分面板高度
                        panel_h = int(h * 0.35)
                        mid_x = w // 2

                        # 得分闪烁效果
                        flash_a = False
                        flash_b = False
                        if self.last_scoring_player and (now - self.last_score_change_time) < self.score_flash_duration:
                            blink = int((now - self.last_score_change_time) / 0.25) % 2 == 0
                            if self.last_scoring_player == 'A':
                                flash_a = blink
                            else:
                                flash_b = blink

                        # 绘制左右半透明背景面板
                        alpha_a = 0.75 if flash_a else 0.55
                        alpha_b = 0.75 if flash_b else 0.55
                        bg_a = COLOR_A_LIGHT if flash_a else COLOR_A
                        bg_b = COLOR_B_LIGHT if flash_b else COLOR_B
                        self._draw_translucent_rect(frame, 0, 0, mid_x - 1, panel_h, bg_a, alpha_a)
                        self._draw_translucent_rect(frame, mid_x + 1, 0, w, panel_h, bg_b, alpha_b)

                        # 中央分割线
                        cv2.line(frame, (mid_x, 0), (mid_x, panel_h), COLOR_WHITE, 3)

                        # 选手名称
                        name_y = 40
                        self._put_text_centered(frame, 'A', mid_x // 2, name_y, 1.8, COLOR_WHITE, 3)
                        self._put_text_centered(frame, 'B', mid_x + mid_x // 2, name_y, 1.8, COLOR_WHITE, 3)

                        # 超大比分数字
                        score_y = panel_h // 2 + 20
                        score_scale = max(min(w, h) / 160.0, 4.0)
                        score_thick = max(int(score_scale * 1.8), 6)
                        self._put_text_centered(frame, str(self.score_a), mid_x // 2, score_y, score_scale, COLOR_WHITE, score_thick)
                        self._put_text_centered(frame, str(self.score_b), mid_x + mid_x // 2, score_y, score_scale, COLOR_WHITE, score_thick)

                        # 中间冒号
                        self._put_text_centered(frame, ':', mid_x, score_y, score_scale * 0.6, COLOR_YELLOW, score_thick - 2)

                        # 总局比分
                        game_y = panel_h - 15
                        game_text = f'Games  {self.total_games_a} : {self.total_games_b}'
                        self._put_text_centered(frame, game_text, mid_x, game_y, 1.2, COLOR_YELLOW, 3)

                        # 发球方指示
                        serve_y = panel_h + 40
                        serve_label = f'Serve >>  {self.serve_side}'
                        if self.serve_side == 'A':
                            ball_cx = mid_x // 2
                        else:
                            ball_cx = mid_x + mid_x // 2
                        cv2.circle(frame, (ball_cx, serve_y - 5), 14, COLOR_YELLOW, -1)
                        cv2.circle(frame, (ball_cx, serve_y - 5), 14, COLOR_DARK, 2)
                        self._put_text_centered(frame, serve_label, mid_x, serve_y, 1.0, COLOR_GREEN, 2)

                        # 冷却倒计时
                        remaining = self.cooldown_period - (now - self.last_score_time)
                        if 0 < remaining < self.cooldown_period:
                            cd_text = f'Cooldown {remaining:.1f}s'
                            cd_y = serve_y + 40
                            self._put_text_centered(frame, cd_text, mid_x, cd_y, 0.9, (0, 0, 255), 2)

                        # 举手悬停进度条/圆环反馈
                        # 位置：A方在左上，B方在右上
                        for player, gesture, cx in [
                            ('A', self.left_gesture, mid_x // 2),
                            ('B', self.right_gesture, mid_x + mid_x // 2)
                        ]:
                            # 冷却期内显示灰色圆环
                            if now < gesture.cooldown_until:
                                pct = (gesture.cooldown_until - now) / 3.0
                                pct = min(max(pct, 0), 1)
                                color = (120, 120, 120)
                                cv2.ellipse(frame, (cx, name_y - 30), (28, 28), 0, 0, int(360 * pct), color, 6)
                                cv2.putText(frame, 'CD', (cx - 18, name_y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            elif gesture.is_holding_high:
                                # 进度条/圆环
                                held = now - gesture.start_hold_time
                                pct = min(max(held / 1.0, 0), 1)
                                color = COLOR_PROGRESS_OK if pct >= 1.0 else COLOR_PROGRESS
                                cv2.ellipse(frame, (cx, name_y - 30), (28, 28), 0, 0, int(360 * pct), color, 6)
                                if pct < 1.0:
                                    cv2.putText(frame, 'Hold', (cx - 24, name_y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                else:
                                    cv2.putText(frame, '+1', (cx - 18, name_y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # 底部状态栏
                        bar_h = 36
                        self._draw_translucent_rect(frame, 0, h - bar_h, w, h, COLOR_DARK, 0.65)
                        cv2.putText(frame, "Q-Quit R-Reset F-FullReset | A/Z:A+/- B/X:B+/-", (10, h - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                        # 右侧：语音识别 + TTS状态（已禁用，显示OFF）
                        vr_label = "MIC OFF"
                        vr_color = (0, 0, 200)
                        cv2.putText(frame, vr_label, (w - 260, h - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, vr_color, 2)
                        tts_label = "TTS OFF"
                        tts_color = (0, 0, 200)
                        cv2.putText(frame, tts_label, (w - 130, h - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tts_color, 2)

                        # 比赛结束大字幕
                        if not self.game_active:
                            self._draw_translucent_rect(frame, 0, 0, w, h, COLOR_DARK, 0.6)
                            winner = 'A' if self.total_games_a > self.total_games_b else 'B'
                            win_color = COLOR_A_LIGHT if winner == 'A' else COLOR_B_LIGHT
                            self._put_text_centered(frame, f'Player {winner} Wins!', mid_x, h // 2 - 30, 3.0, win_color, 6)
                            final_score = f'Match  {self.total_games_a} : {self.total_games_b}'
                            self._put_text_centered(frame, final_score, mid_x, h // 2 + 60, 2.0, COLOR_YELLOW, 4)

                        return frame
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
        cv2.putText(frame, "Q-Quit R-Reset F-FullReset | A/Z:A+/- B/X:B+/-", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        # 右侧：语音识别 + TTS状态
        vr_label = "MIC ON" if self.voice_recognition_available else "MIC OFF"
        vr_color = COLOR_GREEN if self.voice_recognition_available else (0, 0, 200)
        cv2.putText(frame, vr_label, (w - 260, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, vr_color, 2)
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
    
    def manual_adjust_score(self, player, delta):
        """手动调整比分
        player: 'A' 或 'B'
        delta: +1 加分, -1 减分
        """
        if player == 'A':
            self.score_a = max(0, self.score_a + delta)
        else:
            self.score_b = max(0, self.score_b + delta)
        
        # 重新计算发球方
        total_score = self.score_a + self.score_b
        # 根据总分重新推算发球方：初始A发，每SERVE_CHANGE_INTERVAL分轮换
        switches = total_score // SERVE_CHANGE_INTERVAL
        self.serve_side = 'A' if switches % 2 == 0 else 'B'
        
        # 更新闪烁动画
        self.last_scoring_player = player if delta > 0 else None
        self.last_score_change_time = time.time()
        self.score_source = 'manual'
        
        action = '+1' if delta > 0 else '-1'
        log_msg = f"Manual adjust: Player {player} {action}, Score A:{self.score_a} - B:{self.score_b}"
        print(log_msg)
        
        # 播报当前比分
        # if self.tts_available:
        #     serve_text = f"{self.serve_side} serve."
        #     text = f"Score: A {self.score_a}, B {self.score_b}. {serve_text}"
        #     self._speak_async(text)
        
        # if LOG_TO_FILE:
        #     logging.info(log_msg)
    
    def full_reset(self):
        """全局重置（包括总局比分）"""
        self.score_a = 0
        self.score_b = 0
        self.total_games_a = 0
        self.total_games_b = 0
        self.game_active = True
        self.last_score_time = 0
        self.serve_side = 'A'
        self.score_source = ''
        self.last_scoring_player = None
        self.last_score_change_time = 0
        print("Full reset: all scores cleared")
        if self.tts_available:
            self._speak_async("Full reset. All scores cleared.")
        if LOG_TO_FILE:
            logging.info("Full reset: all scores cleared")

    def run(self):
        """运行主循环"""
        print("Starting Ping Pong Smart Scoring System...")
        print(f"Text-to-Speech: Disabled")
        print(f"Voice Recognition: Disabled")
        print(f"Use gesture only to score")
        print("Keys: Q-Quit  R-Reset  F-Full Reset")
        print("      A/S - Player A +1/-1")
        print("      B/N - Player B +1/-1")
        
        # 语音识别已禁用
        
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
            frame = cv2.flip(frame, 1)

            # 检测所有手势
            side, pose_detected = self.detect_pose(frame)
            current_time = time.time()

            # 左右两侧状态机
            for player, gesture in [('A', self.left_gesture), ('B', self.right_gesture)]:
                # 冷却期内，跳过
                if current_time < gesture.cooldown_until:
                    gesture.is_holding_high = False
                    continue
                # 检测到该侧举手
                if pose_detected and side == player:
                    if not gesture.is_holding_high:
                        gesture.is_holding_high = True
                        gesture.start_hold_time = current_time
                    else:
                        held_duration = current_time - gesture.start_hold_time
                        if held_duration > 1.0:
                            self.process_score('pose', player)
                            gesture.cooldown_until = current_time + 3.0
                            gesture.is_holding_high = False
                else:
                    gesture.is_holding_high = False

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
            elif key == ord('a'):
                self.manual_adjust_score('A', +1)
            elif key == ord('s'):
                self.manual_adjust_score('A', -1)
            elif key == ord('b'):
                self.manual_adjust_score('B', +1)
            elif key == ord('n'):
                self.manual_adjust_score('B', -1)
            elif key == ord('f'):
                self.full_reset()

        # 释放资源
        # 语音识别已禁用
        cap.release()
        cv2.destroyAllWindows()
        self._stop_tts_worker()
        
        # 释放资源
        # 语音识别已禁用
        cap.release()
        cv2.destroyAllWindows()
        self._stop_tts_worker()


if __name__ == "__main__":
    scorer = TableTennisScorer()
    scorer.run()