"""
Ping Pong Smart Scoring System
Combining pose recognition technology to achieve automated scoring statistics, 
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

from config import *


class PlayerGestureState:
    def __init__(self):
        self.is_holding_high = False
        self.start_hold_time = 0
        self.cooldown_until = 0


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
        
        # 初始化选手举手状态机
        self.left_gesture = PlayerGestureState()
        self.right_gesture = PlayerGestureState()

    # Removed start_listening call from the main loop

    # Updated main loop to remove voice recognition references
    """运行主循环"""
    print("Starting Ping Pong Smart Scoring System...")
    print("Use touch gestures to score")
    print("Keys: Q-Quit  R-Reset  F-Full Reset")
    print("      A/S - Player A +1/-1")
    print("      B/N - Player B +1/-1")

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

        # 检测举手悬停手势
        side, pose_detected = self.detect_pose(frame)
        current_time = time.time()
        # 左右两侧状态机
        for player, gesture in [('A', self.left_gesture), ('B', self.right_gesture)]:
            if current_time < gesture.cooldown_until:
                gesture.is_holding_high = False
                continue
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
            self.manual_adjust_score('A', +1)  # A方 +1
        elif key == ord('s'):
            self.manual_adjust_score('A', -1)  # A方 -1
        elif key == ord('b'):
            self.manual_adjust_score('B', +1)  # B方 +1
        elif key == ord('n'):
            self.manual_adjust_score('B', -1)  # B方 -1
        elif key == ord('f'):
            self.full_reset()                  # 全局重置
        
    # 释放资源
    self.stop_listening()
    cap.release()
    cv2.destroyAllWindows()
    self._stop_tts_worker()


if __name__ == "__main__":
    scorer = TableTennisScorer()
    scorer.run()