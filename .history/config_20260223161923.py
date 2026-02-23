"""
配置文件 - 乒乓球智能计分系统
定义系统运行的各种参数和阈值
"""

# 摄像头设置
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# 姿态识别设置
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.7
MAX_NUM_HANDS = 2

# 语音识别设置
VOICE_KEYWORDS = ["得分", "score", "好球", "great", "goal", "goat"]
VOICE_ENERGY_THRESHOLD = 300     # 音量阈值（降低以便识别短促喊声）
VOICE_PHRASE_TIME_LIMIT = 0.8    # 最长录音时间（秒，喊A/B很短，缩短加速响应）
VOICE_LISTEN_TIMEOUT = 1.5       # 监听超时（秒，无声音则跳过）

# 游戏规则设置
COOLDOWN_PERIOD = 3            # 得分后冷却时间（秒）
SERVE_CHANGE_INTERVAL = 2      # 每几分更换发球方
WINNING_SCORE = 11             # 获胜分数
MINIMUM_WINNING_DIFFERENCE = 2 # 获胜最小分差

# UI显示设置
FONT_SCALE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)    # 白色
SCORE_COLOR = (0, 255, 0)      # 绿色
SERVE_ARROW_COLOR = (0, 255, 255)  # 黄色
ERROR_COLOR = (0, 0, 255)      # 红色
SCORE_FLASH_DURATION = 1.5     # 得分闪烁持续秒数

# 窗口设置
WINDOW_NAME = "乒乓球智能计分系统"
WINDOW_FULLSCREEN = True        # 全屏显示，方便远距离观看

# 调试选项
DEBUG_MODE = True
LOG_TO_FILE = True
LOG_FILE_PATH = "scoring_log.txt"