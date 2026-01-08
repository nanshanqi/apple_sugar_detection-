import os

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 数据路径
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')

# 模型路径
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# 结果路径
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')