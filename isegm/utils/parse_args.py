import argparse
from pathlib import Path
from easydict import EasyDict as edict

# 如果你的项目中包含 isegm.utils.exp，请保留下面这行，否则请注释掉并使用下方的 mock cfg
try:
    from isegm.utils.exp import load_config_file
except ImportError:
    load_config_file = None


def parse_args_val():
    parser = argparse.ArgumentParser()

    # 模型与检查点相关
    parser.add_argument('--mode', type=str, default='CMRefiner-V2',
                        help='Inference mode (e.g., base, focalclick)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint', type=str, default='/home/ubuntu/code/ca_mfp_icm_v2/experiments/ca_mfp_icm_v2/002_ca24_noW_0.01_mfpv2_100_icm4_448_cclvs/checkpoints/best_069.pth',
                        help='The name of the checkpoint to load')
    parser.add_argument('--resume-path', type=str, default='',
                        help='Path to the checkpoint file')
    parser.add_argument('--exp-path', type=str, default='',
                        help='Relative path to the experiment')

    # 路径相关
    parser.add_argument('--save-path', type=str, default='/home/ubuntu/result',
                        help='Path to save the results (logs and predictions)')
    parser.add_argument('--logs-path', type=str, default='/home/ubuntu/result',
                        help='Root path for logs (if save-path is not specified)')

    # 评估交互参数
    parser.add_argument('--n-clicks', type=int, default=20,
                        help='Maximum number of clicks per image')
    parser.add_argument('--min-n-clicks', type=int, default=1,
                        help='Minimum number of clicks')
    parser.add_argument('--target-iou', type=float, default=0.90,
                        help='Target IoU threshold to stop clicking')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Probability threshold for mask binarization')

    # 缩放(Zoom-In)策略参数
    parser.add_argument('--eval-mode', type=str, default='fixed448',
                        help='Evaluation mode (cvpr, fixed448, etc.)')

    parser.add_argument('--zoom-in-target-size', type=int, default=-1,
                        help='Target size for zoom-in crop')
    parser.add_argument('--zoom-in-expansion-ratio', type=float, default=-1,
                        help='Expansion ratio for zoom-in crop')
    parser.add_argument('--clicks-limit', type=int, default=-1,
                        help='Click limit for the predictor')

    # 增强与精炼
    parser.add_argument('--with-flip', action='store_true',
                        help='Use test-time augmentation (flipping)')
    parser.add_argument('--dilation-ratio', type=float, default=0.0,
                        help='Dilation ratio for masks')
    parser.add_argument('--refinement-mode', type=int, default=-1,
                        help='Refinement mode index')
    parser.add_argument('--refinement-iters', type=int, default=1,
                        help='Number of refinement iterations')

    # 可视化与分析
    parser.add_argument('--vis-preds', action='store_true', default='True',
                        help='Visualize predictions')
    parser.add_argument('--vis-aux', action='store_true',
                        help='Visualize auxiliary outputs')
    parser.add_argument('--iou-analysis', action='store_true',
                        help='Perform IoU analysis')
    parser.add_argument('--print-ious', action='store_true',
                        help='Print IoU for each click')

    # 配置文件
    parser.add_argument('--config-path', type=str, default='../config.yml',
                        help='Path to the config file')

    args = parser.parse_args()

    # 将路径字符串转换为 Path 对象，以匹配代码中的用法 (如 args.resume_path.exists())
    if args.resume_path:
        args.resume_path = Path(args.resume_path)
    if args.save_path:
        args.save_path = Path(args.save_path)
    if args.logs_path:
        args.logs_path = Path(args.logs_path)

    # 加载配置 (cfg)
    # 你的代码使用了 cfg.ZOOM_IN, cfg.EXPS_PATH 等
    if load_config_file is not None:
        cfg = load_config_file(args.config_path, return_edict=True)
    else:
        # --- [关键修改 2] 修改 Mock 配置中的尺寸 ---
        cfg = edict({
            'EXPS_PATH': Path('./experiments'),
            'INTERACTIVE_MODELS_PATH': Path('./weights'),
            'ZOOM_IN': {
                'CVPR': {
                    'TARGET_SIZE': 448,  # 从 600 改为 448
                    'EXPANSION_RATIO': 1.4
                },
                'FIXED': {
                    'SKIP_CLICKS': -1  # 确保一开始就进行 ZoomIn/Resize，避免原图尺寸输入
                }
            }
        })

    return args, cfg