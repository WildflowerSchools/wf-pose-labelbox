
import subprocess
import logging

logger = logging.getLogger(__name__)

def detect_poses_2d(
    image_list_path,
    output_directory,
    docker_image='alphapose-12-1',
    config_file='configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
    model_file='pretrained_models/halpe26_fast_res50_256x192.pth',
    detector_name='yolox-x',
    detector_batch_size_per_gpu=30,
    pose_batch_size_per_gpu=100,
    gpus='0',
    format='coco',
    pose_tracking_reid=True,
    single_process=True,
):
    logger.info(f'Detecting 2D poses for images in {image_list_path}')
    with open(image_list_path, 'r') as fp:
        image_count = 0
        for line in fp.readlines():
            image_count += 1
    logger.info(f'Found {image_count} images in {image_list_path}')
    docker_command = [
        'docker',
        'run',
        '--rm',
        '--gpus',
        'all',
        '--shm-size=2gb',
        '-v',
        '/data:/data',
        '-v',
        '/data/alphapose_models/pretrained_models:/build/AlphaPose/pretrained_models',
        '-v',
        '/data/alphapose_models/detector/yolo/data:/build/AlphaPose/detector/yolo/data',
        '-v',
        '/data/alphapose_models/detector/yolox/data:/build/AlphaPose/detector/yolox/data',
        '-v',
        '/data/alphapose_models/detector/yolox/data:/build/AlphaPose/detector/yolox/data',
        '-v',
        '/data/alphapose_models/trackers/weights:/build/AlphaPose/trackers/weights',
        '-v',
        '/data/alphapose_models/detector/tracker/data:/build/AlphaPose/detector/tracker/data',
        docker_image,
    ]
    alphapose_command = [
        'python3',
        'scripts/demo_inference.py',
        '--cfg',
        config_file,
        '--checkpoint',
        model_file,
        '--detector',
        detector_name,
        '--detbatch',
        f'{detector_batch_size_per_gpu}',
        '--posebatch',
        f'{pose_batch_size_per_gpu}',
        '--gpus',
        gpus,
        '--format',
        format,
        '--list',
        str(image_list_path),
        '--outdir',
        str(output_directory),
    ]
    if single_process:
        alphapose_command.append('--sp')
    if pose_tracking_reid:
        alphapose_command.append('--pose_track')
    arguments = docker_command + alphapose_command
    logger.info(f"Executing: {' '.join(arguments)}")
    subprocess.run(arguments)

