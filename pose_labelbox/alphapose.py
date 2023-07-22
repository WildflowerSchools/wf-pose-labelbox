import pose_labelbox.utils
import pose_labelbox.core
import pandas as pd
import numpy as np
import subprocess
import datetime
import json
import re
import uuid
import pathlib
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

def detect_poses_2d(
    image_list_path,
    output_directory_path,
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
        str(output_directory_path),
    ]
    if single_process:
        alphapose_command.append('--sp')
    if pose_tracking_reid:
        alphapose_command.append('--pose_track')
    arguments = docker_command + alphapose_command
    logger.info(f"Executing: {' '.join(arguments)}")
    subprocess.run(arguments)

def parse_alphapose_output(
    inference_id,
    start,
    end,
    environment_id=None,
    environment_name=None,
    camera_ids=None,
    camera_part_numbers=None,
    camera_serial_numbers=None,
    camera_names=None,
    video_duration=datetime.timedelta(seconds=10),
    frame_period=datetime.timedelta(milliseconds=100),
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None,
    alphapose_output_parent_directory='/data/alphapose_output',
    alphapose_output_filename='alphapose-results.json',
):
    target_camera_ids = pose_labelbox.core.generate_target_camera_ids(
        start=start,
        end=end,
        environment_id=environment_id,
        environment_name=environment_name,
        camera_ids=camera_ids,
        camera_part_numbers=camera_part_numbers,
        camera_serial_numbers=camera_serial_numbers,
        camera_names=camera_names,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret,      
    )
    for camera_id in target_camera_ids:
        alphapose_output_directory_path = generate_alphapose_output_directory_path(
            inference_id=inference_id,
            camera_id=camera_id,
            start=start,
            end=end,
            video_duration=video_duration,
            alphapose_output_parent_directory=alphapose_output_parent_directory,
        )
        alphapose_output_file_path = alphapose_output_directory_path / alphapose_output_filename
        parsed_alphapose_output_filename = generate_parsed_alphapose_output_filename(
            camera_id=camera_id,
            start=start,
            end=end,
            video_duration=video_duration,
        )
        parsed_alphapose_output_file_path = alphapose_output_directory_path / parsed_alphapose_output_filename
        if parsed_alphapose_output_file_path.is_file():
            logger.info(f'Parsed AlphaPose output file {parsed_alphapose_output_file_path} already exists. Skipping.')
            continue
        parse_alphapose_output_file(
            input_path=alphapose_output_file_path,
            output_path=parsed_alphapose_output_file_path,
            frame_period=frame_period,
        )

def parse_alphapose_output_file(
    input_path,
    output_path,
    frame_period=datetime.timedelta(milliseconds=100),
):
    input_path = pathlib.Path(input_path)
    with open(input_path, 'r') as fp:
        poses_2d_raw = json.load(fp)
    poses_2d = parse_poses_2d_raw(
        poses_2d_raw=poses_2d_raw,
        frame_period=frame_period,
    )
    poses_2d.to_pickle(output_path)

def parse_poses_2d_raw(
    poses_2d_raw,
    frame_period=datetime.timedelta(milliseconds=100),
):
    poses_2d_list = list()
    for pose_2d_raw in poses_2d_raw:
        pose_2d = parse_pose_2d_raw(
            pose_2d_raw=pose_2d_raw,
            frame_period=frame_period,
        )
        poses_2d_list.append(pose_2d)
    poses_2d = (
        pd.DataFrame(poses_2d_list)
        .sort_values('timestamp')
        .set_index('pose_2d_id')
    )
    return poses_2d

def parse_pose_2d_raw(
    pose_2d_raw,
    frame_period=datetime.timedelta(milliseconds=100),
):
    pose_2d_id = str(uuid.uuid4())
    image_id = pose_2d_raw['image_id']
    camera_id, timestamp = parse_image_id(
        image_id=image_id,
        frame_period=frame_period,
    )
    keypoints_flat = np.asarray(pose_2d_raw['keypoints'])
    keypoints = keypoints_flat.reshape((-1, 3))
    keypoint_coordinates = keypoints[:, :2]
    keypoint_quality = keypoints[:, 2]
    keypoints = np.where(keypoints == 0.0, np.nan, keypoints)
    keypoint_quality = np.where(keypoint_quality == 0.0, np.nan, keypoint_quality)
    pose_quality = pose_2d_raw['score']
    bounding_box_xywh = np.asarray(pose_2d_raw['box'])
    bounding_box_corners = np.array([
        [bounding_box_xywh[0], bounding_box_xywh[1]],
        [bounding_box_xywh[0] + bounding_box_xywh[2], bounding_box_xywh[1] + bounding_box_xywh[3]]
    ])
    pose_track_label = pose_2d_raw['idx']
    pose_2d = OrderedDict([
        ('pose_2d_id', pose_2d_id),
        ('camera_id', camera_id),
        ('timestamp', timestamp),
        ('bounding_box_xywh', bounding_box_xywh),
        ('bounding_box_corners', bounding_box_corners),
        ('keypoint_coordinates_2d', keypoint_coordinates),
        ('keypoint_quality_2d', keypoint_quality),
        ('pose_quality_2d', pose_quality),
        ('pose_track_label', pose_track_label),
    ])
    return pose_2d


image_id_re = re.compile(r'(?P<environment_id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_(?P<camera_id>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})_(?P<hour>[0-9]{2})-(?P<minute>[0-9]{2})-(?P<second>[0-9]{2})_(?P<frame_number>[0-9]{3})')
def parse_image_id(
    image_id,
    frame_period=datetime.timedelta(milliseconds=100),
):
    m = image_id_re.match(image_id)
    if not m:
        return ValueError(f'Image ID \'{image_id}\' could not be parsed')
    camera_id = m.group('camera_id')
    video_start = datetime.datetime(
        year=int(m.group('year')),
        month=int(m.group('month')),
        day=int(m.group('day')),
        hour=int(m.group('hour')),
        minute=int(m.group('minute')),
        second=int(m.group('second')),
        tzinfo=datetime.timezone.utc
    )
    frame_number = int(m.group('frame_number'))
    timestamp = video_start + (frame_number - 1)*frame_period
    return camera_id, timestamp

def generate_alphapose_output_directory_path(
    inference_id,
    camera_id,
    start,
    end,
    video_duration=datetime.timedelta(seconds=10),
    alphapose_output_parent_directory='/data/alphapose_output',
):
    output_start, output_end = pose_labelbox.utils.generate_output_period(
        start=start,
        end=end,
        video_duration=video_duration,
    )
    output_start_string = output_start.strftime('%Y%m%d_%H%M%S')
    output_end_string = output_end.strftime('%Y%m%d_%H%M%S')
    alphapose_output_directory_path = (
        pathlib.Path(alphapose_output_parent_directory) /
        inference_id /
        f'{camera_id}_{output_start_string}_{output_end_string}'
    )
    return alphapose_output_directory_path

def generate_parsed_alphapose_output_filename(
    camera_id,
    start,
    end,
    video_duration=datetime.timedelta(seconds=10),
):
    output_start, output_end = pose_labelbox.utils.generate_output_period(
        start=start,
        end=end,
        video_duration=video_duration,
    )
    output_start_string = output_start.strftime('%Y%m%d_%H%M%S')
    output_end_string = output_end.strftime('%Y%m%d_%H%M%S')
    parsed_alphapose_output_filename = f'poses_2d_{camera_id}_{output_start_string}_{output_end_string}.pkl'
    return parsed_alphapose_output_filename
