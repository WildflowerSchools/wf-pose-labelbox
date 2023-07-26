import pose_labelbox.core
import pose_labelbox.alphapose
import cv_utils
import honeycomb_io
import pandas as pd
import tqdm
import tqdm.notebook
import datetime
import pathlib
import logging

logger = logging.getLogger(__name__)

def generate_bounding_box_overlays(
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
    show_timestamp=True,
    show_pose_track_label=True,
    local_frames_directory='/data/frames',
    frame_filename_extension='png',
    bounding_box_line_width=1.5,
    bounding_box_color='#00ff00',
    bounding_box_fill=False,
    bounding_box_alpha=1.0,
    timestamp_padding=5,
    timestamp_font_scale=1.5,
    timestamp_text_line_width=1,
    timestamp_text_color='#ffffff',
    timestamp_box_color='#000000',
    timestamp_box_fill=True,
    timestamp_box_alpha=0.3,
    pose_track_label_font_scale=2.0,
    pose_track_label_text_line_width=1.5,
    pose_track_label_text_color='#ffffff',
    pose_track_label_text_alpha=1.0,
    pose_track_label_box_line_width=1.5,
    pose_track_label_box_color='#00ff00',
    pose_track_label_box_fill=True,
    pose_track_label_box_alpha=0.5,
    progress_bar=False,
    notebook=False,
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
    if environment_id is None:
        environment_id = honeycomb_io.fetch_environment_id(environment_name=environment_name)
    for camera_id in target_camera_ids:
        logger.info(f'Generating bounding box overlay images for camera {camera_id}')
        alphapose_output_directory_path = pose_labelbox.alphapose.generate_alphapose_output_directory_path(
            inference_id=inference_id,
            camera_id=camera_id,
            start=start,
            end=end,
            video_duration=video_duration,
            alphapose_output_parent_directory=alphapose_output_parent_directory,
        )
        parsed_alphapose_output_filename = pose_labelbox.alphapose.generate_parsed_alphapose_output_filename(
            camera_id=camera_id,
            start=start,
            end=end,
            video_duration=video_duration,
        )
        parsed_alphapose_output_file_path = alphapose_output_directory_path / parsed_alphapose_output_filename
        poses_2d = (
            pd.read_pickle(parsed_alphapose_output_file_path)
            .sort_values([
                'pose_track_label',
                'timestamp'
            ])
        )
        base_pose_track_iterator = poses_2d.groupby(
            'pose_track_label',
            group_keys=False
        )
        num_pose_tracks = len(poses_2d['pose_track_label'].unique())
        if progress_bar:
            if notebook:
                pose_track_iterator = tqdm.notebook.tqdm(base_pose_track_iterator, total=num_pose_tracks)
            else:
                pose_track_iterator = tqdm(base_pose_track_iterator, total=num_pose_tracks)
        else:
            pose_track_iterator = base_pose_track_iterator
        for pose_track_label, pose_track in pose_track_iterator:
            pose_track_start = pose_track['timestamp'].min()
            pose_track_end = pose_track['timestamp'].max()
            timestamps = pd.date_range(
                start=pose_track_start,
                end=pose_track_end,
                freq=frame_period
            )
            for timestamp in timestamps:
                num_timestamp_occurrences = (pose_track['timestamp'] == timestamp).sum()
                if num_timestamp_occurrences > 1:
                    raise ValueError(f'Pose track {pose_trackl_label} contains duplicate timestamps')
                if num_timestamp_occurrences == 1:
                    show_bounding_box = True
                    bounding_box_corners=pose_track.loc[pose_track['timestamp'] == timestamp].iloc[0]['bounding_box_corners']
                else:
                    show_bounding_box = False
                    bounding_box_corners = None
                image_output_path = generate_bounding_box_overlay(
                    inference_id=inference_id,
                    environment_id=environment_id,
                    camera_id=camera_id,
                    timestamp=timestamp,
                    show_bounding_box=show_bounding_box,
                    bounding_box_corners=bounding_box_corners,
                    pose_track_label=pose_track_label,
                    show_timestamp=show_timestamp,
                    show_pose_track_label=show_pose_track_label,
                    video_duration=video_duration,
                    frame_period=frame_period,
                    local_frames_directory=local_frames_directory,
                    frame_filename_extension=frame_filename_extension,
                    bounding_box_line_width=bounding_box_line_width,
                    bounding_box_color=bounding_box_color,
                    bounding_box_fill=bounding_box_fill,
                    bounding_box_alpha=bounding_box_alpha,
                    timestamp_padding=timestamp_padding,
                    timestamp_font_scale=timestamp_font_scale,
                    timestamp_text_line_width=timestamp_text_line_width,
                    timestamp_text_color=timestamp_text_color,
                    timestamp_box_color=timestamp_box_color,
                    timestamp_box_fill=timestamp_box_fill,
                    timestamp_box_alpha=timestamp_box_alpha,
                    pose_track_label_font_scale=pose_track_label_font_scale,
                    pose_track_label_text_line_width=pose_track_label_text_line_width,
                    pose_track_label_text_color=pose_track_label_text_color,
                    pose_track_label_text_alpha=pose_track_label_text_alpha,
                    pose_track_label_box_line_width=pose_track_label_box_line_width,
                    pose_track_label_box_color=pose_track_label_box_color,
                    pose_track_label_box_fill=pose_track_label_box_fill,
                    pose_track_label_box_alpha=pose_track_label_box_alpha,
                )            



def generate_bounding_box_overlay(
    inference_id,
    environment_id,
    camera_id,
    timestamp,
    pose_track_label,
    show_timestamp=True,
    show_pose_track_label=True,
    show_bounding_box=False,
    bounding_box_corners=None,
    video_duration=datetime.timedelta(seconds=10),
    frame_period=datetime.timedelta(milliseconds=100),
    local_frames_directory='/data/frames',
    frame_filename_extension='png',
    bounding_box_line_width=1.5,
    bounding_box_color='#00ff00',
    bounding_box_fill=False,
    bounding_box_alpha=1.0,
    timestamp_padding=5,
    timestamp_font_scale=1.5,
    timestamp_text_line_width=1,
    timestamp_text_color='#ffffff',
    timestamp_box_color='#000000',
    timestamp_box_fill=True,
    timestamp_box_alpha=0.3,
    pose_track_label_font_scale=2.0,
    pose_track_label_text_line_width=1.5,
    pose_track_label_text_color='#ffffff',
    pose_track_label_text_alpha=1.0,
    pose_track_label_box_line_width=1.5,
    pose_track_label_box_color='#00ff00',
    pose_track_label_box_fill=True,
    pose_track_label_box_alpha=0.5,
):
    image_input_path = pose_labelbox.core.generate_frame_path(
        environment_id=environment_id,
        camera_id=camera_id,
        timestamp=timestamp,
        video_duration=video_duration,
        frame_period=frame_period,
        local_frames_directory=local_frames_directory,
        frame_filename_extension=frame_filename_extension,
    )
    image_output_path = generate_bounding_box_overlay_path(
        inference_id=inference_id,
        camera_id=camera_id,
        timestamp=timestamp,
        pose_track_label=pose_track_label,
    )
    if image_output_path.is_file():
        return
    image_output_path.parent.mkdir(parents=True, exist_ok=True)
    image = cv_utils.read_image(path=str(image_input_path))
    image = overlay_bounding_box(
        image=image,
        show_bounding_box=show_bounding_box,
        bounding_box_corners=bounding_box_corners,
        show_timestamp=show_timestamp,
        timestamp=timestamp,
        show_pose_track_label=show_pose_track_label,
        pose_track_label=pose_track_label,
        bounding_box_line_width=bounding_box_line_width,
        bounding_box_color=bounding_box_color,
        bounding_box_fill=bounding_box_fill,
        bounding_box_alpha=bounding_box_alpha,
        timestamp_padding=timestamp_padding,
        timestamp_font_scale=timestamp_font_scale,
        timestamp_text_line_width=timestamp_text_line_width,
        timestamp_text_color=timestamp_text_color,
        timestamp_box_color=timestamp_box_color,
        timestamp_box_fill=timestamp_box_fill,
        timestamp_box_alpha=timestamp_box_alpha,
        pose_track_label_font_scale=pose_track_label_font_scale,
        pose_track_label_text_line_width=pose_track_label_text_line_width,
        pose_track_label_text_color=pose_track_label_text_color,
        pose_track_label_text_alpha=pose_track_label_text_alpha,
        pose_track_label_box_line_width=pose_track_label_box_line_width,
        pose_track_label_box_color=pose_track_label_box_color,
        pose_track_label_box_fill=pose_track_label_box_fill,
        pose_track_label_box_alpha=pose_track_label_box_alpha,
    )
    cv_utils.write_image(
        image=image,
        path=str(image_output_path)
    )

def overlay_bounding_box(
    image,
    show_bounding_box=True,
    bounding_box_corners=None,
    show_timestamp=True,
    timestamp=None,
    show_pose_track_label=True,
    pose_track_label=None,
    bounding_box_line_width=1.5,
    bounding_box_color='#00ff00',
    bounding_box_fill=False,
    bounding_box_alpha=1.0,
    timestamp_padding=5,
    timestamp_font_scale=1.5,
    timestamp_text_line_width=1,
    timestamp_text_color='#ffffff',
    timestamp_box_color='#000000',
    timestamp_box_fill=True,
    timestamp_box_alpha=0.3,
    pose_track_label_font_scale=2.0,
    pose_track_label_text_line_width=1.5,
    pose_track_label_text_color='#ffffff',
    pose_track_label_text_alpha=1.0,
    pose_track_label_box_line_width=1.5,
    pose_track_label_box_color='#00ff00',
    pose_track_label_box_fill=True,
    pose_track_label_box_alpha=0.5,
):  
    if show_bounding_box:
        if bounding_box_corners is None:
            raise ValueError('Bounding box not specified')
        image = cv_utils.draw_rectangle(
            original_image=image,
            coordinates=bounding_box_corners,
            line_width=bounding_box_line_width,
            color=bounding_box_color,
            fill=bounding_box_fill,
            alpha=bounding_box_alpha
        )
    if show_timestamp:
        if show_timestamp is None:
            raise ValueError('Timestamp not specified')
        image = cv_utils.draw_timestamp(
            original_image=image,
            timestamp=timestamp,
            padding=timestamp_padding,
            font_scale=timestamp_font_scale,
            text_line_width=timestamp_text_line_width,
            text_color=timestamp_text_color,
            box_color=timestamp_box_color,
            box_fill=timestamp_box_fill,
            box_alpha=timestamp_box_alpha,
        )
    if show_pose_track_label:
        if pose_track_label is None:
            raise ValueError('Pose track label not specified')
        image = cv_utils.draw_text_box(
            original_image=image,
            anchor_coordinates=[0 + 5, 0 + 5],
            text=str(pose_track_label),
            horizontal_alignment='left',
            vertical_alignment='top',
            font_scale=pose_track_label_font_scale,
            text_line_width=pose_track_label_text_line_width,
            text_color=pose_track_label_text_color,
            text_alpha=pose_track_label_text_alpha,
            box_line_width=pose_track_label_box_line_width,
            box_color=pose_track_label_box_color,
            box_fill=pose_track_label_box_fill,
            box_alpha=pose_track_label_box_alpha,
        )
    return image

def generate_bounding_box_overlay_path(
    inference_id,
    camera_id,
    timestamp,
    pose_track_label,
    bounding_box_overlay_parent_directory='/data/bounding_box_overlays',
    overlay_image_extension='png',
):
    bounding_box_overlay_path = (
        pathlib.Path(bounding_box_overlay_parent_directory) /
        inference_id /
        camera_id /
        str(pose_track_label) /
        f"pose_track_overlay_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.{overlay_image_extension}"
    )
    return bounding_box_overlay_path
