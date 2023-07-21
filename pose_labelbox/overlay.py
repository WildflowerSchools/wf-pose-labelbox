import pose_labelbox.core
import cv_utils
import datetime
import pathlib
import logging

logger = logging.getLogger(__name__)


def generate_bounding_box_overlay(
    inference_id,
    environment_id,
    camera_id,
    timestamp,
    bounding_box_corners,
    pose_track_label,
    show_timestamp=True,
    show_pose_track_label=True,
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
    bounding_box_corners,
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
    image = cv_utils.draw_rectangle(
        original_image=image,
        coordinates=bounding_box_corners,
        line_width=bounding_box_line_width,
        color=bounding_box_color,
        fill=bounding_box_fill,
        alpha=bounding_box_alpha
    )
    if show_timestamp:
        if timestamp is None:
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
