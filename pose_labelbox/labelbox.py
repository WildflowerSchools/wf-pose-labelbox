import pose_labelbox.utils
import honeycomb_io
import labelbox as lb
import pandas as pd
import slugify
from collections import OrderedDict
import pathlib
import datetime
import uuid
import re
import os
import logging

logger = logging.getLogger(__name__)

LABELBOX_DATETIME_FORMAT = '%Y%m%dT%H-%M-%S.%fUTC'

def create_project(
    inference_id,
    ontology_id=None,
    person_descriptions=None,
    unusuable_bounding_box_label='Unusable bounding box',
    person_feature_schema_id=None,
    dataset_id=None,
    start=None,
    end=None,
    environment_name=None,
    environment_id=None,
    video_duration=datetime.timedelta(seconds=10),
    bounding_box_overlay_video_parent_directory='/data/bounding_box_overlay_videos',
    overlay_video_extension='mp4',
    frame_period=datetime.timedelta(milliseconds=100),
    client=None
):
    if client is None:
        client = generate_labelbox_client()
    name=f'Identify people ({inference_id})'
    existing_projects = client.get_projects(where=(lb.Project.name == name))
    existing_project = existing_projects.get_one()
    if existing_project is not None:
        logger.info(f'Person identification project for inference ID {inference_id} already exists. Skipping')
        return existing_project.uid
    create_metadata_fields(client)
    if ontology_id is None:
        logger.info(f'Person ontology for inference ID {inference_id} not specified. Creating.')
        ontology_id = create_person_ontology(
            inference_id=inference_id,
            person_descriptions=person_descriptions,
            unusuable_bounding_box_label=unusuable_bounding_box_label,
            person_feature_schema_id=person_feature_schema_id,
            client=client,
        )
    ontology = client.get_ontology(ontology_id)
    if dataset_id is None:
        logger.info(f'Dataset for inference ID {inference_id} not specified. Creating.')
        dataset_id = create_dataset(
            start=start,
            end=end,
            inference_id=inference_id,
            environment_name=environment_name,
            environment_id=environment_id,
            video_duration=video_duration,
            bounding_box_overlay_video_parent_directory=bounding_box_overlay_video_parent_directory,
            overlay_video_extension=overlay_video_extension,
            frame_period=frame_period,
            client=client,
        )
    dataset = client.get_dataset(dataset_id)
    logger.info('Creating project')
    project = client.create_project(
        name=name,
        media_type=lb.MediaType.Video
    )
    batch_name=f'First batch ({inference_id})'
    batch = project.create_batch(
        name=batch_name,
        data_rows=dataset.export_data_rows(),
        priority=1
    )
    project.setup_editor(ontology)
    return project.uid

def create_metadata_fields(
    client=None,
):
    if client is None:
        client = generate_labelbox_client()
    metadata_ontology = client.get_data_row_metadata_ontology()
    create_metadata_field(
        name='inference_id',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataString'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='environment_id',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataString'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='camera_id',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataString'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='labeling_period_start_isoformat',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataString'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='labeling_period_end_isoformat',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataString'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='pose_track_2d_label',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataString'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='video_start_isoformat',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataString'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='video_end_isoformat',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataString'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='num_frames',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataString'),
        metadata_ontology=metadata_ontology,
        client=client,
    )

def create_metadata_field(
    name,
    kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataString'),
    metadata_ontology=None,
    client=None,
):
    if client is None:
        client = generate_labelbox_client()
    if metadata_ontology is None:
        metadata_ontology = client.get_data_row_metadata_ontology()
    try:
        metadata_schema = metadata_ontology.get_by_name(name)
        logger.info(f'Metadata field \'{name}\' already exists. Skipping.')
    except KeyError:
        logger.info(f'Metadata field \'{name}\' doesn\'t exist. Creating.')
        metadata_ontology.create_schema(
            name=name,
            kind=kind,
        )

def create_person_ontology(
    inference_id,
    person_descriptions,
    unusuable_bounding_box_label='Unusable bounding box',
    person_feature_schema_id=None,
    client=None,
):
    if client is None:
        client = generate_labelbox_client()
    name = f'Person ({inference_id})'
    existing_ontologies = client.get_ontologies(name_contains=name)
    existing_ontology = existing_ontologies.get_one()
    if existing_ontology is not None:
        logger.info('Person ontology for inference ID {inference_id} already exists. Skipping')
        return existing_ontology.uid
    if person_descriptions is None:
        raise ValueError('Person descriptions not provided.')
    if person_feature_schema_id is None:
        logger.info('Person feature schema ID not provided. Creating.')
        person_feature_schema_id = create_person_feature_schema(
            inference_id=inference_id,
            person_descriptions=person_descriptions,
            unusuable_bounding_box_label=unusuable_bounding_box_label,
            client=client,
        )
    logger.info('Creating ontology')
    ontology = client.create_ontology_from_feature_schemas(
        name=name,
        feature_schema_ids=[person_feature_schema_id],
        media_type=lb.MediaType.Video
    )
    return ontology.uid

def create_person_feature_schema(
    inference_id,
    person_descriptions,
    unusuable_bounding_box_label='Unusable bounding box',
    client=None,
):
    if client is None:
        client = generate_labelbox_client()
    name = f'person_{inference_id}'
    instructions = f'Person ({inference_id})'
    existing_feature_schemas = client.get_feature_schemas(name_contains=instructions)
    existing_feature_schema = existing_feature_schemas.get_one()
    if existing_feature_schema is not None:
        logger.info('Person feature schema for inference ID {inference_id} already exists. Skipping')
        return existing_feature_schema.uid
    if person_descriptions is None:
        raise ValueError('Person descriptions not provided.')
    options = list()
    for person_description in person_descriptions:
        options.append(lb.Option(
            value=slugify.slugify(person_description),
            label=person_description,
        ))
    options.append(lb.Option(
        value='unusable',
        label=unusuable_bounding_box_label,
    ))
    person_radio_classification = lb.Classification(
        class_type=lb.Classification.Type.RADIO,
        name=name,
        instructions=instructions,
        options=options,
        scope=lb.Classification.Scope.INDEX
    )
    person_radio = client.create_feature_schema(person_radio_classification.asdict())
    return person_radio.uid

def create_dataset(
    start,
    end,
    inference_id,
    environment_name=None,
    environment_id=None,
    video_duration=datetime.timedelta(seconds=10),
    bounding_box_overlay_video_parent_directory='/data/bounding_box_overlay_videos',
    overlay_video_extension='mp4',
    frame_period=datetime.timedelta(milliseconds=100),
    client=None,
):
    if client is None:
        client = generate_labelbox_client()
    name = f'Pose tracks 2D ({inference_id})'
    existing_datasets = client.get_datasets(where=(lb.Dataset.name == name))
    existing_dataset = existing_datasets.get_one()
    if existing_dataset is not None:
        logger.info(f'Dataset for inference ID {inference_id} already exists. Skipping')
        return existing_dataset.uid
    environment_id = honeycomb_io.fetch_environment_id(
        environment_id=environment_id,
        environment_name=environment_name,
    )
    if start is None or end is None:
        raise ValueError('Start and end values not provided')
    labeling_period_start, labeling_period_end = pose_labelbox.utils.generate_output_period(
        start=start,
        end=end,
        video_duration=video_duration,
    )
    inference_directory_path = (
        pathlib.Path(bounding_box_overlay_video_parent_directory) /
        inference_id
    )
    logger.info('Creating dataset')
    dataset = client.create_dataset(
        iam_integration=None,
        name=name,
        description=name
    )
    datarows=list()
    for camera_directory_path in sorted(inference_directory_path.iterdir()):
        camera_id = camera_directory_path.name
        logger.info(f'Generating data rows for camera {camera_id}')
        for video_local_path in sorted(camera_directory_path.iterdir()):
            pose_track_label, video_start, video_end = parse_bounding_box_overlay_video_path(video_local_path)
            num_frames = round((video_end - video_start)/frame_period)
            logger.info(f'Generating data row for camera {camera_id} and pose track label {pose_track_label}')
            data_id = str(uuid.uuid4())
            video_url = client.upload_file(video_local_path)
            datarows.append({
                lb.DataRow.row_data: video_url,
                lb.DataRow.external_id: data_id,
                lb.DataRow.global_key: data_id,
                lb.DataRow.metadata_fields: [
                    lb.DataRowMetadataField(name='environment_id', value=environment_id),
                    lb.DataRowMetadataField(name='inference_id',  value=inference_id),
                    lb.DataRowMetadataField(name='labeling_period_start_isoformat',  value=labeling_period_start.strftime(LABELBOX_DATETIME_FORMAT)),
                    lb.DataRowMetadataField(name='labeling_period_end_isoformat',  value=labeling_period_end.strftime(LABELBOX_DATETIME_FORMAT)),
                    lb.DataRowMetadataField(name='camera_id',  value=camera_id),
                    lb.DataRowMetadataField(name='pose_track_2d_label',  value=pose_track_label),
                    lb.DataRowMetadataField(name='video_start_isoformat',  value=video_start.strftime(LABELBOX_DATETIME_FORMAT)),
                    lb.DataRowMetadataField(name='video_end_isoformat',  value=video_end.strftime(LABELBOX_DATETIME_FORMAT)),
                    lb.DataRowMetadataField(name='num_frames',  value=str(num_frames)),
                ]
            })
    create_task = dataset.create_data_rows(datarows)
    create_task.wait_till_done()
    if create_task.errors:
        raise Exception(f'Creation task errors: {create_task.errors}')
    logger.info('Creation task status: {create_task.status}')
    return dataset.uid

bounding_box_overlay_filename_re = re.compile(r'(?P<pose_track_label>[0-9]+)_(?P<start_year_string>[0-9]{4})(?P<start_month_string>[0-9]{2})(?P<start_day_string>[0-9]{2})_(?P<start_hour_string>[0-9]{2})(?P<start_minute_string>[0-9]{2})(?P<start_second_string>[0-9]{2})_(?P<start_microsecond_string>[0-9]{6})_(?P<end_year_string>[0-9]{4})(?P<end_month_string>[0-9]{2})(?P<end_day_string>[0-9]{2})_(?P<end_hour_string>[0-9]{2})(?P<end_minute_string>[0-9]{2})(?P<end_second_string>[0-9]{2})_(?P<end_microsecond_string>[0-9]{6})')
def parse_bounding_box_overlay_video_path(
    path
):
    m = bounding_box_overlay_filename_re.match(path.stem)
    if not m:
        raise ValueError(f'Failed to parse filename \'{filename_stem}\'')
    pose_track_label = m.group('pose_track_label')
    video_start = datetime.datetime(
        year=int(m.group('start_year_string')),
        month=int(m.group('start_month_string')),
        day=int(m.group('start_day_string')),
        hour=int(m.group('start_hour_string')),
        minute=int(m.group('start_minute_string')),
        second=int(m.group('start_second_string')),
        microsecond=int(m.group('start_microsecond_string')),
        tzinfo=datetime.timezone.utc
    )
    video_end = datetime.datetime(
        year=int(m.group('end_year_string')),
        month=int(m.group('end_month_string')),
        day=int(m.group('end_day_string')),
        hour=int(m.group('end_hour_string')),
        minute=int(m.group('end_minute_string')),
        second=int(m.group('end_second_string')),
        microsecond=int(m.group('end_microsecond_string')),
        tzinfo=datetime.timezone.utc
    )
    return pose_track_label, video_start, video_end

def fetch_labels(
    project_id,
    frame_period=datetime.timedelta(milliseconds=100),
    client=None,
):
    if client is None:
        client = generate_labelbox_client()
    export_params= {
    "attachments": False,
    "metadata_fields": True,
    "data_row_details": False,
    "project_details": False,
    "label_details": True,
    "performance_details": False
    }
    filters=None
    project = client.get_project(project_id)
    export_task = project.export_v2(params=export_params, filters=filters)
    export_task.wait_till_done()
    if export_task.errors:
        raise Exception(f'Export task errors: {export_task.errors}')
    data_rows = export_task.result
    label_data_df_list = list()
    for data_row in data_rows:
        metadata = dict()
        for metadata_field in data_row.get('metadata_fields'):
            metadata[metadata_field['schema_name']] = metadata_field['value']
        inference_id = metadata.get('inference_id')
        environment_id = metadata.get('environment_id')
        labeling_period_start = pd.to_datetime(
            metadata.get('labeling_period_start_isoformat'),
            format=LABELBOX_DATETIME_FORMAT,
            utc=True
        )
        labeling_period_end = pd.to_datetime(
            metadata.get('labeling_period_end_isoformat'),
            format=LABELBOX_DATETIME_FORMAT,
            utc=True
        )
        camera_id = metadata.get('camera_id')
        pose_track_2d_label = int(metadata.get('pose_track_2d_label'))
        pose_track_start = pd.to_datetime(
            metadata.get('video_start_isoformat'),
            format=LABELBOX_DATETIME_FORMAT,
            utc=True
        )
        pose_track_end = pd.to_datetime(
            metadata.get('video_end_isoformat'),
            format=LABELBOX_DATETIME_FORMAT,
            utc=True
        )
        num_frames = int(metadata.get('num_frames'))
        labels = data_row['projects'][project_id]['labels']
        if len(labels) == 0:
            continue
        if len(labels) > 1:
            raise ValueError(f'More than one label found for pose track {pose_track_2d_label}')
        label = labels[0]
        pose_track_data_list = list()
        for frame_number, frame_data in label['annotations']['frames'].items():
            frame_number = int(frame_number)
            timestamp = pose_track_start + (frame_number - 1)*frame_period
            classifications = frame_data['classifications']
            if len(classifications) == 0:
                continue
            if len(classifications) > 1:
                raise ValueError(f'More than one classification found for frame number {frame_number} in pose track {pose_track_2d_label}')
            classification = classifications[0]
            person_name = classifications[0]['radio_answer']['name']
            pose_track_data_list.append({
                'timestamp': timestamp,
                'person_name': person_name
            })
        pose_track_data = (
            pd.DataFrame(pose_track_data_list)
            .set_index('timestamp')
            .sort_index()
        )
        new_index = pd.date_range(
            start=pose_track_data.index.min(),
            end=pose_track_data.index.max(),
            freq=frame_period,
            name='timestamp'
        )
        pose_track_data_filled  = pose_track_data.reindex(
            index=new_index,
            method='ffill'
        )
        pose_track_data_filled['camera_id'] = camera_id
        pose_track_data_filled['pose_track_2d_label'] = pose_track_2d_label
        label_data_df_list.append(pose_track_data_filled)
    label_data = (
        pd.concat(label_data_df_list)
        .reset_index()
        .set_index([
            'camera_id',
            'pose_track_2d_label',
            'timestamp'
        ])
        .sort_index()
    )
    return label_data

def generate_labelbox_client(api_key=None):
    if api_key is None:
        api_key = os.getenv('LABELBOX_API_KEY')
    client = lb.Client(api_key=api_key)
    return client