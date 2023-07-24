import labelbox as lb
import pathlib
import os
import logging

logger = logging.getLogger(__name__)

def create_metadata_fields(
    client=None,
):
    if client is None:
        client = generate_labelbox_client()
    metadata_ontology = client.get_data_row_metadata_ontology()
    create_metadata_field(
        name='inference_id',
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='environment_id',
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='camera_id',
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='labeling_period_start',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataDateTime'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='labeling_period_end',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataDateTime'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='video_start',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataDateTime'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='video_end',
        kind=lb.schema.data_row_metadata.DataRowMetadataKind('CustomMetadataDateTime'),
        metadata_ontology=metadata_ontology,
        client=client,
    )
    create_metadata_field(
        name='pose_track_label',
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

def generate_labelbox_client(api_key=None):
    if api_key is None:
        api_key = os.getenv('LABELBOX_API_KEY')
    client = lb.Client(api_key=api_key)
    return client