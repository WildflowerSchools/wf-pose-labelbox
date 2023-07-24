import labelbox as lb
import slugify
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
        name='pose_track_2d_label',
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
    person_feature_schema_id = create_person_feature_schema(
        inference_id=inference_id,
        person_descriptions=person_descriptions,
        unusuable_bounding_box_label=unusuable_bounding_box_label,
        client=client,
    )
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



def generate_labelbox_client(api_key=None):
    if api_key is None:
        api_key = os.getenv('LABELBOX_API_KEY')
    client = lb.Client(api_key=api_key)
    return client