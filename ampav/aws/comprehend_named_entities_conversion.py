"""Convert native AWS Comprehend entities into AMPAV schema objects."""

from typing import Any

from pydantic import ValidationError

from ampav.core.schema import NamedEntity, NamedEntityType

from .errors import AwsComprehendNamedEntitiesSchemaError


def aws_entities_to_named_entities(
    native_entities: object,
    *,
    language: str | None = None,
    path: str = "$.Entities",
) -> list[NamedEntity]:
    """Convert and validate a native Comprehend ``Entities`` collection.

    Args:
        native_entities: Value of the native response's ``Entities`` field.
        language: Optional source language assigned to each converted entity.
        path: Validation path identifying the collection in its native response.
    """
    if not isinstance(native_entities, list):
        raise AwsComprehendNamedEntitiesSchemaError(path, "expected list")
    return [
        aws_entity_to_named_entity(
            entity,
            language=language,
            path=f"{path}[{index}]",
        )
        for index, entity in enumerate(native_entities)
    ]


def aws_entity_to_named_entity(
    entity: Any,
    *,
    language: str | None = None,
    path: str = "$.Entities[]",
) -> NamedEntity:
    """Convert and validate one native Comprehend entity.

    Args:
        entity: Native entity mapping returned by AWS Comprehend.
        language: Optional source language assigned to the converted entity.
        path: Validation path identifying the entity in its native response.
    """
    if not isinstance(entity, dict):
        raise AwsComprehendNamedEntitiesSchemaError(path, "expected JSON object")
    try:
        label = str(entity["Type"])
        return NamedEntity(
            text=str(entity["Text"]),
            type=_named_entity_type_for_label(label),
            label=label,
            confidence=None if entity.get("Score") is None else float(entity["Score"]),
            begin_offset=int(entity["BeginOffset"]),
            end_offset=int(entity["EndOffset"]),
            language=language,
        )
    except KeyError as exc:
        raise AwsComprehendNamedEntitiesSchemaError(
            path,
            f"missing required field {exc.args[0]!r}",
        ) from exc
    except (TypeError, ValueError, ValidationError) as exc:
        raise AwsComprehendNamedEntitiesSchemaError(
            path,
            f"invalid entity data: {exc}",
        ) from exc


def _named_entity_type_for_label(label: str) -> NamedEntityType:
    """Map a native Comprehend label to the AMPAV canonical type."""
    normalized_label = label.strip().casefold()
    if normalized_label == "commercial_item":
        return NamedEntityType.BRAND
    try:
        return NamedEntityType(normalized_label)
    except ValueError:
        # Custom Comprehend recognizers may return caller-defined labels.
        return NamedEntityType.OTHER
