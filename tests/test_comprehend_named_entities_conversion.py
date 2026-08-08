import unittest

from ampav.core.schema import NamedEntityType

from ampav.aws.comprehend_named_entities_conversion import (
    aws_entities_to_named_entities,
)
from ampav.aws.errors import AwsComprehendNamedEntitiesSchemaError


class AwsComprehendNamedEntitiesConversionTest(unittest.TestCase):
    def test_maps_commercial_and_custom_labels(self) -> None:
        entities = aws_entities_to_named_entities(
            [
                {
                    "Text": "Kindle",
                    "Type": "COMMERCIAL_ITEM",
                    "Score": 0.9,
                    "BeginOffset": 0,
                    "EndOffset": 6,
                },
                {
                    "Text": "ENG-42",
                    "Type": "PRODUCT_CODE",
                    "Score": 0.8,
                    "BeginOffset": 7,
                    "EndOffset": 13,
                },
            ],
            language="en",
        )

        self.assertEqual(entities[0].label, "COMMERCIAL_ITEM")
        self.assertEqual(entities[0].type, NamedEntityType.BRAND)
        self.assertEqual(entities[0].language, "en")
        self.assertEqual(entities[1].label, "PRODUCT_CODE")
        self.assertEqual(entities[1].type, NamedEntityType.OTHER)

    def test_rejects_non_list_collection_with_native_path(self) -> None:
        with self.assertRaisesRegex(
            AwsComprehendNamedEntitiesSchemaError,
            r"\$\.response\.Entities: expected list",
        ):
            aws_entities_to_named_entities(
                {},
                path="$.response.Entities",
            )

    def test_reports_indexed_path_for_invalid_entity(self) -> None:
        with self.assertRaisesRegex(
            AwsComprehendNamedEntitiesSchemaError,
            r"\$\.response\.Entities\[1\]: missing required field 'Text'",
        ):
            aws_entities_to_named_entities(
                [
                    {
                        "Text": "Maya Chen",
                        "Type": "PERSON",
                        "BeginOffset": 0,
                        "EndOffset": 9,
                    },
                    {
                        "Type": "PERSON",
                        "BeginOffset": 10,
                        "EndOffset": 15,
                    },
                ],
                path="$.response.Entities",
            )


if __name__ == "__main__":
    unittest.main()
