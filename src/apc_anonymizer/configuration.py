"""Read configuration."""

import collections
import os
import pathlib
from importlib import resources

import jsonschema

from apc_anonymizer import yaml_workaround as yaml


def read_configuration_schema():
    """Read the configuration schema."""
    return (
        resources.files("apc_anonymizer")
        .joinpath("apc-anonymizer-schema.json")
        .read_text(encoding="utf-8")
    )


# The following function and statement are modified from
# https://github.com/python-jsonschema/jsonschema/blob/v4.17.3/docs/faq.rst#why-doesnt-my-schemas-default-property-set-the-default-on-my-instance
# on 2023-03-22 under MIT license
# https://github.com/python-jsonschema/jsonschema/blob/v4.17.3/COPYING .
def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])
        for error in validate_properties(
            validator,
            properties,
            instance,
            schema,
        ):
            yield error

    return jsonschema.validators.extend(
        validator_class,
        {"properties": set_defaults},
    )


DefaultFillingValidator = extend_with_default(jsonschema.Draft202012Validator)


def list_duplicates(iterable):
    """List only once all duplicate values in an iterable."""
    return [
        item
        for item, count in collections.Counter(iterable).items()
        if count > 1
    ]


def validate_output_filenames_unique(config):
    output_filenames = [vm["outputFilename"] for vm in config["vehicleModels"]]
    duplicate_filenames = list_duplicates(output_filenames)
    if duplicate_filenames:
        raise ValueError(
            "Every outputFilename must be unique. These filenames are "
            f"duplicated: {str(duplicate_filenames)}."
        )


def validate_minimum_counts_unique(config):
    for vm in config["vehicleModels"]:
        duplicate_minimum_counts = list_duplicates(
            vm["minimumCounts"].values()
        )
        if duplicate_minimum_counts:
            raise ValueError(
                "minimumCounts must have unique values. These counts repeat "
                f"for outputFilename {str(vm['outputFilename'])}: "
                f"{str(duplicate_minimum_counts)}."
            )


def validate_single_minimum_count_zero(config):
    for vm in config["vehicleModels"]:
        if 0 not in vm["minimumCounts"].values():
            raise ValueError(
                "One value in minimumCounts must be zero. Zero value is "
                f"missing for outputFilename {str(vm['outputFilename'])}."
            )


def validate_maximum_count_highest(config):
    for vm in config["vehicleModels"]:
        if vm["maximumCount"] < max(vm["minimumCounts"].values()):
            raise ValueError(
                "maximumCount must be at least as high as the maximum "
                "integer value in minimumCounts. maximumCount is too low for "
                f"outputFilename {str(vm['outputFilename'])}."
            )


def sort_dict_by_item_values(d):
    return dict(sorted(d.items(), key=lambda item: item[1]))


def order_minimum_counts(config):
    vehicle_models = []
    for vm in config["vehicleModels"]:
        vm["minimumCounts"] = sort_dict_by_item_values(vm["minimumCounts"])
        vehicle_models.append(vm)
    config["vehicleModels"] = vehicle_models
    return config


def find_matching_position(unique_vehicle_models, vm):
    index = None
    for unique_index, unique_vm in enumerate(unique_vehicle_models):
        if (
            unique_vm["minimumCounts"] == vm["minimumCounts"]
            and unique_vm["maximumCount"] == vm["maximumCount"]
        ):
            index = unique_index
            break
    return index


def keep_unique_vehicle_models(config):
    unique_vehicle_models = []
    for vm in config["vehicleModels"]:
        index = find_matching_position(unique_vehicle_models, vm)
        if index is None:
            vm["outputFilenames"] = [vm["outputFilename"]]
            del vm["outputFilename"]
            unique_vehicle_models.append(vm)
        else:
            unique_vehicle_models[index]["outputFilenames"].append(
                vm["outputFilename"]
            )
    config["vehicleModels"] = unique_vehicle_models
    return config


def reinforce_configuration(config):
    """Reinforce configuration.

    Fill in the defaults, validate the configuration, order the minimum counts
    and keep unique vehicle models. Lean on the JSON schema and check
    afterwards what is not expressed in the schema.
    """
    schema = yaml.safe_load(read_configuration_schema())
    DefaultFillingValidator(schema).validate(config)
    validate_output_filenames_unique(config)
    validate_minimum_counts_unique(config)
    validate_single_minimum_count_zero(config)
    validate_maximum_count_highest(config)
    config = order_minimum_counts(config)
    config = keep_unique_vehicle_models(config)
    return config


def read_configuration():
    """Read the configuration.

    The environment variable APC_ANONYMIZER_CONFIG_PATH defines the path to the
    configuration file. The default value is "./configuration.yaml".
    """
    # FIXME: Choose a sensible default.
    DEFAULT_CONFIG_PATH = "./configuration.yaml"
    config_path = os.environ.get(
        "APC_ANONYMIZER_CONFIG_PATH", DEFAULT_CONFIG_PATH
    )
    raw_config_text = pathlib.Path(config_path).read_text()
    config = yaml.safe_load(raw_config_text)
    reinforced_config = reinforce_configuration(config)
    return reinforced_config
