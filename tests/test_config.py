import jsonschema
import pytest
from apc_anonymizer import configuration
from apc_anonymizer import yaml_workaround as yaml


def test_schema():
    schema = None
    try:
        schema = configuration.read_configuration_schema()
    except BaseException:
        pytest.fail("Reading configuration schema failed")
    assert isinstance(schema, str)
    assert schema != ""


def test_invalid_configuration():
    config_string = """
    foo: bar
    """
    config = yaml.safe_load(config_string)
    with pytest.raises(jsonschema.exceptions.ValidationError):
        configuration.reinforce_configuration(config)


def test_minimum_configuration():
    config_string = """
    vehicleModels:
      - outputFilename: "foo"
        minimumCounts:
          EMPTY: 0
          FULL: 1
        maximumCount: 1
    """
    config = yaml.safe_load(config_string)
    try:
        configuration.reinforce_configuration(config)
    except BaseException:
        pytest.fail("Reinforcing minimum configuration failed")


def test_realistic_configuration():
    config_string = """---
    configurationVersion: "1-0-0"
    outputDirectory: "/output"
    vehicleModels:
      - outputFilename: "volvo-8908rle.csv"
        minimumCounts:
          EMPTY: 0
          MANY_SEATS_AVAILABLE: 6
          FEW_SEATS_AVAILABLE: 36
          STANDING_ROOM_ONLY: 46
          CRUSHED_STANDING_ROOM_ONLY: 84
          FULL: 110
        maximumCount: 126
      - outputFilename: "vdl-cites-lle-120-255.csv"
        minimumCounts:
          EMPTY: 0
          MANY_SEATS_AVAILABLE: 5
          FEW_SEATS_AVAILABLE: 28
          STANDING_ROOM_ONLY: 36
          CRUSHED_STANDING_ROOM_ONLY: 55
          FULL: 69
        maximumCount: 77
      - outputFilename: "scania-citywide-teli.csv"
        minimumCounts:
          EMPTY: 0
          MANY_SEATS_AVAILABLE: 6
          FEW_SEATS_AVAILABLE: 36
          STANDING_ROOM_ONLY: 46
          CRUSHED_STANDING_ROOM_ONLY: 79
          FULL: 103
        maximumCount: 117
    inference:
      mechanism: "simple"
      options:
        numberOfHyperparameterTrialsPerProcess: 20
        numberOfProcesses: "one-per-core"
    """
    config = yaml.safe_load(config_string)
    try:
        configuration.reinforce_configuration(config)
    except BaseException:
        pytest.fail("Reinforcing realistic configuration failed")


def test_defaults():
    config_string = """
    vehicleModels:
      - outputFilename: "foo"
        minimumCounts:
          EMPTY: 0
          FULL: 1
        maximumCount: 1
    """
    expected_config = {
        "configurationVersion": "1-0-0",
        "outputDirectory": "/output",
        "vehicleModels": [
            {
                "outputFilenames": ["foo"],
                "minimumCounts": {
                    "EMPTY": 0,
                    "FULL": 1,
                },
                "maximumCount": 1,
            }
        ],
        "inference": {
            "mechanism": "simple",
            "options": {
                "epsilon": 1.0,
                "delta": 1e-5,
                "numberOfIterationsPerHyperparameterTrial": int(1e7),
                "numberOfHyperparameterTrialsPerProcess": 50,
                "numberOfProcesses": "one-per-core",
            },
        },
    }
    config = yaml.safe_load(config_string)
    try:
        reinforced_config = configuration.reinforce_configuration(config)
    except BaseException:
        pytest.fail("Reinforcing default test configuration failed")
    assert reinforced_config == expected_config


def test_reordering_minimum_counts():
    config_string = """
    vehicleModels:
      - outputFilename: "foo"
        minimumCounts:
          FULL: 1
          EMPTY: 0
        maximumCount: 1
    """
    config = yaml.safe_load(config_string)
    given_list = list(config["vehicleModels"][0]["minimumCounts"].items())
    expected_list = list(
        {
            "EMPTY": 0,
            "FULL": 1,
        }.items()
    )
    try:
        reinforced_config = configuration.reinforce_configuration(config)
    except BaseException:
        pytest.fail(
            "Reinforcing reordering minimum counts test configuration failed"
        )
    reinforced_list = list(
        reinforced_config["vehicleModels"][0]["minimumCounts"].items()
    )
    assert reinforced_list != given_list
    assert reinforced_list == expected_list


def test_repeated_minimum_counts():
    config_string = """
    vehicleModels:
      - outputFilename: "foo"
        minimumCounts:
          EMPTY: 0
          FULL: 1
          FULL: 2
        maximumCount: 3
    """
    with pytest.raises(ValueError, match="Duplicate"):
        yaml.safe_load(config_string)


def test_keeping_unique_vehicle_models():
    config_string = """
    vehicleModels:
      - outputFilename: "foo"
        minimumCounts:
          EMPTY: 0
          FULL: 1
        maximumCount: 1
      - outputFilename: "bar"
        minimumCounts:
          EMPTY: 0
          FULL: 1
        maximumCount: 1
    """
    config = yaml.safe_load(config_string)
    try:
        reinforced_config = configuration.reinforce_configuration(config)
    except BaseException:
        pytest.fail(
            "Reinforcing keeping unique vehicle models test configuration "
            "failed"
        )
    assert len(reinforced_config["vehicleModels"]) == 1
    assert reinforced_config["vehicleModels"][0]["outputFilenames"] == [
        "foo",
        "bar",
    ]
