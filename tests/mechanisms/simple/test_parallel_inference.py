import csv
import pathlib
import tempfile

import optuna
import optuna.storages
from apc_anonymizer.mechanisms.simple import hyperparameter_optimization


def test_fast_simple_run(mocker):
    output_filename = "volvo-8908rle.csv"
    journal_filename = "journal.log"
    test_epsilon = 1e-5
    vehicle_model = {
        "outputFilenames": [output_filename],
        "minimumCounts": {
            "EMPTY": 0,
            "MANY_SEATS_AVAILABLE": 6,
            "FEW_SEATS_AVAILABLE": 36,
            "STANDING_ROOM_ONLY": 46,
            "CRUSHED_STANDING_ROOM_ONLY": 84,
            "FULL": 110,
        },
        "maximumCount": 126,
    }
    inference_config = {
        "mechanism": "simple",
        "options": {
            "epsilon": 1.0,
            "delta": 1e-3,
            "numberOfIterationsPerHyperparameterTrial": int(1e5),
            "minimumNumberOfHyperparameterTrialsPerProcess": 3,
            "minimumNumberOfHyperparameterTrials": 36,
            "numberOfProcesses": "one-per-core",
        },
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_filepath = tmp_dir + "/" + journal_filename

        def create_study(vm, db_name):
            storage = optuna.storages.JournalStorage(
                optuna.storages.JournalFileStorage(tmp_filepath)
            )
            study = optuna.create_study(
                study_name=vm["outputFilenames"][0],
                direction="minimize",
                storage=storage,
                load_if_exists=True,
            )
            return study

        mocker.patch(
            "apc_anonymizer.mechanisms.simple.hyperparameter_optimization.create_study",
            side_effect=create_study,
        )

        hyperparameter_optimization.run_inference_in_parallel(
            inference_config, vehicle_model, "hyperparameter_opt", tmp_dir
        )

        tmp_dir_path = pathlib.Path(tmp_dir)
        files_in_tmp_dir = set((f.name for f in tmp_dir_path.iterdir()))
        expected_files = set([output_filename, journal_filename])
        assert files_in_tmp_dir == expected_files

        with open(tmp_dir_path / output_filename, newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            previous_passenger_count = -1
            for row_index, row in enumerate(csv_reader):
                if row_index == 0:
                    assert row == [
                        "passenger_count",
                        "EMPTY",
                        "MANY_SEATS_AVAILABLE",
                        "FEW_SEATS_AVAILABLE",
                        "STANDING_ROOM_ONLY",
                        "CRUSHED_STANDING_ROOM_ONLY",
                        "FULL",
                    ]
                else:
                    passenger_count = int(row[0])
                    assert passenger_count == previous_passenger_count + 1
                    previous_passenger_count = passenger_count

                    floats = list(map(float, row[1:]))
                    assert len(floats) == 6
                    assert all(f >= 0 and f <= 1 for f in floats)
                    assert abs(sum(floats) - 1) < test_epsilon
