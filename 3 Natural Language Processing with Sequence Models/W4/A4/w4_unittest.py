import shutil
import pickle
import numpy as np
import random as rnd
import trax
from jax.interpreters.xla import DeviceArray
from trax.fastmath import numpy as jnp


def test_data_generator(target):

    successful_cases = 0
    failed_cases = []

    train_Q1_testing = np.array(
        [
            list(
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            ),
            list([32, 33, 4, 34, 6, 35, 36, 21]),
            list([32, 38, 4, 41, 11, 42, 43, 44, 45, 21]),
            list([30, 33, 49, 50, 51, 39, 52, 21]),
            list([30, 55, 56, 57, 58, 59, 60, 21]),
            list(
                [30, 61, 6, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 21]
            ),
            list([30, 16, 84, 64, 21]),
            list([86, 87, 88, 89, 90, 91, 92, 93, 17, 87, 94, 95, 72, 96, 21]),
            list([86, 38, 102, 103, 104, 21]),
            list([32, 37, 4, 107, 65, 108, 109, 110, 21]),
        ],
        dtype=object,
    )

    train_Q2_testing = np.array(
        [
            list(
                [
                    4,
                    22,
                    6,
                    23,
                    7,
                    24,
                    8,
                    25,
                    26,
                    11,
                    27,
                    28,
                    7,
                    29,
                    30,
                    16,
                    31,
                    18,
                    19,
                    20,
                    21,
                ]
            ),
            list([30, 37, 4, 38, 39, 34, 6, 40, 36, 21]),
            list([32, 33, 4, 46, 47, 43, 48, 45, 21]),
            list([32, 33, 53, 49, 54, 51, 39, 52, 21]),
            list([30, 55, 56, 57, 58, 59, 21]),
            list(
                [32, 76, 6, 62, 63, 77, 78, 71, 79, 28, 80, 81, 82, 39, 83, 28, 80, 21]
            ),
            list([30, 16, 84, 85, 21]),
            list([86, 38, 97, 98, 90, 93, 99, 33, 34, 95, 100, 101, 96, 21]),
            list([86, 87, 102, 11, 105, 106, 104, 21]),
            list([32, 111, 37, 112, 17, 113, 114, 107, 65, 108, 109, 115, 21]),
        ],
        dtype=object,
    )

    test_cases = [
        {
            "name": "check_batch_size_3",
            "input": {
                "Q1": train_Q1_testing,
                "Q2": train_Q2_testing,
                "batch_size": 3,
                "pad": 1,
                "shuffle": False,
            },
            "expected": {
                "output1": np.array(
                    [
                        [
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                            19,
                            20,
                            21,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                        ],
                        [
                            32,
                            33,
                            4,
                            34,
                            6,
                            35,
                            36,
                            21,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                        ],
                        [
                            32,
                            38,
                            4,
                            41,
                            11,
                            42,
                            43,
                            44,
                            45,
                            21,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                        ],
                    ]
                ),
                "output2": np.array(
                    [
                        [
                            4,
                            22,
                            6,
                            23,
                            7,
                            24,
                            8,
                            25,
                            26,
                            11,
                            27,
                            28,
                            7,
                            29,
                            30,
                            16,
                            31,
                            18,
                            19,
                            20,
                            21,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                        ],
                        [
                            30,
                            37,
                            4,
                            38,
                            39,
                            34,
                            6,
                            40,
                            36,
                            21,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                        ],
                        [
                            32,
                            33,
                            4,
                            46,
                            47,
                            43,
                            48,
                            45,
                            21,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                        ],
                    ]
                ),
            },
        },
        {
            "name": "check_batch_size_5",
            "input": {
                "Q1": train_Q1_testing,
                "Q2": train_Q2_testing,
                "batch_size": 5,
                "pad": -1,
                "shuffle": True,
            },
            "expected": {
                "output1": np.array(
                    [
                        [
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                            19,
                            20,
                            21,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
                        [
                            30,
                            55,
                            56,
                            57,
                            58,
                            59,
                            60,
                            21,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
                        [
                            30,
                            61,
                            6,
                            62,
                            63,
                            64,
                            65,
                            66,
                            67,
                            68,
                            69,
                            70,
                            71,
                            72,
                            73,
                            74,
                            75,
                            21,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
                        [
                            32,
                            33,
                            4,
                            34,
                            6,
                            35,
                            36,
                            21,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
                        [
                            86,
                            87,
                            88,
                            89,
                            90,
                            91,
                            92,
                            93,
                            17,
                            87,
                            94,
                            95,
                            72,
                            96,
                            21,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
                    ]
                ),
                "output2": np.array(
                    [
                        [
                            4,
                            22,
                            6,
                            23,
                            7,
                            24,
                            8,
                            25,
                            26,
                            11,
                            27,
                            28,
                            7,
                            29,
                            30,
                            16,
                            31,
                            18,
                            19,
                            20,
                            21,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
                        [
                            30,
                            55,
                            56,
                            57,
                            58,
                            59,
                            21,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
                        [
                            32,
                            76,
                            6,
                            62,
                            63,
                            77,
                            78,
                            71,
                            79,
                            28,
                            80,
                            81,
                            82,
                            39,
                            83,
                            28,
                            80,
                            21,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
                        [
                            30,
                            37,
                            4,
                            38,
                            39,
                            34,
                            6,
                            40,
                            36,
                            21,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
                        [
                            86,
                            38,
                            97,
                            98,
                            90,
                            93,
                            99,
                            33,
                            34,
                            95,
                            100,
                            101,
                            96,
                            21,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                        ],
                    ]
                ),
            },
        },
    ]

    for test_case in test_cases:
        if test_case["name"] == "check_batch_size_5":
            rnd.seed(33)

        test_generator = target(**test_case["input"])
        res1, res2 = next(test_generator)

        try:
            assert res1.shape == test_case["expected"]["output1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["output1"].shape,
                    "got": res1.shape,
                }
            )
            print(
                f"Output for questions in batch 1 has the wrong size.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

        try:
            assert res2.shape == test_case["expected"]["output2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["output1"].shape,
                    "got": res2.shape,
                }
            )
            print(
                f"Output for questions in batch 2 has the wrong size.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(res1, test_case["expected"]["output1"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["output1"],
                    "got": res1,
                }
            )
            print(
                f"Wrong output for questions in batch 1.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(res2, test_case["expected"]["output2"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["output2"],
                    "got": res2,
                }
            )
            print(
                f"Wrong output for questions in batch 2.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_Siamese(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_input_check",
            "input": {"vocab_size": 41699, "d_model": 128, "mode": "train"},
            "expected": {
                "expected_str": "Parallel_in2_out2[\n  Serial[\n    Embedding_41699_128\n    LSTM_128\n    Mean\n    Normalize\n  ]\n  Serial[\n    Embedding_41699_128\n    LSTM_128\n    Mean\n    Normalize\n  ]\n]",
                "expected_type": trax.layers.combinators.Parallel,
                "expected_sublayers_type": [
                    trax.layers.combinators.Serial,
                    trax.layers.combinators.Serial,
                ],
                "expected_sublayer0_type": [
                    trax.layers.core.Embedding,
                    trax.layers.combinators.Serial,
                    trax.layers.base.PureLayer,
                    trax.layers.base.PureLayer,
                ],
                "expected_sublayer1_type": [
                    trax.layers.core.Embedding,
                    trax.layers.combinators.Serial,
                    trax.layers.base.PureLayer,
                    trax.layers.base.PureLayer,
                ],
            },
        },
        {
            "name": "small_input_check",
            "input": {"vocab_size": 200, "d_model": 16, "mode": "train"},
            "expected": {
                "expected_str": "Parallel_in2_out2[\n  Serial[\n    Embedding_200_16\n    LSTM_16\n    Mean\n    Normalize\n  ]\n  Serial[\n    Embedding_200_16\n    LSTM_16\n    Mean\n    Normalize\n  ]\n]",
                "expected_type": trax.layers.combinators.Parallel,
                "expected_sublayers_type": [
                    trax.layers.combinators.Serial,
                    trax.layers.combinators.Serial,
                ],
                "expected_sublayer0_type": [
                    trax.layers.core.Embedding,
                    trax.layers.combinators.Serial,
                    trax.layers.base.PureLayer,
                    trax.layers.base.PureLayer,
                ],
                "expected_sublayer1_type": [
                    trax.layers.core.Embedding,
                    trax.layers.combinators.Serial,
                    trax.layers.base.PureLayer,
                    trax.layers.base.PureLayer,
                ],
            },
        },
    ]

    for test_case in test_cases:
        model = target(**test_case["input"])
        description = str(model)

        try:
            assert description.replace(" ", "") == test_case["expected"][
                "expected_str"
            ].replace(" ", "")
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_str"],
                    "got": description,
                }
            )
            print(
                f"Wrong model.\n\tExpected: {failed_cases[-1].get('expected')}. \n\tGot: {failed_cases[-1].get('got')}."
            )

        # Test the output type
        try:
            assert isinstance(model, test_case["expected"]["expected_type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_type"],
                    "got": type(model),
                }
            )
            print(
                f"Model has the wrong type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        sublayers_type = lambda x: list(map(type, x.sublayers))
        model_sublayers_type = sublayers_type(model)
        try:
            for i in range(len(test_case["expected"]["expected_sublayers_type"])):
                assert str(model_sublayers_type[i]) == str(
                    test_case["expected"]["expected_sublayers_type"][i]
                )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "sublayers_type_check",
                    "expected": test_case["expected"]["expected_sublayers_type"],
                    "got": model_sublayers_type,
                }
            )
            print(
                f"Model sublayers do not have the correct type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

        try:
            model_sublayer0_type = sublayers_type(model.sublayers[0])
            for i in range(len(test_case["expected"]["expected_sublayer0_type"])):
                assert str(model_sublayer0_type[i]) == str(
                    test_case["expected"]["expected_sublayer0_type"][i]
                )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "sublayer0_type_check",
                    "expected": test_case["expected"]["expected_sublayer0_type"],
                    "got": model_sublayer0_type,
                }
            )
            print(
                f"Sublayers in layer 0 do not have the correct type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

        try:
            model_sublayer1_type = sublayers_type(model.sublayers[1])
            for i in range(len(test_case["expected"]["expected_sublayer1_type"])):
                assert str(model_sublayer1_type[i]) == str(
                    test_case["expected"]["expected_sublayer1_type"][i]
                )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "sublayer1_type_check",
                    "expected": test_case["expected"]["expected_sublayer1_type"],
                    "got": model_sublayer1_type,
                }
            )
            print(
                f"Sublayers in layer 1 do not have the correct type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_TripletLossFn(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "v1": np.array(
                    [
                        [0.26726124, 0.53452248, 0.80178373],
                        [-0.5178918, -0.57543534, -0.63297887],
                    ]
                ),
                "v2": np.array(
                    [
                        [0.26726124, 0.53452248, 0.80178373],
                        [0.5178918, 0.57543534, 0.63297887],
                    ]
                ),
            },
            "expected": 0.703507661819458,  # 0.5,
        },
        {
            "name": "check_small_float_arr",
            "input": {
                "v1": np.array(
                    [
                        [0.26726124, 0.53452248, 0.80178373],
                        [0.64616234, 0.57436653, 0.50257071],
                        [-0.21821789, -0.87287156, -0.43643578],
                        [0.13608276, -0.95257934, 0.27216553],
                    ]
                ),
                "v2": np.array(
                    [
                        [0.32929278, 0.5488213, 0.76834982],
                        [0.64231723, 0.57470489, 0.50709255],
                        [-0.20313388, -0.8802468, -0.42883819],
                        [0.09298683, -0.96971978, 0.22582515],
                    ]
                ),
            },
            "expected": 0.30219173431396484,  # 1.4382333,
        },
        {
            "name": "check_small_float_arr_margin",
            "input": {
                "v1": np.array(
                    [
                        [0.26726124, 0.53452248, 0.80178373],
                        [0.64616234, 0.57436653, 0.50257071],
                        [-0.21821789, -0.87287156, -0.43643578],
                        [0.13608276, -0.95257934, 0.27216553],
                    ]
                ),
                "v2": np.array(
                    [
                        [0.32929278, 0.5488213, 0.76834982],
                        [0.64231723, 0.57470489, 0.50709255],
                        [-0.20313388, -0.8802468, -0.42883819],
                        [0.09298683, -0.96971978, 0.22582515],
                    ]
                ),
                "margin": 0.8,
            },
            "expected": 2.4262490272521973,  # 2.1715667,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, DeviceArray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": DeviceArray,
                    "got": type(result),
                }
            )
            print(
                f"Output has the wrong type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
            )

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            if (
                test_case["name"] == "check_small_float_arr_margin"
                or test_case["name"] == "check_random_arr_margin"
            ):
                print(
                    f"Wrong output. Take a look to the use of the margin variable.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}"
                )
            else:
                print(
                    f"Wrong output.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}"
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_train_model(target, model, loss, data_generator):
    successful_cases = 0
    failed_cases = []

    train_steps = 0
    output_dir = "/tmp/model"

    try:
        shutil.rmtree(output_dir)
    except OSError as e:
        pass

    train_Q1_testing = np.array(
        [
            list(
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            ),
            list([32, 33, 4, 34, 6, 35, 36, 21]),
            list([32, 38, 4, 41, 11, 42, 43, 44, 45, 21]),
            list([30, 33, 49, 50, 51, 39, 52, 21]),
            list([30, 55, 56, 57, 58, 59, 60, 21]),
            list(
                [30, 61, 6, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 21]
            ),
            list([30, 16, 84, 64, 21]),
            list([86, 87, 88, 89, 90, 91, 92, 93, 17, 87, 94, 95, 72, 96, 21]),
            list([86, 38, 102, 103, 104, 21]),
            list([32, 37, 4, 107, 65, 108, 109, 110, 21]),
        ],
        dtype=object,
    )

    train_Q2_testing = np.array(
        [
            list(
                [
                    4,
                    22,
                    6,
                    23,
                    7,
                    24,
                    8,
                    25,
                    26,
                    11,
                    27,
                    28,
                    7,
                    29,
                    30,
                    16,
                    31,
                    18,
                    19,
                    20,
                    21,
                ]
            ),
            list([30, 37, 4, 38, 39, 34, 6, 40, 36, 21]),
            list([32, 33, 4, 46, 47, 43, 48, 45, 21]),
            list([32, 33, 53, 49, 54, 51, 39, 52, 21]),
            list([30, 55, 56, 57, 58, 59, 21]),
            list(
                [32, 76, 6, 62, 63, 77, 78, 71, 79, 28, 80, 81, 82, 39, 83, 28, 80, 21]
            ),
            list([30, 16, 84, 85, 21]),
            list([86, 38, 97, 98, 90, 93, 99, 33, 34, 95, 100, 101, 96, 21]),
            list([86, 87, 102, 11, 105, 106, 104, 21]),
            list([32, 111, 37, 112, 17, 113, 114, 107, 65, 108, 109, 115, 21]),
        ],
        dtype=object,
    )

    # Create simple data generator
    tr_generator = data_generator(
        Q1=train_Q1_testing, Q2=train_Q2_testing, batch_size=2, pad=1, shuffle=False
    )

    # Create simple data generator
    ev_generator = data_generator(
        Q1=train_Q1_testing, Q2=train_Q2_testing, batch_size=2, pad=1, shuffle=False
    )

    trainer = target(model, loss, tr_generator, ev_generator, output_dir=output_dir)

    # Test the output type
    try:
        assert isinstance(trainer, trax.supervised.training.Loop)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "trainer_type",
                "expected": trax.supervised.training.Loop,
                "got": type(trainer),
            }
        )
        print(
            f"Wrong type for the training object.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
        )

    # Test correct model loss_fn
    loss_fn = "TripletLoss_in2"
    description = str(trainer._tasks[0].loss_layer)

    try:
        assert description == loss_fn
        successful_cases += 1
    except:
        failed_cases.append(
            {"name": "loss_fn_check", "expected": loss_fn, "got": description}
        )
        print(
            f"Wrong loss function.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
        )

    # Test the optimizer parameter
    try:
        assert isinstance(trainer._tasks[0].optimizer, trax.optimizers.adam.Adam)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "optimizer_check",
                "expected": trax.optimizers.adam.Adam,
                "got": type(trainer._tasks[0].optimizer),
            }
        )
        print(
            f"Wrong optimizer. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
        )

    opt_params_dict = {
        "weight_decay_rate": jnp.array(1.0e-5),
        "b1": jnp.array(0.9),
        "b2": jnp.array(0.999),
        "eps": jnp.array(1.0e-5),
        "learning_rate": jnp.array(0.01),
    }

    try:
        assert trainer._tasks[0]._optimizer.opt_params == opt_params_dict
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "optimizer_parameters_check",
                "expected": opt_params_dict,
                "got": trainer._tasks[0]._optimizer.opt_params,
            }
        )

    # Test the metrics in the evaluation task
    test_func = lambda x: list(map(str, x._eval_tasks[0]._metric_names))

    try:
        assert test_func(trainer) == ["TripletLoss"]
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "metrics_check",
                "expected": ["TripletLoss"],
                "got": test_func(trainer),
            }
        )
        print(
            f"Wrong metrics in evaluations task. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
        )

    # Test correct output_dir
    try:
        assert trainer._output_dir == output_dir
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "output_dir_check",
                "expected": output_dir,
                "got": trainer._output_dir,
            }
        )
        print("Wrong output dir.")

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


class model_mock:
    """Class that represents a mock of the model.     


    Attributes:    
        path_test_files (str): path of directory that contains .pkl files.    

    Methods:    
        read_path_test_files(): Reads the files in .pkl format with 
            the actual input/output mapping       
    """

    def __init__(self, path_test_files):
        self.path_test_files = path_test_files
        self.dict_in_out_map = self.read_path_test_files()

    def read_path_test_files(self):
        """Reads files in .pkl format.
                
        Returns:
            dict: Dictionary that maps the input and output directly.
        """

        dict_raw_output = pickle.load(open(self.path_test_files, "rb"))

        return dict_raw_output

    def mocked_fn(self, input_tuple):
        """Returns the input/output mapping using the dictionary that 
        was read in read_path_test_files().

        Args:
            NMTAttn (tl.Serial): Instantiated model. This parameter is not actually used but 
                is left as the learner implementation requires it.
            input_tokens (np.ndarray 1 x n_tokens): tokenized representation of the input sentence
            cur_output_tokens (list): tokenized representation of previously translated words
            temperature (float): parameter for sampling ranging from 0.0 to 1.0. This parameter 
                is not actually used but is left as the learner implementation requires it.
            
            vocab_file (str): filename of the vocabulary
            vocab_dir (str): path to the vocabulary file

        Returns:
            tuple: (int, float)
                int: index of the next token in the translated sentence
                float: log probability of the next symbol
        """
        input_q1 = input_tuple[0]
        input_q2 = input_tuple[1]

        return self.dict_in_out_map.get(
            (tuple(map(tuple, input_q1)), tuple(map(tuple, input_q2)))
        )


def test_classify(target, vocab, data_generator):
    successful_cases = 0
    failed_cases = []

    with open("./support_files/classify_fn/Q1_test.pkl", "rb") as f:
        Q1_test = pickle.load(f)

    with open("./support_files/classify_fn/Q2_test.pkl", "rb") as f:
        Q2_test = pickle.load(f)

    with open("./support_files/classify_fn/y_test.pkl", "rb") as f:
        y_test = pickle.load(f)

    test_cases = [
        {
            "name": "default_example_check",
            "input": {
                "test_Q1": Q1_test,
                "test_Q2": Q2_test,
                "y": y_test,
                "threshold": 0.7,
                "model": model_mock(
                    "./support_files/classify_fn/accuracy_metric_batch512.pkl"
                ).mocked_fn,
                "vocab": vocab,
                "data_generator": data_generator,
                "batch_size": 512,
            },
            "expected": 0.74423828125,
        },
        {
            "name": "default_example_check_threshold",
            "input": {
                "test_Q1": Q1_test,
                "test_Q2": Q2_test,
                "y": y_test,
                "threshold": 0.75,
                "model": model_mock(
                    "./support_files/classify_fn/accuracy_metric_batch512.pkl"
                ).mocked_fn,
                "vocab": vocab,
                "data_generator": data_generator,
                "batch_size": 512,
            },
            "expected": 0.74951171875,
        },
        {
            "name": "small_batch_check",
            "input": {
                "test_Q1": Q1_test,
                "test_Q2": Q2_test,
                "y": y_test,
                "threshold": 0.7,
                "model": model_mock(
                    "./support_files/classify_fn/accuracy_metric_batch256.pkl"
                ).mocked_fn,
                "vocab": vocab,
                "data_generator": data_generator,
                "batch_size": 256,
            },
            "expected": 0.74453125,
        },
        {
            "name": "small_batch_check_threshold",
            "input": {
                "test_Q1": Q1_test,
                "test_Q2": Q2_test,
                "y": y_test,
                "threshold": 0.8,
                "model": model_mock(
                    "./support_files/classify_fn/accuracy_metric_batch256.pkl"
                ).mocked_fn,
                "vocab": vocab,
                "data_generator": data_generator,
                "batch_size": 256,
            },
            "expected": 0.7384765625,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                },
            )
            print(
                f"{test_case['name']} Wrong output for accuracy metric.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_predict(target, vocab, data_generator):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_example1",
            "input": {
                "question1": "When will I see you?",
                "question2": "When can I see you again?",
                "threshold": 0.7,
                "model": model_mock(
                    "./support_files/predict_fn/see_you_again.pkl"
                ).mocked_fn,
                "vocab": vocab,
            },
            "expected": True,
        },
        {
            "name": "default_example2",
            "input": {
                "question1": "Do they enjoy eating the dessert?",
                "question2": "Do they like hiking in the desert?",
                "threshold": 0.7,
                "model": model_mock(
                    "./support_files/predict_fn/dessert_desert.pkl"
                ).mocked_fn,
                "vocab": vocab,
            },
            "expected": False,
        },
        {
            "name": "check_threshold_low",
            "input": {
                "question1": "How does a long distance relationship work?",
                "question2": "How are long distance relationships maintained?",
                "threshold": 0.5,
                "model": model_mock(
                    "./support_files/predict_fn/relationships.pkl"
                ).mocked_fn,
                "vocab": vocab,
            },
            "expected": True,
        },
        {
            "name": "check_threshold_high",
            "input": {
                "question1": "How does a long distance relationship work?",
                "question2": "How are long distance relationships maintained?",
                "threshold": 0.8,
                "model": model_mock(
                    "./support_files/predict_fn/relationships.pkl"
                ).mocked_fn,
                "vocab": vocab,
            },
            "expected": True,
        },
        {
            "name": "check_false",
            "input": {
                "question1": "Why don't we still do great music like in the 70's and 80's?",
                "question2": "Should I raise my young child on 80's music?",
                "threshold": 0.5,
                "model": model_mock(
                    "./support_files/predict_fn/old_music.pkl"
                ).mocked_fn,
                "vocab": vocab,
            },
            "expected": False,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result == test_case["expected"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"{test_case['name']} Wrong output for prediction.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases

