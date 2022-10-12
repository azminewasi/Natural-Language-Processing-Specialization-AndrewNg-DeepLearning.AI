# -*- coding: utf-8 -*-
import os
import jax
import trax
import pickle
import numpy
import random as rnd
from trax import fastmath
from trax import layers as tl
from trax.fastmath import numpy as jnp


def test_line_to_tensor(target):
    test_cases = [
        {
            "name": "simple_test_case1",
            "input": {"line": "abc xyz", "EOS_int": 1},
            "expected": [97, 98, 99, 32, 120, 121, 122, 1],
        },
        {
            "name": "simple_test_case2",
            "input": {"line": "abc xyz", "EOS_int": -1},
            "expected": [97, 98, 99, 32, 120, 121, 122, -1],
        },
        {
            "name": "simple_test_case3",
            "input": {"line": "hello world!"},
            "expected": [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33, 1],
        },
        {
            "name": "simple_test_case4",
            "input": {"line": "12345.", "EOS_int": 1},
            "expected": [49, 50, 51, 52, 53, 46, 1],
        },
        {"name": "simple_test_case5", "input": {"line": ""}, "expected": [1]},
        {
            "name": "simple_test_case6",
            "input": {"line": "", "EOS_int": -10},
            "expected": [-10],
        },
        {
            "name": "simple_test_case6",
            "input": {"line": " ", "EOS_int": 1},
            "expected": [32, 1],
        },
    ]

    successful_cases = 0
    failed_cases = []

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, list)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]),
                    "got": type(result),
                }
            )
            print(
                f"Output from line_to_tensor is not of type list.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
            )

        try:
            assert len(result) == len(test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(test_case["expected"]),
                    "got": len(result),
                }
            )
            print(
                f"Output from line_to_tensor has not the correct number of elements.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
            )

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
                f"Output from line_to_tensor has not the correct values.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_data_generator(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "check_default_example",
            "input": {
                "batch_size": 2,
                "max_length": 10,
                "data_lines": ["12345678901", "123456789", "234567890", "345678901"],
                "shuffle": False,
            },
            "expected": (
                jnp.array(
                    [
                        [49, 50, 51, 52, 53, 54, 55, 56, 57, 1],
                        [50, 51, 52, 53, 54, 55, 56, 57, 48, 1],
                    ]
                ),
                jnp.array(
                    [
                        [49, 50, 51, 52, 53, 54, 55, 56, 57, 1],
                        [50, 51, 52, 53, 54, 55, 56, 57, 48, 1],
                    ]
                ),
                jnp.array(
                    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
                ),
            ),
        },
        {
            "name": "check_lenghts_batchsize",
            "input": {
                "batch_size": 4,
                "max_length": 11,
                "data_lines": [
                    "0123456",
                    "01234567890",
                    "123456789",
                    "345678901",
                    "5678901",
                    "234567890",
                ],
                "shuffle": False,
            },
            "expected": (
                jnp.array(
                    [
                        [48, 49, 50, 51, 52, 53, 54, 1, 0, 0, 0],
                        [49, 50, 51, 52, 53, 54, 55, 56, 57, 1, 0],
                        [51, 52, 53, 54, 55, 56, 57, 48, 49, 1, 0],
                        [53, 54, 55, 56, 57, 48, 49, 1, 0, 0, 0],
                    ]
                ),
                jnp.array(
                    [
                        [48, 49, 50, 51, 52, 53, 54, 1, 0, 0, 0],
                        [49, 50, 51, 52, 53, 54, 55, 56, 57, 1, 0],
                        [51, 52, 53, 54, 55, 56, 57, 48, 49, 1, 0],
                        [53, 54, 55, 56, 57, 48, 49, 1, 0, 0, 0],
                    ]
                ),
                jnp.array(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    ]
                ),
            ),
        },
        {
            "name": "check_shuffle",
            "input": {
                "batch_size": 3,
                "max_length": 10,
                "data_lines": [
                    "0123456",
                    "01234567890",
                    "123456789",
                    "345678901",
                    "5678901",
                    "234567890",
                ],
                "shuffle": True,
            },
            "expected": (
                jnp.array(
                    [
                        [49, 50, 51, 52, 53, 54, 55, 56, 57, 1],
                        [50, 51, 52, 53, 54, 55, 56, 57, 48, 1],
                        [51, 52, 53, 54, 55, 56, 57, 48, 49, 1],
                    ]
                ),
                jnp.array(
                    [
                        [49, 50, 51, 52, 53, 54, 55, 56, 57, 1],
                        [50, 51, 52, 53, 54, 55, 56, 57, 48, 1],
                        [51, 52, 53, 54, 55, 56, 57, 48, 49, 1],
                    ]
                ),
                jnp.array(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ]
                ),
            ),
        },
    ]

    for test_case in test_cases:

        if test_case["name"] == "check_shuffle":
            rnd.seed(32)

        tmp_data_gen = target(**test_case["input"])

        tmp_batch = next(tmp_data_gen)

        try:
            assert len(tmp_batch) == 3
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "data_generator_output_size",
                    "expected": 3,
                    "got": len(tmp_batch),
                }
            )
            print(f"Your data generator is yielding more than 3 objects.")

        # Testing types
        for index, elem_batch in enumerate(tmp_batch):
            try:
                assert isinstance(elem_batch, jax.interpreters.xla.DeviceArray)
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": "data_generator_output_types",
                        "expected": jax.interpreters.xla.DeviceArray,
                        "got": type(elem_batch),
                    }
                )
                print(
                    f"Element with index {index} in the output tuple has incorrect type.\n\t Expected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
                )

        # Testing shapes
        for index, elem_batch in enumerate(tmp_batch):
            try:
                assert elem_batch.shape == (
                    test_case["input"]["batch_size"],
                    test_case["input"]["max_length"],
                )
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": "data_generator_output_shapes",
                        "expected": (
                            test_case["input"]["batch_size"],
                            test_case["input"]["max_length"],
                        ),
                        "got": elem_batch.shape,
                    }
                )
                print(
                    f"Element with index {index} in the output tuple has incorrect shape. It should be (batch_size, max_length).\n\t Expected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
                )

        # Testing values
        for index, elem_batch in enumerate(tmp_batch):
            try:
                assert jnp.allclose(elem_batch, test_case["expected"][index])
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": "data_generator_output_values",
                        "expected": test_case["expected"][index],
                        "got": elem_batch,
                    }
                )
                print(
                    f"Element with index {index} in the output tuple has incorrect shape. It should be (batch_size, max_length).\n\t Expected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_GRULM(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "check_default_model",
            "input": {
                "vocab_size": 256,
                "d_model": 512,
                "n_layers": 2,
                "mode": "train",
            },
            "expected": {
                "expected_str": "Serial[\n  Serial[\n    ShiftRight(1)\n  ]\n  Embedding_256_512\n  GRU_512\n  GRU_512\n  Dense_256\n  LogSoftmax\n]",
                "expected_sublayers_types": [
                    trax.layers.combinators.Serial,
                    trax.layers.core.Embedding,
                    trax.layers.combinators.Serial,
                    trax.layers.combinators.Serial,
                    trax.layers.core.Dense,
                    trax.layers.base.PureLayer,
                ],
                "expected_type": trax.layers.combinators.Serial,
            },
        },
        {
            "name": "check_small_layer",
            "input": {"vocab_size": 128, "d_model": 8, "n_layers": 3, "mode": "train",},
            "expected": {
                "expected_str": "Serial[\n  Serial[\n    ShiftRight(1)\n  ]\n  Embedding_128_8\n  GRU_8\n  GRU_8\n  GRU_8\n  Dense_128\n  LogSoftmax\n]",
                "expected_sublayers_types": [
                    trax.layers.combinators.Serial,
                    trax.layers.core.Embedding,
                    trax.layers.combinators.Serial,
                    trax.layers.combinators.Serial,
                    trax.layers.combinators.Serial,
                    trax.layers.core.Dense,
                    trax.layers.base.PureLayer,
                ],
                "expected_type": trax.layers.combinators.Serial,
            },
        },
    ]

    for test_case in test_cases:
        # print(test_case["name"])
        classifier = target(**test_case["input"])
        proposed = str(classifier)

        try:
            assert proposed.replace(" ", "") == test_case["expected"][
                "expected_str"
            ].replace(" ", "")
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "str_rep_check",
                    "expected": test_case["expected"]["expected_str"],
                    "got": proposed,
                }
            )

            print(
                f"Wrong model.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected:\n%s {failed_cases[-1].get('expected')}"
            )

        # Test the output type
        try:
            assert isinstance(classifier, test_case["expected"]["expected_type"])
            successful_cases += 1
            # Test the number of layers
        except:
            failed_cases.append(
                {
                    "name": "object_type_check",
                    "expected": test_case["expected"]["expected_type"],
                    "got": type(classifier),
                }
            )
            print(
                "The classifier is not an object of type",
                test_case["expected"]["expected_type"],
                "Got: ",
                failed_cases[-1].get("got"),
            )

        try:
            # Test
            assert len(classifier.sublayers) == len(
                test_case["expected"]["expected_sublayers_types"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "num_layers_check",
                    "expected": len(test_case["expected"]["expected_sublayers_types"]),
                    "got": len(classifier.sublayers),
                }
            )
            print(
                f"The number of sublayers does not match.\n\tGot: {len(classifier.sublayers)}.\n\tExpected: {len(test_case['expected']['expected_sublayers_types'])}"
            )

        try:
            sublayers_type = lambda x: list(map(type, x.sublayers))
            classifier_sublayers_type = sublayers_type(classifier)

            for i in range(len(test_case["expected"]["expected_sublayers_types"])):
                assert str(classifier_sublayers_type[i]) == str(
                    test_case["expected"]["expected_sublayers_types"][i]
                )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "sublayers_type_check",
                    "expected": test_case["expected"]["expected_sublayers_types"],
                    "got": classifier_sublayers_type,
                }
            )
            print(
                f"Classifier sublayers do not have the correct type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_train_model(target, model, data_generator):
    successful_cases = 0
    failed_cases = []

    input_lines = [
        "a lover's complaint",
        "from off a hill whose concave womb reworded",
        "a plaintful story from a sistering vale,",
        "my spirits to attend this double voice accorded,",
        "and down i laid to list the sad-tuned tale;",
        "ere long espied a fickle maid full pale,",
        "tearing of papers, breaking rings a-twain,",
        "storming her world with sorrow's wind and rain.",
        "upon her head a platted hive of straw,",
        "which fortified her visage from the sun,",
        "whereon the thought might think sometime it saw",
        "the carcass of beauty spent and done:",
        "time had not scythed all that youth begun,",
        "nor youth all quit; but, spite of heaven's fell rage,",
        "some beauty peep'd through lattice of sear'd age.",
        "oft did she heave her napkin to her eyne,",
    ]

    input_eval_lines = [
        "[exeunt jessica and lorenzo]",
        "now, balthasar,",
        "as i have ever found thee honest-true,",
        "so let me find thee still. take this same letter,",
        "and use thou all the endeavour of a man",
        "in speed to padua: see thou render this",
        "into my cousin's hand, doctor bellario;",
        "and, look, what notes and garments he doth give thee,",
        "bring them, i pray thee, with imagined speed",
        "unto the tranect, to the common ferry",
        "which trades to venice. waste no time in words,",
        "but get thee gone: i shall be there before thee.",
        "balthasar\tmadam, i go with all convenient speed.",
        "[exit]",
        "portia\tcome on, nerissa; i have work in hand",
        "that you yet know not of: we'll see our husbands",
    ]

    output_loop = target(
        model,
        data_generator,
        batch_size=8,
        max_length=32,
        lines=input_lines,
        eval_lines=input_eval_lines,
        n_steps=1,
        output_dir="test_model/",
    )

    try:
        assert isinstance(output_loop, trax.supervised.training.Loop)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "check_loop_type",
                "expected": trax.supervised.training.Loop,
                "got": type(output_loop),
            }
        )
        print(f"Training object has the wrong type. Got {failed_cases[-1].get('got')}.")

    try:
        strlabel = str(output_loop._tasks[0]._loss_layer)
        assert strlabel == "CrossEntropyLoss_in3"
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "loss_layer_check",
                "expected": "CrossEntropyLoss_in3",
                "got": strlabel,
            }
        )
        print(
            f"Wrong loss functions. CrossEntropyLoss_in3 was expected. Got {failed_cases[-1].get('got')}."
        )

    opt_params_dict = {
        "weight_decay_rate": jnp.array(1.0e-5),
        "b1": jnp.array(0.9),
        "b2": jnp.array(0.999),
        "eps": jnp.array(1.0e-5),
        "learning_rate": jnp.array(0.0005),
    }

    try:
        assert output_loop._tasks[0]._optimizer.opt_params == opt_params_dict
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "optimizer_parameters_check",
                "expected": opt_params_dict,
                "got": output_loop._tasks[0]._optimizer.opt_params,
            }
        )
        print(
            f"Wrong optimizer parameters. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
        )

    # Test the optimizer parameter
    try:
        assert isinstance(output_loop._tasks[0].optimizer, trax.optimizers.adam.Adam)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "optimizer_check",
                "expected": trax.optimizers.adam.Adam,
                "got": type(output_loop._tasks[0].optimizer),
            }
        )
        print(
            f"Wrong optimizer. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
        )

    # Test the metrics in the evaluation task
    test_func = lambda x: list(map(str, x._eval_tasks[0]._metric_names))

    try:
        assert test_func(output_loop) == ["CrossEntropyLoss", "Accuracy"]
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "metrics_check",
                "expected": ["CrossEntropyLoss", "Accuracy"],
                "got": test_func(output_loop),
            }
        )
        print(
            f"Wrong metrics in evaluations task. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
        )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def bk2_unittest_test_model(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "check_default_example",
            "input": {
                "preds": pickle.load(
                    open("./test_support_files/test_preds_1.pkl", "rb")
                ),
                "target": pickle.load(
                    open("./test_support_files/test_batch_1.pkl", "rb")
                )[1],
            },
            "expected": 1.7646706,
        },  # 1.7646706 5.8396482
        {
            "name": "check_example_2",
            "input": {
                "preds": pickle.load(
                    open("./test_support_files/test_preds_2.pkl", "rb")
                ),
                "target": pickle.load(
                    open("./test_support_files/test_batch_2.pkl", "rb")
                )[1],
            },
            "expected": 1.5336857,
        },  #  1.5336857 4.635229
        {
            "name": "check_example_3",
            "input": {
                "preds": pickle.load(
                    open("./test_support_files/test_preds_3.pkl", "rb")
                ),
                "target": pickle.load(
                    open("./test_support_files/test_batch_3.pkl", "rb")
                )[1],
            },
            "expected": 1.5870862,
        },  #  1.5870862 4.889481
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        # print("target", test_case["input"]["target"])
        # print("preds", test_case["input"]["preds"])
        print("result", result)
        print("jnp.exp(result)", jnp.exp(result))
        print("np.exp(result)", numpy.exp(result))

        try:
            assert jnp.isclose(result, test_case["expected"], atol=1e-5)
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
                f"Your log perplexity does not match with expected value.\nCheck if you are getting rid of the padding or checking if the target equals 0.\nIf your result is an array instead of a float, check if you are using numpy to perform the sums.\n\t Expected value near: {failed_cases[-1].get('expected')}.\n\t Got {failed_cases[-1].get('got')}."
            )

        try:
            assert not jnp.exp(result) == jnp.inf
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": jnp.exp(test_case["expected"]),
                    "got": jnp.exp(result),
                }
            )
            print(
                f"Your perplexity overflowed. Take a look to the axis you are using in np.sum() function.\n\t Expected value near: {failed_cases[-1].get('expected')}.\n\t Got {failed_cases[-1].get('got')}."
            )
        try:
            assert jnp.isclose(
                jnp.exp(result), jnp.exp(test_case["expected"]), atol=1e-5
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": jnp.exp(test_case["expected"]),
                    "got": jnp.exp(result),
                }
            )
            print(
                f"Your perplexity does not match with expected.\n\t Expected value near: {failed_cases[-1].get('expected')}.\n\t Got {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def unittest_test_model(target, pretrained_model):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "check_default_example",
            "input": {
                "file": pickle.load(
                    open("./test_support_files/test_batch_1.pkl", "rb")
                ),
            },
            "expected": 1.7646706,
        },  # 1.7646706 5.8396482
        {
            "name": "check_example_2",
            "input": {
                "file": pickle.load(
                    open("./test_support_files/test_batch_2.pkl", "rb")
                ),
            },
            "expected": 1.5336857,
        },  #  1.5336857 4.635229
        {
            "name": "check_example_3",
            "input": {
                "file": pickle.load(
                    open("./test_support_files/test_batch_3.pkl", "rb")
                ),
            },
            "expected": 1.5870862,
        },  #  1.5870862 4.889481
    ]

    for test_case in test_cases:
        preds = pretrained_model(test_case["input"]["file"][0])

        test_case["input"]["preds"] = preds
        test_case["input"]["target"] = test_case["input"]["file"][1]

        test_case["input"].pop("file", None)

        result = target(**test_case["input"])

        try:
            assert jnp.isclose(result, test_case["expected"], atol=1e-5)
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
                f"Your log perplexity does not match with expected value.\nCheck if you are getting rid of the padding or checking if the target equals 0.\nIf your result is an array instead of a float, check if you are using numpy to perform the sums.\n\t Expected value near: {failed_cases[-1].get('expected')}.\n\t Got {failed_cases[-1].get('got')}."
            )

        try:
            assert not jnp.exp(result) == jnp.inf
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": jnp.exp(test_case["expected"]),
                    "got": jnp.exp(result),
                }
            )
            print(
                f"Your perplexity overflowed. Take a look to the axis you are using in np.sum() function.\n\t Expected value near: {failed_cases[-1].get('expected')}.\n\t Got {failed_cases[-1].get('got')}."
            )
        try:
            assert jnp.isclose(
                jnp.exp(result), jnp.exp(test_case["expected"]), atol=1e-5
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": jnp.exp(test_case["expected"]),
                    "got": jnp.exp(result),
                }
            )
            print(
                f"Your perplexity does not match with expected.\n\t Expected value near: {failed_cases[-1].get('expected')}.\n\t Got {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases
