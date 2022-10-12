# -*- coding: utf-8 -*-
import shutil
import trax
import numpy as np
from trax.fastmath import numpy as jnp

# UNIT TEST
# test data_generator
def test_data_generator(target):

    test_cases = [
        {
            "name": "next_equ_output_check",
            "input": [1, [[1]], [[1]], 0],
            "expected": {
                "expected_output": (np.array([[1]]), np.array([[1]])),
                "expected_type": type((lambda: (yield (0)))()),
            },
        },
        {
            "name": "next_equ_output_check",
            "input": [2, [[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5]], -1],
            "expected": {
                "expected_output": (
                    np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
                    np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
                ),
                "expected_type": type((lambda: (yield (0)))()),
            },
        },
        {
            "name": "next_equ_output_check",
            "input": [
                2,
                [[1, 2, 3, 4, 5], [2, 3, 4, 5]],
                [[1, 2, 3, 4, 5], [1, 2, 3, 4]],
                -1,
            ],
            "expected": {
                "expected_output": (
                    (np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, -1]])),
                    (np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, -1]])),
                ),
                "expected_type": type((lambda: (yield (0)))()),
            },
            "error": "Wrong output",
        },
        {
            "name": "next_two_outputs_check",
            "input": [
                3,
                [[1, 2, 3, 4, 5], [2, 3, 4, 5], [1], [2]],
                [[1, 2, 3, 4, 5], [1, 2, 3, 4], [1], [2]],
                -1,
            ],
            "expected": {
                "expected_output_first_iter": (
                    (
                        np.array(
                            [[1, 2, 3, 4, 5], [2, 3, 4, 5, -1], [1, -1, -1, -1, -1]]
                        )
                    ),
                    (
                        np.array(
                            [[1, 2, 3, 4, 5], [1, 2, 3, 4, -1], [1, -1, -1, -1, -1]]
                        )
                    ),
                ),
                "expected_output_sec_iter": (
                    (
                        np.array(
                            [[2, -1, -1, -1, -1], [1, 2, 3, 4, 5], [2, 3, 4, 5, -1]]
                        )
                    ),
                    (
                        np.array(
                            [[2, -1, -1, -1, -1], [1, 2, 3, 4, 5], [1, 2, 3, 4, -1]]
                        )
                    ),
                ),
                "expected_type": type((lambda: (yield (0)))()),
            },
        },
    ]

    successful_cases = 0
    failed_cases = []

    for test_case in test_cases:
        gen_result = target(*test_case["input"])
        #  Checking data type
        try:
            assert isinstance(gen_result, test_case["expected"]["expected_type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_type"],
                    "got": type(gen_result),
                }
            )
            print(
                f"Data type mistmatch.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
            )

        if test_case["name"] == "next_equ_output_check": 
            result = next(gen_result)
            try:
                assert np.allclose(result, test_case["expected"]["expected_output"])
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case["expected"]["expected_output"],
                        "got": result,
                    }
                )
                print(
                    f"Wrong output from data generator.\n\t Expected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
                )
            
            try:
                assert issubclass(
                        result[0].dtype.type, trax.fastmath.numpy.integer
                )
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": "check_type1",
                        "expected": trax.fastmath.numpy.integer,
                        "got": result[0].dtype.type,
                    }
                )
                print(
                    f"First output from data_generator has elements with the wrong type.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
                )
            
        
            try:
                assert issubclass(
                    result[1].dtype.type, trax.fastmath.numpy.integer
                )
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": "check_type2",
                        "expected": trax.fastmath.numpy.integer,
                        "got": result[1].dtype.type,
                    }
                )
                print(
                    f"Second output from data_generator has elements with the wrong type.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
                )

            

        else:
            result1 = next(gen_result)
            try:
                assert np.allclose(
                    result1, test_case["expected"]["expected_output_first_iter"]
                )
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case["expected"]["expected_output_first_iter"],
                        "got": result,
                    }
                )
                print(
                    f"Wrong output from data generator in the first iteration.\n\t Expected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
                )

            result2 = next(gen_result)
            try:
                assert np.allclose(
                    result2, test_case["expected"]["expected_output_sec_iter"]
                )
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": test_case["expected"]["expected_output_sec_iter"],
                        "got": result,
                    }
                )
                print(
                    f"Wrong output from data generator in the second iteration.\n\t Expected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
                )
                        

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


# UNIT TEST
# test data_generator
def test_NER(target):
    successful_cases = 0
    failed_cases = []

    tag_map = {"O": 0, "B-geo": 1, "B-gpe": 2}

    test_cases = [
        {
            "name": "default_input_check",
            "input": {
                "vocab_size": 35181,
                "d_model": 50,
                "tags": {"O": 0, "B-geo": 1, "B-gpe": 2},
            },
            "expected": {
                "expected_str": "Serial[\n  Embedding_35181_50\n  LSTM_50\n  Dense_3\n  LogSoftmax\n]",
                "expected_type": trax.layers.combinators.Serial,
                "expected_sublayers_type": [
                    trax.layers.core.Embedding,
                    trax.layers.combinators.Serial,
                    trax.layers.core.Dense,
                    trax.layers.base.PureLayer,
                ],
            },
        },
        {
            "name": "small_input_check",
            "input": {
                "vocab_size": 100,
                "d_model": 8,
                "tags": {"O": 0, "B-geo": 1, "B-gpe": 2, "B-per": 3, "I-geo": 4},
            },
            "expected": {
                "expected_str": "Serial[\n  Embedding_100_8\n  LSTM_8\n  Dense_5\n  LogSoftmax\n]",
                "expected_type": trax.layers.combinators.Serial,
                "expected_sublayers_type": [
                    trax.layers.core.Embedding,
                    trax.layers.combinators.Serial,
                    trax.layers.core.Dense,
                    trax.layers.base.PureLayer,
                ],
            },
        },
    ]

    # In trax 1.2.4 the LSTM layer appears as a Branch instead of a Serial
    # Test all layers are in the expected sequence
    for test_case in test_cases:
        # print(test_case)
        # print(target)
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

        try:
            sublayers_type = lambda x: list(map(type, x.sublayers))
            model_sublayers_type = sublayers_type(model)

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

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


# UNIT TEST
# test train_model
def test_train_model(target, model, data_generator):

    successful_cases = 0
    failed_cases = []

    train_steps = 0
    output_dir = "/tmp/model"

    try:
        shutil.rmtree(output_dir)
    except OSError as e:
        pass

    # Create simple data generator
    tr_generator = trax.data.inputs.add_loss_weights(
        data_generator(*[2, [[1]], [[1]], 35180]), id_to_mask=35180
    )

    # Create simple data generator
    ev_generator = trax.data.inputs.add_loss_weights(
        data_generator(*[2, [[1]], [[1]], 35180]), id_to_mask=35180
    )

    trainer = target(
        model, tr_generator, ev_generator, train_steps, output_dir=output_dir
    )

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
    loss_fn = "CrossEntropyLoss_in3"
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
        assert test_func(trainer) == ["CrossEntropyLoss", "Accuracy"]
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "metrics_check",
                "expected": ["CrossEntropyLoss", "Accuracy"],
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


# like_pred: Creates a prediction like matrix base on a set of true labels
def like_pred(labels, pad, tag_map_size):
    nb_sentences = len(labels)
    max_len = len(labels[0])
    result = []
    for i in range(0, nb_sentences):
        sentence = []
        for label in labels[i]:
            word = np.full(tag_map_size, 0)
            if label != pad:
                word[label] = 1
            sentence.append(word)
        result.append(np.array(sentence))

    return np.array(result)


# test test_evaluate_prediction
def test_evaluate_prediction(target):
    successful_cases = 0
    failed_cases = []

    pad = 35180
    test_cases = [
        {
            "name": "equ_output_check",  # worked!
            "input": [
                like_pred(np.array([[1, 0, 0], [2, 0, 0], [3, 4, 16]]), pad, 17),
                np.array([[1, 0, 0], [2, 0, 0], [3, 4, 16]]),  # <= labels
                pad,
            ],
            "expected": {"expected_output": 1, "expected_type": np.float64},
            "error": "Wrong output.",
        },
        {
            "name": "equ_output_check",  # leave this due to pad word
            "input": [
                np.array(
                    [
                        [
                            [0, 1, 0, 0, 0, 0, 0],  # <= predictions
                            [1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                        ],  # pad word
                        [
                            [0, 0, 1, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                        ],
                    ]
                ),
                np.array([[1, 0, pad], [2, 0, 0], [3, 4, 5]]),  # <= labels
                pad,
            ],
            "expected": {"expected_output": 1, "expected_type": np.float64},
            "error": "Wrong output: Pad token is being considered in accuracy calculation. Make sure to apply the mask.",
        },
        {
            "name": "equ_output_check",  # leave this due to pad word
            "input": [
                np.array(
                    [
                        [
                            [0, 1, 0, 0, 0, 0, 0],  # <= predictions
                            [1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                        ],  # pad word
                        [
                            [0, 0, 1, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                        ],
                    ]
                ),
                np.array(
                    [[1, 0, pad], [2, 1, 0], [3, 0, 0]]  # <= labels  # <= 1 error
                ),  # <= 2 errors
                pad,
            ],
            "expected": {"expected_output": 0.625, "expected_type": np.float64},
            "error": "Wrong output: Accuracy must be 5/8 = 0.625",
        },
        {
            "name": "equ_output_check",
            "input": [
                like_pred([[1, 1, 16, 35180], [0, 1, 2, 3], [1, 0, 0, 6]], pad, 17),
                np.array(
                    [[1, 1, 16, 35180], [0, 1, 2, 3], [1, 0, 0, 6]]
                ),  # <= 2 errors
                pad,
            ],
            "expected": {"expected_output": 1, "expected_type": np.float64},
            "error": "Wrong output: 3 sentences with perfect match",
        },
    ]

    for test_case in test_cases:
        result = target(*test_case["input"])

        try:
            assert test_case["expected"]["expected_output"] == result
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_output"],
                    "got": result,
                }
            )
            print(
                f"{test_case['error']}.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result, test_case["expected"]["expected_type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_type"],
                    "got": type(result),
                }
            )
            print(
                f"Wrong data type for output.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}"
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    #  return failed_cases, len(failed_cases) + successful_cases

