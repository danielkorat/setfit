import argparse
import json
import os
import pathlib
import sys
import warnings
from collections import Counter
from shutil import copyfile
from warnings import simplefilter

from datasets import load_dataset
from sentence_transformers import models
from typing_extensions import LiteralString
from torch.cuda import set_device as set_cuda_device

from setfit import SetFitModel, SetFitTrainer
from setfit.data import get_templated_dataset
from setfit.utils import DEV_DATASET_TO_METRIC, LOSS_NAME_TO_CLASS, TEST_DATASET_TO_METRIC


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from create_summary_table import create_summary_table  # noqa: E402

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument("--eval_dataset", default=None)
    parser.add_argument("--candidate_labels", nargs="+")
    parser.add_argument("--reference_dataset", default=None)
    parser.add_argument("--label_names_column", default="label_text")
    parser.add_argument("--augment_sample_size", type=int, default=4)
    parser.add_argument("--select_sample_size", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--select_iterations", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument(
        "--classifier",
        default="logistic_regression",
        choices=[
            "logistic_regression",
            "svc-rbf",
            "svc-rbf-norm",
            "knn",
            "pytorch",
            "pytorch_complex",
        ],
    )
    parser.add_argument("--loss", default="CosineSimilarityLoss")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--add_normalization_layer", default=False, action="store_true")
    parser.add_argument("--optimizer_name", default="AdamW")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--allow_skip", default=False, action="store_true")
    parser.add_argument("--zeroshot_warmup", default=False, action="store_true")
    parser.add_argument("--keep_body_frozen", default=False, action="store_true")
    parser.add_argument("--disable_data_augmentation", default=False, action="store_true")
    parser.add_argument("--is_dev_set", type=bool, default=True)
    parser.add_argument("--is_test_set", type=bool, default=False)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--select_method", default="mixed")
    parser.add_argument("--num_datasets", type=int, default=None)
    parser.add_argument("--continual", default=False, action="store_true")
    args = parser.parse_args()

    return args


def create_results_path(dataset: str, split_name: str, output_path: str) -> LiteralString:
    results_path = os.path.join(output_path, dataset, split_name, "results.json")
    print(f"\n\n======== {os.path.dirname(results_path)} =======")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    return results_path


def main(args_override):
    args = parse_args(args_override)
    print(args)
    set_cuda_device(args.cuda_device)

    # full_exp_name = f"{args.model.replace('/', '-')}-{args.loss}-{args.classifier}-iterations_{args.num_iterations}-batch_{args.batch_size}-{args.exp_name}".rstrip("-")
    full_exp_name = f"{args.select_method}-iter={args.select_iterations}_select={args.select_sample_size}" \
        f"_aug={args.augment_sample_size}_no_aug={args.disable_data_augmentation}_continual={args.continual}".rstrip("-")
    
    parent_directory = pathlib.Path(__file__).parent.absolute()
    output_path = parent_directory / "results" / full_exp_name
    os.makedirs(output_path, exist_ok=True)

    # Save a copy of this training script and the run command in results directory
    train_script_path = os.path.join(output_path, "train_script.py")
    copyfile(__file__, train_script_path)
    with open(train_script_path, "a") as f_out:
        f_out.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    # Configure loss function
    loss_class = LOSS_NAME_TO_CLASS[args.loss]


    # Configure dataset <> metric mapping. Defaults to accuracy
    if args.is_dev_set:
        dataset_to_metric = DEV_DATASET_TO_METRIC
    elif args.is_test_set:
        dataset_to_metric = TEST_DATASET_TO_METRIC
    else:
        metric = DEV_DATASET_TO_METRIC.get(args.eval_dataset, TEST_DATASET_TO_METRIC.get(args.eval_dataset, "accuracy"))
        dataset_to_metric = {args.eval_dataset: metric}

    for eval_dataset, metric in list(dataset_to_metric.items())[:args.num_datasets]:
        results_path = create_results_path(eval_dataset, f"train-{args.select_sample_size}-0", output_path)
        if os.path.exists(results_path) and args.allow_skip:
            print(f"Skipping finished experiment: {results_path}")
            exit()

        if args.reference_dataset is None or args.is_dev_set or args.is_test_set:
            args.reference_dataset = eval_dataset

        augmented_data = get_templated_dataset(
            reference_dataset=f"SetFit/{args.reference_dataset}",
            candidate_labels=args.candidate_labels,
            sample_size=args.augment_sample_size,
            label_names_column=args.label_names_column,
        )
        test_data = load_dataset(f"SetFit/{eval_dataset}", split="test")

        print(f"Evaluating {eval_dataset} using {metric!r}.")

        # Report on an imbalanced test set
        counter = Counter(test_data["label"])
        label_samples = sorted(counter.items(), key=lambda label_samples: label_samples[1])
        smallest_n_samples = label_samples[0][1]
        largest_n_samples = label_samples[-1][1]
        # If the largest class is more than 50% larger than the smallest
        if largest_n_samples > smallest_n_samples * 1.5:
            warnings.warn(
                "The test set has a class imbalance "
                f"({', '.join(f'label {label} w. {n_samples} samples' for label, n_samples in label_samples)})."
            )

        # Load model
        if args.classifier == "pytorch":
            model = SetFitModel.from_pretrained(
                args.model,
                use_differentiable_head=True,
                head_params={"out_features": len(set(augmented_data["label"]))},
            )
        else:
            model = SetFitModel.from_pretrained(args.model)
        model.model_body.max_seq_length = args.max_seq_length
        if args.add_normalization_layer:
            model.model_body._modules["2"] = models.Normalize()

        # Train on augmented data only
        trainer = SetFitTrainer(
            model=model,
            train_dataset=augmented_data,
            eval_dataset=test_data,
            metric=metric,
            loss_class=loss_class,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            num_iterations=args.num_iterations,
        )
        if args.classifier == "pytorch":
            trainer.freeze()
            trainer.train()
            trainer.unfreeze(keep_body_frozen=args.keep_body_frozen)
            trainer.train(
                num_epochs=25,
                body_learning_rate=1e-5,
                learning_rate=args.lr,  # recommend: 1e-2
                l2_weight=0.0,
                batch_size=args.batch_size,
            )
        else:
            trainer.train()

        unlabeled_data = load_dataset(f"SetFit/{eval_dataset}", split="train").shuffle(seed=0)

        for i in range(args.select_iterations):
            # Select few-shot examples from unlabeled data

            unlabeled_scores = trainer.predict_proba(unlabeled_data)

            num_labels = len(set(augmented_data["label"]))
            selected_dataset = select_few_shot_examples(unlabeled_scores, unlabeled_data,
                sample_size=args.select_sample_size, num_labels=num_labels, select_method=args.select_method)

            if not args.disable_data_augmentation:
                selected_dataset = get_templated_dataset(selected_dataset, \
                    reference_dataset=f"SetFit/{args.reference_dataset}", sample_size=args.augment_sample_size)

            if (i > 0 and not args.continual) or (i == 0 and not args.zeroshot_warmup):
                model = SetFitModel.from_pretrained(args.model)
                model.model_body.max_seq_length = args.max_seq_length
                if args.add_normalization_layer:
                    model.model_body._modules["2"] = models.Normalize()

            trainer = SetFitTrainer(
                model=model,
                train_dataset=selected_dataset,
                eval_dataset=test_data,
                metric=metric,
                loss_class=loss_class,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                num_iterations=args.num_iterations,
            )

            trainer.train()

        # Evaluate the model on the test data
        metrics = trainer.evaluate()
        print(f"Metrics: {metrics}")

        with open(results_path, "w") as f_out:
            json.dump(
                {"score": metrics[metric] * 100, "measure": metric},
                f_out,
                sort_keys=True,
            )

    # Create a summary_table.csv file that computes means and standard deviations
    # for all of the results in `output_path`.
    average_score = create_summary_table(str(output_path))

    # os.rename(output_path, f"{average_score}_{output_path}")


def select_few_shot_examples(probabilities, data: Dataset, sample_size: int, num_labels: int, select_method: str):
    total_samples = sample_size * num_labels
    if select_method == "entropy":
        certainties = -entropy(probabilities, base=2, axis=1)
        selected_data = get_top_n_by_criterion(data, total_samples, certainties)

    elif select_method == "top_2_diff":
        transposed = np.sort(probabilities).transpose()
        certainties = transposed[-1] - transposed[-2]
        selected_data = get_top_n_by_criterion(data, total_samples, certainties)

    elif select_method == "top_2_and_rand":
        transposed = np.sort(probabilities).transpose()
        certainties = transposed[-1] - transposed[-2]
        n_top_samples = int(total_samples * 0.7)
        selected_data = get_top_n_by_criterion(data, n_top_samples, certainties)
        random_data = data.shuffle(seed=0).select(range(total_samples - n_top_samples))
        selected_data = concatenate_datasets([selected_data, random_data])

    elif select_method == "random":
        selected_per_class = []
        for label_id in range(num_labels):
            class_examples = data.filter(lambda example: example["label"] == label_id)
            if len(class_examples) <= sample_size:
                selected_per_class.append(class_examples)
            else:
                selected_per_class.append(class_examples.shuffle(seed=0).select(range(sample_size)))
        selected_data = concatenate_datasets(selected_per_class)

        if len(selected_data) < total_samples:
            selected_data = concatenate_datasets([selected_data, data.shuffle(seed=0).select(range(total_samples - len(selected_data)))])

    elif select_method == "mixed":
        data = data.add_column("prediction", np.argmax(probabilities, axis=-1).tolist())

        transposed = np.sort(probabilities).transpose()
        certainties = transposed[-1] - transposed[-2]
        data = data.add_column("criterion", certainties)
    
        counts = {}

        selected_per_class = []
        for label_id in range(num_labels):
            class_predictions = data.filter(lambda example: example["prediction"] == label_id)
            class_size = len(class_predictions)
            n = int(sample_size / 2)
            sorted_data = class_predictions.sort("criterion")

            print("LENGTH", len(sorted_data))
            partition_factor = 6
            partition_size = class_size / partition_factor
            
            if class_size == 0:
                counts[label_id] = 0
                continue
            elif class_size <= n:
                selected_per_class.append(sorted_data)
                counts[label_id] = len(sorted_data)
            else:    
                if int(partition_size) <= n:
                    top_n = sorted_data.select(range(n))
                    bottom_n = sorted_data.select(range(class_size - n, class_size))
                    
                else:
                    top_range = range(int(1 * partition_size), int(2 * partition_size))
                    bottom_range = range(int(4 * partition_size), int(6 * partition_size))

                    top_n = sorted_data.select(top_range).shuffle(seed=0).select(range(n))
                    bottom_n = sorted_data.select(bottom_range).shuffle(seed=0).select(range(n))

                print(f"\n\n********************\nSelected examples for label {label_id}:")
                print("top n: \n", '\n'.join(str(example) for example in top_n))
                print("bottom n: \n", '\n'.join(str(example) for example in bottom_n))

                selected_per_class.extend([top_n, bottom_n])
                counts[label_id] = len(top_n) + len(bottom_n)
            
        selected_data = concatenate_datasets(selected_per_class)

        if len(selected_data) < total_samples:
            selected_data = concatenate_datasets([selected_data, data.shuffle(seed=0).select(range(total_samples - len(selected_data)))])

        selected_data = selected_data.remove_columns(["prediction", "criterion"])
        assert len(selected_data) == total_samples

    elif select_method == "hard_rand":
        data = data.add_column("prediction", np.argmax(probabilities, axis=-1).tolist())

        transposed = np.sort(probabilities).transpose()
        certainties = transposed[-1] - transposed[-2]
        data = data.add_column("criterion", certainties)
    
        counts = {}

        selected_per_class = []
        for label_id in range(num_labels):
            class_predictions = data.filter(lambda example: example["prediction"] == label_id)
            class_size = len(class_predictions)
            n = int(sample_size / 2)
            sorted_data = class_predictions.sort("criterion")

            print("LENGTH", len(sorted_data))
            partition_size = class_size / 6
            
            if class_size == 0:
                counts[label_id] = 0
                continue
            elif class_size <= n:
                selected_per_class.append(sorted_data)
                counts[label_id] = len(sorted_data)
            else:    
                if int(partition_size) <= n:
                    top_n = sorted_data.select(range(n))
                    
                else:
                    top_range = range(int(1 * partition_size), int(2 * partition_size))
                    top_n = sorted_data.select(top_range).shuffle(seed=0).select(range(n))
                
                rand_n = sorted_data.shuffle(seed=0).select(range(n))
                print(f"\n\n********************\nSelected examples for label {label_id}:")
                print("top n: \n", '\n'.join(str(example) for example in top_n))
                print("rand n: \n", '\n'.join(str(example) for example in rand_n))

                selected_per_class.extend([top_n, rand_n])
                counts[label_id] = len(top_n) + len(rand_n)
            
        selected_data = concatenate_datasets(selected_per_class)

        if len(selected_data) < total_samples:
            selected_data = concatenate_datasets([selected_data, data.shuffle(seed=0).select(range(total_samples - len(selected_data)))])

        selected_data = selected_data.remove_columns(["prediction", "criterion"])
        print("\n\n=========================================================")
        print(f"Selected label counts: {sorted(counts.values())}")
        print("=========================================================\n\n")
    else:
        raise ValueError("Invalid `--select_method`!")

    assert len(selected_data) == total_samples

    return selected_data
    
if __name__ == "__main__":
    args = [
        '--select_method', 'mixed',
        '--select_iterations', '4',
        '--select_sample_size', '2',
        '--augment_sample_size', '4',
        '--continual',
        '--num_datasets', '4',
    ]
    main()
