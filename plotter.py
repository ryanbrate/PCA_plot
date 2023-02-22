from __future__ import \
    annotations  # ensure compatibility with cluster python version

import pathlib
import re
import sys
import typing
from collections import Counter, defaultdict
from functools import reduce
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import orjson
from sklearn.decomposition import PCA
from tqdm import tqdm

# load scripts in ./data_loader
for p in pathlib.Path("loaders").glob("*.py"):
    exec(f"import loaders.{p.stem}")


def main(argv):

    # load the path synonynms using in configs
    # e.g., {"DATA": "~/surfdrive/data"}
    with open("path_syns.json", "rb") as f:
        path_syns = orjson.loads(f.read())

    # load the configs - prefer CL specified over default
    try:
        configs: list = get_configs(argv[0], path_syns=path_syns)
    except:
        configs: list = get_configs("plotter_configs.json", path_syns=path_syns)

    # iterate over configs
    for config in configs:

        # get config options
        desc = config["desc"]
        print(f"running config={desc}")

        switch: bool = config["switch"]  # run config or skip?
        if not switch:
            break

        vector_labels_ = config["vector_labels"]
        if type(vector_labels_) == str:  # i.e., a fp to a json list given
            with open(resolve_fp(vector_labels_, path_syns=path_syns), "rb") as f:
                vector_labels = orjson.loads(f.read())
        else:  # assume passed a list of (vector label, score)
            vector_labels = vector_labels_

        ignore_weights_less_than = config["ignore_weights_less_than"]
        ignore_weights_greater_than = config["ignore_weights_greater_than"]
        if ignore_weights_less_than != None:
            vector_labels = [
                [label, weight]
                for label, weight in vector_labels
                if weight >= ignore_weights_less_than
            ]
        if ignore_weights_greater_than != None:
            vector_labels = [
                [label, weight]
                for label, weight in vector_labels
                if weight <= ignore_weights_greater_than
            ]

        vectors_fp = resolve_fp(config["vectors_fp"], path_syns=path_syns)
        loader = eval(config["loader"])
        output_dir: pathlib.Path = resolve_fp(config["output_dir"], path_syns=path_syns)
        plot_size: int = config["plot_size"]

        # config to be run?
        if switch == False:
            print("\tconfig switched off ... skipping")

        else:

            # load the embeddings model
            print("\tload the vector space information")
            vectors = loader(vectors_fp)  # vectors[label] returns corres. vector

            # ngrams and weightings
            if type(vector_labels[0]) != list:
                labels_and_weights = [[label, 0] for label in vector_labels]
            else:
                labels_and_weights = vector_labels

            # produce the plot
            print("\tbuild a PCA plot")
            try:
                # make plot
                plt, missing_labels = make_plot(vectors, labels_and_weights, plot_size)
                plt.tight_layout()

                # save plot
                save_fp = output_dir / f"{desc.replace(' ', '_')}.png"
                save_fp.parent.mkdir(exist_ok=True, parents=True)
                plt.savefig(save_fp, dpi=300)

                # save missing labels
                save_fp = output_dir / f"{desc}_missing_labels.json"
                save_fp.parent.mkdir(exist_ok=True, parents=True)
                with open(save_fp, "wb") as f:
                    f.write(orjson.dumps(missing_labels))

            except:
                pass

        # save a copy of the config
        save_fp = output_dir / "config.json"
        save_fp.parent.mkdir(exist_ok=True, parents=True)
        with open(save_fp, "wb") as f:
            f.write(orjson.dumps(config))


def make_plot(vectors, labels_and_weights: list, plot_size):
    "Return a 2D plot of ngrams wrt., model vectorspace"

    # separate ngrams and corresponding weightings
    labels = list(zip(*labels_and_weights))[0]
    weightings = np.array(list(zip(*labels_and_weights))[1])

    # set the marker size
    markersizes = {label: 12 for label in labels}

    # set colours - RdYIBu 'diverging colormap' from https://matplotlib.org/stable/tutorials/colors/colormaps.html
    colors = {
        label: weight for label, weight in zip(labels, scale(weightings, reverse=False))
    }
    cmap = plt.cm.get_cmap("binary")

    # get the tokens and their labels
    found_labels = []
    tokens = []
    missing_labels = []
    for label in labels:
        if vectors.has_label(label):
            tokens.append(vectors[label])
            found_labels.append(label)
        else:
            missing_labels.append(label)
    tokens = np.array(tokens)

    # define and fit the model
    xy = PCA(random_state=2).fit_transform(tokens)[:, :2]

    x = xy[:, 0]
    y = xy[:, 1]

    plt.figure(figsize=(plot_size, plot_size))
    for i, s, c in tqdm(
        zip(
            range(len(x)),
            [markersizes[label] for label in found_labels],
            [colors[label] for label in found_labels],
        )
    ):
        plt.scatter(x[i], y[i], facecolors="none", edgecolors="none")
        plt.annotate(
            found_labels[i],
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords="offset points",
            fontsize=s,
            color=cmap(c)[:3],
            ha="right",
            va="bottom",
        )

    return (plt, missing_labels)


def scale(values: np.ndarray, min_max: list = [0.2, 1], reverse: bool = False):
    """Return 'values', scaled between the specified min_max values."""

    range_ = min_max[1] - min_max[0]
    min_ = min_max[0]
    max_ = min_max[1]

    log_values = np.log(values)  # log scale the values, to account for very common

    # if all values the same, do nothing.
    if np.all(values == values[0]):
        return values
    # otherwise scale relative to oneanother
    else:
        range_ = min_max[1] - min_max[0]
        min_ = min_max[0]

        new_values = (
            range_
            * ((log_values - min(log_values)) / (max(log_values) - min(log_values)))
            + min_
        )

        if reverse:

            return list(-1 * (new_values - max_))

        else:

            return list(new_values)


def resolve_fp(path: str, path_syns: typing.Union[None, dict] = None) -> pathlib.Path:
    """Resolve path synonyns, ~, and make absolute, returning pathlib.Path.

    Args:
        path (str): file path or dir
        path_syns (dict): dict of
            string to be replaced : string to do the replacing

    E.g.,
        path_syns = {"DATA": "~/documents/data"}

        resolve_fp("DATA/project/run.py")
        # >> user/home/john_smith/documents/data/project/run.py
    """

    # resolve path synonyms
    if path_syns is not None:
        for fake, real in path_syns.items():
            path = path.replace(fake, real)

    # expand user and resolve path
    return pathlib.Path(path).expanduser().resolve()


def get_configs(config_fp_str: str, *, path_syns=None) -> list:
    """Return the configs to run."""

    configs_fp = resolve_fp(config_fp_str, path_syns)

    with open(configs_fp, "rb") as f:
        configs = orjson.loads(f.read())

    return configs


def gen_dir(
    dir_path: pathlib.Path,
    *,
    pattern: re.Pattern = re.compile(".+"),
    ignore_pattern: typing.Union[re.Pattern, None] = None,
) -> typing.Generator:
    """Return a generator yielding pathlib.Path objects in a directory,
    optionally matching a pattern.

    Args:
        dir (str): directory from which to retrieve file names [default: script dir]
        pattern (re.Pattern): re.search pattern to match wanted files [default: all files]
        ignore (re.Pattern): re.search pattern to ignore wrt., previously matched files
    """

    for fp in filter(lambda fp: re.search(pattern, str(fp)), dir_path.glob("*")):

        # no ignore pattern specified
        if ignore_pattern is None:
            yield fp
        else:
            # ignore pattern specified, but not met
            if re.search(ignore_pattern, str(fp)):
                pass
            else:
                yield fp


if __name__ == "__main__":
    main(sys.argv[1:])  # assumes an alternative config path may be passed to CL
