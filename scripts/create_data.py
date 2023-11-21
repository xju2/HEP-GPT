import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import logging
logging.basicConfig(
    filename='create_data.log', encoding='utf-8', level=logging.DEBUG,
    format='[%(asctime)s %(levelname)s %(name)s]: %(message)s',
    datefmt='%Y/%m/%d %I:%M:%S %p')
logger = logging.getLogger(__name__)

from src.datamodules.components.odd_reader import ActsReader


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Data Reader')
    add_arg = parser.add_argument
    add_arg('inputdir', help='Input directory')
    add_arg('outputdir', help='Output directory')
    add_arg('-w', '--num_workers', type=int, default=1,)
    add_arg("--num-train", type=int, default=20, help="Number of training events")
    add_arg("--num-val", type=int, default=20, help="Number of validation events")
    add_arg("--num-test", type=int, default=0, help="Number of test events")

    add_arg("--outname-prefix", type=str, default="", help="Prefix for the output files")
    add_arg("--min-truth-hits", type=int, default=4, help="Minimum number of truth hits")
    add_arg("--spname", type=str, default="spacepoint", help="Name of the spacepoints file")

    add_arg("--with-padding", action="store_true", help="Whether to pad the tracks to the same length")
    add_arg("--with-event-tokens", action="store_true", help="Whether to add event tokens")
    add_arg("--with-event-padding", action="store_true", help="Whether to pad the events to the same length")
    add_arg("--with-seperate-event-pad-token", action="store_true", help="Whether to use a seperate event pad token")

    args = parser.parse_args()

    reader = ActsReader(**vars(args))
    reader.run(
        {
            "train": (0, args.num_train),
            "val": (args.num_train, args.num_train + args.num_val),
            "test": (args.num_train + args.num_val, args.num_train + args.num_val + args.num_test),
        }
    )
