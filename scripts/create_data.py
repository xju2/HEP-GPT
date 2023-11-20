from src.datamodules.components.odd_reader import ActsReader


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TrackML Reader')
    add_arg = parser.add_argument
    add_arg('inputdir', help='Input directory')
    add_arg('output_dir', help='Output directory')
    add_arg('-w', '--num_workers', type=int, default=1,)
    add_arg("--num-train", type=int, default=20, help="Number of training events")
    add_arg("--num-val", type=int, default=20, help="Number of validation events")
    add_arg("--num-test", type=int, default=0, help="Number of test events")
    add_arg("--with-padding", action="store_true", help="Whether to pad the tracks to the same length")
    add_arg("--outname-prefix", type=str, default="", help="Prefix for the output files")
    add_arg("--with-event-tokens", action="store_true", help="Whether to add event tokens")
    add_arg("--with-event-padding", action="store_true", help="Whether to pad the events to the same length")
    add_arg("--with-seperate-event-pad-token", action="store_true", help="Whether to use a seperate event pad token")
    add_arg("--min-truth-hits", type=int, default=4, help="Minimum number of truth hits")
    add_arg("--spname", type=str, default="spacepoints", help="Name of the spacepoints file")
    args = parser.parse_args()


    reader = ActsReader(vars(args))
    reader.run(
        {
            "train": (0, args.num_train),
            "val": (args.num_train, args.num_train + args.num_val),
            "test": (args.num_train + args.num_val, args.num_train + args.num_val + args.num_test),
        }
    )
