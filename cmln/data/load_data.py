from .crossdomain import CrossDomainUniDataset
from .ecomm import EcommUniDataset
from .yelp import YelpNCLFDataset
from .covid import CovidUniDataset


def load_data(args):
    shuffle, test_full, is_dynamic = (
        args.shuffle,
        args.test_full,
        args.dynamic,
    )
    if args.dataset == "Aminer":
        time_window = args.twin = (
            13 if args.twin == -1 or not args.dynamic else args.twin
        )  # static models use full training data
        dataset = CrossDomainUniDataset(
            time_window=time_window,
            shuffle=shuffle,
            test_full=test_full,
            is_dynamic=is_dynamic,
        )
        args.in_dim, args.hid_dim, args.out_dim = (
            -1 if not args.homo else 32,
            8,
            8,
        )  # -1 refers to lazy parameterization for hetero models
        args.predict_type = "author"

    elif args.dataset == "Ecomm":
        time_window = args.twin = (
            7 if args.twin == -1 or not args.dynamic else args.twin
        )  # static models use full training data
        dataset = EcommUniDataset(
            time_window=time_window,
            shuffle=shuffle,
            test_full=test_full,
            is_dynamic=is_dynamic,
        )
        args.in_dim, args.hid_dim, args.out_dim = (
            -1 if not args.homo else 8,
            8,
            8,
        )  # -1 refers to lazy parameterization for hetero models
        args.predict_type = ["user", "item"]

    elif args.dataset == "Yelp-nc":
        time_window = args.twin = (
            12 if args.twin == -1 or not args.dynamic else args.twin
        )  # static models use full training data
        dataset = YelpNCLFDataset(
            time_window=time_window,
            shuffle=shuffle,
            test_full=test_full,
            is_dynamic=is_dynamic,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        args.in_dim, args.hid_dim, args.out_dim = (
            -1 if not args.homo else 32,
            8,
            8,
        )  # -1 refers to lazy parameterization for hetero models
        args.num_classes = 3
        args.predict_type = "item"
    elif args.dataset == "covid":
        time_window = args.twin = (
            7 if args.twin == -1 or not args.dynamic else args.twin
        )  # static models use full training data
        dataset = CovidUniDataset(
            time_window=time_window, 
            shuffle=shuffle, 
            test_full=test_full, 
            is_dynamic=is_dynamic, 
            seed=args.seed
        )
        args.in_dim, args.hid_dim, args.out_dim = (
            -1 if not args.homo else 1,
            8,
            8,
        )  # -1 refers to lazy parameterization for hetero models
        args.num_classes = 2
        args.predict_type = "state"
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")
    return dataset, args
