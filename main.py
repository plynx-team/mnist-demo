from __future__ import print_function

import argparse
import logging
import mnist_demo.train
import mnist_demo.evaluation


class ACTIONS:
    TRAIN = 'train'
    SUMMARY = 'summary'
    PREDICT = 'predict'


def set_logging_level(verbose):
    LOG_LEVELS = {
        0: logging.CRITICAL,
        1: logging.ERROR,
        2: logging.WARNING,
        3: logging.INFO,
        4: logging.DEBUG
    }
    logging.basicConfig(level=LOG_LEVELS.get(verbose, 4))


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v',
        '--verbose',
        help='Set logging output more verbose',
        default=4,
        action='count',
        )

    subparsers = parser.add_subparsers(
        dest='action',
        help='Chose one of the actions'
    )
    subparsers.required = True

    sub = subparsers.add_parser(ACTIONS.TRAIN)
    sub.add_argument('--x-train', help='Path to X train features numpy array', required=True)
    sub.add_argument('--y-train', help='Path to Y train labels numpy array', required=True)
    sub.add_argument('--x-test', help='Path to X test features numpy array', required=True)
    sub.add_argument('--y-test', help='Path to Y test labels numpy array', required=True)
    sub.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=4)
    sub.add_argument('-o', '--model-path', help='Path to output model', required=True)
    sub.add_argument('-l', '--logs-path', help='Path to output logs file')

    sub = subparsers.add_parser(ACTIONS.SUMMARY)
    sub.add_argument('-m', '--model-path', help='Path to existing model', required=True)

    sub = subparsers.add_parser(ACTIONS.PREDICT)
    sub.add_argument('-m', '--model-path', help='Path to existing model', required=True)
    sub.add_argument('-x', '--x-val', help='Path to X validation features numpy array', required=True)
    sub.add_argument('-o', '--y-output', help='Path to output Y labels', required=True)

    return parser.parse_args()


ACTION_TO_FUNC = {
    ACTIONS.TRAIN: mnist_demo.train.train,
    ACTIONS.SUMMARY: mnist_demo.evaluation.model_summary,
    ACTIONS.PREDICT: mnist_demo.evaluation.batch_inference,
}


def main():
    args = vars(_get_args())
    set_logging_level(args.pop('verbose'))
    return ACTION_TO_FUNC[args.pop('action')](**args)


if __name__ == "__main__":
    main()
