
from enum import Enum
from qis.utils.generic import DotDict


class Locals(Enum):
    DOT_DICT = 1


def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local == Locals.DOT_DICT:
        this = DotDict({'me': 3, 'you': 10})
        print(this)
        print(this.me)
        print(this.you)

        for k, v in this.items():
            print(f"{k}: {v}")

        this['me1'] = 6
        this.me2 = 12
        for k, v in this.items():
            print(f"{k}: {v}")


if __name__ == '__main__':

    run_local(local=Locals.DOT_DICT)
