from enum import Enum


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    """
    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

    # def __setattr__(self, name, value):
    #    self._data[name] = value


class UnitTests(Enum):
    DOT_DICT = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.DOT_DICT:
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

    unit_test = UnitTests.DOT_DICT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)


