import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from qis.plots.table import plot_df_table



class LocalTests(Enum):
    TABLE = 0


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.TABLE:

        cars = {'Brand': ['Honda Civic', 'Toyota Corolla', 'Ford Focus', 'Audi A4'],
                'Price': [220.0, 250.0, 270.0, 35.0],
                'Engine': [175.0, 300.0, 100.0, 500.0],
                'Speed': [200.0, 150.0, 200.0, 175.0]}

        data = pd.DataFrame.from_dict(cars)
        data = data.set_index('Brand', drop=False)
        print(data)
        data['Price'] = data['Price']
        data['Engine'] = data['Engine']
        plot_df_table(df=data, heatmap_columns=[2], bold_font=False)
        plot_df_table(df=data, heatmap_rows_columns=((0, len(data.index)), (3, 4)))

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.TABLE)
