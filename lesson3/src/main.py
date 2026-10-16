import numpy as np
import pandas as pd


def main() -> None:
    data = pd.DataFrame({"value": np.array([1, 2, 3])})
    print(data)


if __name__ == "__main__":
    main()
