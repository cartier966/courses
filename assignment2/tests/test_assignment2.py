import unittest
from unittest.mock import patch

from assignment2.__main__ import main


class MainTests(unittest.TestCase):
    def test_main_prints_greeting(self) -> None:
        with patch("builtins.print") as print_mock:
            main()

        print_mock.assert_called_once_with("Hello from assignment2!")


if __name__ == "__main__":
    unittest.main()
