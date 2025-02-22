#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import monic


def main():
    parser = monic.expressions.ExpressionsParser()
    context = monic.expressions.ExpressionsContext(
        allow_return_at_top_level=True, enable_cpu_profiling=True
    )
    interpreter = monic.expressions.ExpressionsInterpreter(context)

    code = """
        result = 0
        for i in range(1000):
            a = 1
            b = 2
            result = a + b
        print("Hello, world!")
        return result
    """
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    print(result)

    if interpreter.cpu_profiler:
        print()
        print(
            interpreter.cpu_profiler.get_report_as_string(code=code, top_n=10)
        )
        interpreter.cpu_profiler.reset()

    print()

    code = """
        def binary_search(arr: list[int], target: int) -> int:
            if not arr:
                return -1

            left = 0
            right = len(arr) - 1

            while left <= right:
                mid = (left + right) // 2

                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1

            return -1

        # Test cases
        test_arr = [1, 3, 5, 7, 9, 11, 13, 15]

        print(f"Array: {test_arr}")
        print(f"Search for 7: {binary_search(test_arr, 7)}")  # Should return 3
        print(f"Search for 15: {binary_search(test_arr, 15)}")  # Should return 7
        print(f"Search for 10: {binary_search(test_arr, 10)}")  # Should return -1
        print(f"Search for 1: {binary_search(test_arr, 1)}")  # Should return 0

        # Edge cases
        empty_arr = []
        print()
        print(f"Empty array test: {binary_search(empty_arr, 5)}")  # Should return -1

        single_arr = [42]
        print(f"Single element array, search for 42: {binary_search(single_arr, 42)}")  # Should return 0
        print(f"Single element array, search for 1: {binary_search(single_arr, 1)}")  # Should return -1

        # Performance test with larger array
        large_arr = [x for x in range(0, 1000, 2)]  # Even numbers from 0 to 998
        result = binary_search(large_arr, 998)
        print()
        print(f"Large array search for 998: {result}")  # Should return 499
    """
    tree = parser.parse(code)
    result = interpreter.execute(tree)
    print(result)

    if interpreter.cpu_profiler:
        print()
        print(
            interpreter.cpu_profiler.get_report_as_string(code=code, top_n=10)
        )


if __name__ == "__main__":
    main()
