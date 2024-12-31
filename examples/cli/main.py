#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

import monic


def main():
    parser = monic.expressions.ExpressionsParser()
    interpreter = monic.expressions.ExpressionsInterpreter()

    # Ctrl-D to exit
    while True:
        try:
            code = input("monic> ")
            if not code:
                continue
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print()
            break

        try:
            tree = parser.parse(code)
            result = interpreter.execute(tree)
            if result is not None:
                print(result)
        except KeyboardInterrupt:
            print()
            continue
        except Exception as e:  # pylint: disable=broad-except
            print(e)


if __name__ == "__main__":
    main()
