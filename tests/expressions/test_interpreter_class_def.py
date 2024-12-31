#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

from monic.expressions import ExpressionsInterpreter, ExpressionsParser


def test_simple_class_definition():
    interpreter = ExpressionsInterpreter()
    parser = ExpressionsParser()
    code = """
    class Person:
        name = "John"
        age = 30

        def greet(self):
            return f"Hello, {self.name}!"
    """
    interpreter.execute(parser.parse(code))

    # Get the class from interpreter
    Person = interpreter.get_name_value("Person")

    # Test class attributes
    assert Person.name == "John"
    assert Person.age == 30

    # Create an instance and test method
    person = Person()
    assert person.greet() == "Hello, John!"


def test_class_with_inheritance():
    interpreter = ExpressionsInterpreter()
    parser = ExpressionsParser()
    code = """
    class Animal:
        def speak(self):
            return "Some sound"

    class Dog(Animal):
        def speak(self):
            return "Woof!"
    """
    interpreter.execute(parser.parse(code))

    # Get classes from interpreter
    Dog = interpreter.get_name_value("Dog")
    Animal = interpreter.get_name_value("Animal")

    # Test inheritance
    assert issubclass(Dog, Animal)

    # Test method override
    dog = Dog()
    assert dog.speak() == "Woof!"


def test_class_with_constructor():
    interpreter = ExpressionsInterpreter()
    parser = ExpressionsParser()
    code = """
    class Student:
        def __init__(self, name, grade):
            self.name = name
            self.grade = grade

        def get_info(self):
            return f"{self.name} is in grade {self.grade}"
    """
    interpreter.execute(parser.parse(code))

    # Get the class from interpreter
    Student = interpreter.get_name_value("Student")

    # Create instance with constructor
    student = Student("Alice", 10)
    assert student.name == "Alice"
    assert student.grade == 10
    assert student.get_info() == "Alice is in grade 10"


def test_class_with_class_method():
    interpreter = ExpressionsInterpreter()
    parser = ExpressionsParser()
    code = """
    class MathHelper:
        @classmethod
        def add(cls, a, b):
            return a + b

        @staticmethod
        def multiply(a, b):
            return a * b
    """
    interpreter.execute(parser.parse(code))

    # Get the class from interpreter
    MathHelper = interpreter.get_name_value("MathHelper")

    # Test class method and static method
    assert MathHelper.add(2, 3) == 5
    assert MathHelper.multiply(4, 5) == 20


def test_class_creation_and_usage_in_script():
    interpreter = ExpressionsInterpreter()
    parser = ExpressionsParser()
    code = """
    class Counter:
        def __init__(self, start=0):
            self.value = start

        def increment(self):
            self.value += 1
            return self.value

    # Create and use instances in the script
    counter1 = Counter()
    counter2 = Counter(10)

    result1 = counter1.increment()  # Should be 1
    result2 = counter1.increment()  # Should be 2
    result3 = counter2.increment()  # Should be 11

    # Store results for testing
    results = [result1, result2, result3, counter1.value, counter2.value]
    """
    interpreter.execute(parser.parse(code))

    # Get the results from the script
    results = interpreter.get_name_value("results")

    # Test the results
    assert results == [1, 2, 11, 2, 11]


def test_class_interaction_in_script():
    interpreter = ExpressionsInterpreter()
    parser = ExpressionsParser()
    code = """
    class BankAccount:
        def __init__(self, balance=0):
            self.balance = balance

        def deposit(self, amount):
            self.balance += amount
            return self.balance

        def transfer_to(self, other_account, amount):
            if self.balance >= amount:
                self.balance -= amount
                other_account.deposit(amount)
                return True
            return False

    # Create accounts and perform transactions
    account1 = BankAccount(100)
    account2 = BankAccount()

    # Perform operations
    deposit_result = account2.deposit(50)
    transfer_success = account1.transfer_to(account2, 30)
    failed_transfer = account2.transfer_to(account1, 100)

    # Store results for testing
    results = [
        deposit_result,        # Should be 50
        transfer_success,      # Should be True
        failed_transfer,       # Should be False
        account1.balance,      # Should be 70
        account2.balance,      # Should be 80
    ]
    """
    interpreter.execute(parser.parse(code))

    # Get the results from the script
    results = interpreter.get_name_value("results")

    # Test the results
    assert results == [50, True, False, 70, 80]
