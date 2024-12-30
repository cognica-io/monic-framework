#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import importlib
import types
import typing as t


class NamespaceProxy:
    """Proxy object for accessing nested namespaces."""

    def __init__(self, namespace: dict[str, t.Any]) -> None:
        self._namespace = namespace

    def __getattr__(self, name: str) -> t.Any:
        if name not in self._namespace:
            raise AttributeError(f"'{name}' not found in namespace")

        value = self._namespace[name]
        if isinstance(value, dict):
            return NamespaceProxy(value)
        return value


class Registry:
    """Registry for user-defined objects and functions."""

    def __init__(self) -> None:
        self._objects: dict[str, t.Any] = {}
        self._modules: dict[str, types.ModuleType] = {}
        self._default_objects: dict[str, t.Any] = {}
        self._default_modules: dict[str, types.ModuleType] = {}

    def reset(self) -> None:
        """Reset the registry to its initial state."""
        self._objects = {}
        self._modules = {}

    def _register_in_namespace(
        self,
        name: str,
        obj: t.Any,
        namespace: dict[str, t.Any],
    ) -> None:
        """Register an object in the given namespace, supporting nested names.

        Args:
            name: The name to register under, can include dots for nesting.
            obj: The object to register.
            namespace: The namespace dictionary to register in.

        Raises:
            ValueError: If there's a naming conflict or if the name is empty.
            TypeError: If the name is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("Name must be a string")

        if not name:
            raise ValueError("Name cannot be empty")

        if "." in name:
            parts = name.split(".")
            # Check for empty parts
            if any(not part for part in parts):
                raise ValueError("Name cannot contain empty parts")

            current_dict = namespace

            # Create or traverse the namespace hierarchy
            for part in parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                elif not isinstance(current_dict[part], dict):
                    raise ValueError(
                        f"Cannot create nested name '{name}': "
                        f"'{part}' is already registered as a non-namespace"
                    )
                current_dict = current_dict[part]

            # Register the object in the final namespace
            final_name = parts[-1]
            if final_name in current_dict:
                raise ValueError(
                    f"Name '{final_name}' is already registered in "
                    f"namespace '{'.'.join(parts[:-1])}'"
                )
            current_dict[final_name] = obj
        else:
            if name in namespace:
                raise ValueError(f"Name '{name}' is already registered")
            namespace[name] = obj

    def register(
        self, name_or_func: str | t.Callable | None = None
    ) -> t.Callable[[t.Any], t.Any] | t.Any:
        """Register an object with a given name.

        This decorator can be used in two ways:
            1. With parentheses: @register() or @register("custom.name")
            2. Without parentheses: @register

        Args:
            name_or_func: Either a string name to register under (can include
                          dots for nesting), or the function itself when used
                          as @register without parentheses.

        Returns:
            Either a decorator function or the decorated object itself.
        """
        # Case 1: @register (no parentheses)
        if callable(name_or_func):
            register_name = getattr(name_or_func, "__name__", None)
            if register_name is None:
                raise ValueError(
                    "Object has no __name__ attribute and no name was provided"
                )
            return self._register_object(register_name, name_or_func)

        # Case 2: @register() or @register("custom.name")
        def decorator(obj: t.Any) -> t.Any:
            register_name = name_or_func or getattr(obj, "__name__", None)
            if register_name is None:
                raise ValueError(
                    "No name provided and object has no __name__ attribute"
                )
            return self._register_object(register_name, obj)

        return decorator

    def is_registered(self, name_or_obj: str | t.Any) -> bool:
        """Check if an object is registered.

        Args:
            name_or_obj: The name or object to check

        Returns:
            True if the object is registered, False otherwise
        """
        if isinstance(name_or_obj, str):
            try:
                obj = self.get(name_or_obj)
            except KeyError:
                return False
        else:
            obj = name_or_obj

        if isinstance(obj, types.FunctionType):
            return hasattr(obj, "__is_expressions_type__")

        return True

    def _register_object(self, name: str, obj: t.Any) -> t.Any:
        """Register an object with a given name.

        Args:
            name: The name to register the object under
            obj: The object to register

        Returns:
            The registered object

        Raises:
            ValueError: If there's a naming conflict
        """
        # Split the name into parts
        parts = name.split(".")

        # Start with the root registry
        current = self._objects

        # Create nested dictionaries for each part except the last
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                raise ValueError(
                    f"'{part}' is already registered as a non-namespace"
                )
            current = current[part]

        # Check for conflicts
        if parts[-1] in current:
            raise ValueError(f"'{name}' is already registered in namespace")

        # Store the object in the registry
        current[parts[-1]] = obj

        # Mark functions as registered
        if isinstance(obj, types.FunctionType):
            setattr(obj, "__is_expressions_type__", True)

        return obj

    def register_module(
        self, module_name: str, alias: str | None = None
    ) -> types.ModuleType:
        """Register a Python module in the registry.

        Args:
            module_name: The name of the module to import and register.
            alias: Optional alias to register the module under.
                  If not provided, uses the last part of the module name.
                  Can include dots for nested names (e.g., 'np.random').

        Returns:
            The imported module.

        Raises:
            ImportError: If the module cannot be imported.
            ValueError: If the module is already registered or if the alias
                       conflicts with existing names.
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import module '{module_name}': {e}"
            ) from e

        # Use the last part of the module name if no alias is provided
        # e.g., 'numpy.random' -> 'random'
        register_name = alias or module_name.split(".")[-1]

        # Handle nested names in alias (e.g., 'np.random')
        if "." in register_name:
            self._register_in_namespace(register_name, module, self._objects)
        else:
            if register_name in self._modules:
                raise ValueError(
                    f"Module '{register_name}' is already registered"
                )
            self._modules[register_name] = module

        return module

    def bind_default(
        self, name_or_func: str | t.Callable | None = None
    ) -> t.Callable[[t.Any], t.Any] | t.Any:
        """Bind an object with a given name to the default registry.

        This decorator can be used in two ways:
            1. With parentheses: @bind_default() or @bind_default("custom.name")
            2. Without parentheses: @bind_default

        Args:
            name_or_func: Either a string name to register under (can include
                          dots for nesting), or the function itself when used
                          as @bind_default without parentheses.

        Returns:
            Either a decorator function or the decorated object itself.
        """
        # Case 1: @bind_default (no parentheses)
        if callable(name_or_func):
            register_name = getattr(name_or_func, "__name__", None)
            if register_name is None:
                raise ValueError(
                    "Object has no __name__ attribute and no name was provided"
                )
            return self._bind_default_object(register_name, name_or_func)

        # Case 2: @bind_default() or @bind_default("custom.name")
        def decorator(obj: t.Any) -> t.Any:
            register_name = name_or_func or getattr(obj, "__name__", None)
            if register_name is None:
                raise ValueError(
                    "No name provided and object has no __name__ attribute"
                )
            return self._bind_default_object(register_name, obj)

        return decorator

    def _bind_default_object(self, name: str, obj: t.Any) -> t.Any:
        """Bind an object with a given name to the default registry.

        Args:
            name: The name to register the object under
            obj: The object to register

        Returns:
            The registered object

        Raises:
            ValueError: If there's a naming conflict
        """
        # Split the name into parts
        parts = name.split(".")

        # Start with the root registry
        current = self._default_objects

        # Create nested dictionaries for each part except the last
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                raise ValueError(
                    f"'{part}' is already registered as a non-namespace"
                )
            current = current[part]

        # Check for conflicts
        if parts[-1] in current:
            raise ValueError(f"'{name}' is already registered in namespace")

        # Store the object in the registry
        current[parts[-1]] = obj

        # Mark functions as registered
        if isinstance(obj, types.FunctionType):
            setattr(obj, "__is_expressions_type__", True)

        return obj

    def bind_default_module(
        self, module_name: str, alias: str | None = None
    ) -> types.ModuleType:
        """Bind a Python module to the default registry.

        Args:
            module_name: The name of the module to import and register.
            alias: Optional alias to register the module under.
                  If not provided, uses the last part of the module name.
                  Can include dots for nested names (e.g., 'np.random').

        Returns:
            The imported module.

        Raises:
            ImportError: If the module cannot be imported.
            ValueError: If the module is already registered or if the alias
                       conflicts with existing names.
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import module '{module_name}': {e}"
            ) from e

        # Use the last part of the module name if no alias is provided
        # e.g., 'numpy.random' -> 'random'
        register_name = alias or module_name.split(".")[-1]

        # Handle nested names in alias (e.g., 'np.random')
        if "." in register_name:
            self._register_in_namespace(
                register_name, module, self._default_objects
            )
        else:
            if register_name in self._default_modules:
                raise ValueError(
                    f"Module '{register_name}' is already registered"
                )
            self._default_modules[register_name] = module

        return module

    def get_all(self) -> dict[str, t.Any]:
        """Get all registered objects and modules.

        Returns:
            Dictionary of registered objects and modules, with nested namespaces
            wrapped in NamespaceProxy objects.
        """
        result: dict[str, t.Any] = {}
        for name, value in self._default_objects.items():
            if isinstance(value, dict):
                result[name] = NamespaceProxy(value)
            else:
                result[name] = value
        for name, value in self._objects.items():
            if isinstance(value, dict):
                result[name] = NamespaceProxy(value)
            else:
                result[name] = value
        result.update(self._default_modules)
        result.update(self._modules)

        return result

    def _get_from_namespace(
        self, name: str, namespace: dict[str, t.Any]
    ) -> t.Any:
        # Split the name into parts
        parts = name.split(".")

        # Start with the root namespace
        current = namespace

        # Traverse the namespace
        for part in parts:
            if part not in current:
                raise KeyError(f"Name '{name}' is not defined")
            current = current[part]

        return current

    def get(self, name: str) -> t.Any:
        """Get an object from the registry.

        Args:
            name: The name to get the object for

        Returns:
            The registered object

        Raises:
            KeyError: If the name is not found in the registry
        """
        try:
            # Check if the name is in the default registry first
            return self._get_from_namespace(name, self._default_objects)
        except KeyError:
            # If not found, check the main registry
            return self._get_from_namespace(name, self._objects)


# Global registry instance
registry = Registry()

# Decorator for registering objects
register = registry.register

# Function for registering modules
register_module = registry.register_module

# Decorator for binding objects to the registry
monic_bind = registry.register

# Function for binding modules to the registry
monic_bind_module = registry.register_module

# Decorator for binding objects to the default registry
monic_bind_default = registry.bind_default

# Function for binding modules to the default registry
monic_bind_default_module = registry.bind_default_module
