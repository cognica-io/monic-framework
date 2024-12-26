#
# Monic Framework
#
# Copyright (c) 2024 Cognica, Inc.
#

import importlib
import typing as t

from types import ModuleType


class NamespaceProxy:
    """Proxy object for accessing nested namespaces."""

    def __init__(self, namespace: t.Dict[str, t.Any]) -> None:
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
        self.reset()

    def reset(self) -> None:
        """Reset the registry to its initial state."""
        self._objects: t.Dict[str, t.Any] = {}
        self._modules: t.Dict[str, ModuleType] = {}

    def _register_in_namespace(
        self,
        name: str,
        obj: t.Any,
        namespace: t.Dict[str, t.Any],
    ) -> None:
        """Register an object in the given namespace, supporting nested names.

        Args:
            name: The name to register under, can include dots for nesting.
            obj: The object to register.
            namespace: The namespace dictionary to register in.

        Raises:
            ValueError: If there's a naming conflict.
        """
        if "." in name:
            parts = name.split(".")
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
        self, name_or_func: t.Optional[t.Union[str, t.Callable]] = None
    ) -> t.Union[t.Callable[[t.Any], t.Any], t.Any]:
        """Register an object or function in the registry.

        This decorator can be used in two ways:
            1. With parentheses: @register() or @register("custom.name")
            2. Without parentheses: @register

        The name can include dots for nested namespaces, e.g.:
            @register("math.functions.add")
            def my_add(x, y):
                return x + y

        Args:
            name_or_func: Either a string name to register under (can include
                         dots for nesting), or the function itself when used
                         as @register without parentheses.

        Returns:
            Either a decorator function or the decorated object itself.

        Raises:
            ValueError: If there's a naming conflict or if the object has no
                       name and none is provided.
        """
        # Case 1: @register (no parentheses)
        if callable(name_or_func):
            register_name = getattr(name_or_func, "__name__", None)
            if register_name is None:
                raise ValueError(
                    "Object has no __name__ attribute and no name was provided"
                )
            self._register_in_namespace(
                register_name, name_or_func, self._objects
            )
            return name_or_func

        # Case 2: @register() or @register("custom.name")
        def decorator(obj: t.Any) -> t.Any:
            register_name = name_or_func or getattr(obj, "__name__", None)
            if register_name is None:
                raise ValueError(
                    "No name provided and object has no __name__ attribute"
                )

            self._register_in_namespace(register_name, obj, self._objects)
            return obj

        return decorator

    def register_module(
        self, module_name: str, alias: t.Optional[str] = None
    ) -> ModuleType:
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

    def get_all(self) -> t.Dict[str, t.Any]:
        """Get all registered objects and modules.

        Returns:
            Dictionary of registered objects and modules, with nested namespaces
            wrapped in NamespaceProxy objects.
        """
        result: t.Dict[str, t.Any] = {}
        for name, value in self._objects.items():
            if isinstance(value, dict):
                result[name] = NamespaceProxy(value)
            else:
                result[name] = value
        result.update(self._modules)
        return result


# Global registry instance
registry = Registry()

# Decorator for registering objects
register = registry.register

# Function for registering modules
register_module = registry.register_module
