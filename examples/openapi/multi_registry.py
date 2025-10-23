#
# Monic Framework
#
# Copyright (c) 2024-2025 Cognica, Inc.
#

"""
Multiple Registries Example

Demonstrates advanced usage with multiple isolated registries,
useful for multi-tenant systems, environment separation, or plugin architectures.
"""

import json

from monic.expressions.registry import Registry, monic_bind
from monic.extra.openapi import generate_openapi_spec


# ==============================================================================
# Scenario 1: Environment-specific APIs
# ==============================================================================


def scenario_environment_separation():
    """Separate registries for different environments."""
    print("Scenario 1: Environment Separation")
    print("-" * 70)
    print()

    # Production registry - optimized, stable functions
    prod_registry = Registry()

    @prod_registry.bind("api.process")
    def prod_process(data: list[int]) -> int:
        """Production data processing (optimized)."""
        return sum(data)

    @prod_registry.bind("api.validate")
    def prod_validate(value: str) -> bool:
        """Production validation (strict)."""
        return len(value) > 0 and value.isalnum()

    # Development registry - includes debug functions
    dev_registry = Registry()

    @dev_registry.bind("api.process")
    def dev_process(data: list[int]) -> int:
        """Development data processing (with logging)."""
        print(f"[DEBUG] Processing: {data}")
        return sum(data)

    @dev_registry.bind("api.validate")
    def dev_validate(value: str) -> bool:
        """Development validation (lenient)."""
        print(f"[DEBUG] Validating: {value}")
        return len(value) > 0

    @dev_registry.bind("debug.inspect")
    def dev_inspect(obj: dict) -> dict:
        """Debug function (only in development)."""
        return {"keys": list(obj.keys()), "size": len(obj)}

    # Generate separate specs
    prod_spec = generate_openapi_spec(
        registry=prod_registry,
        title="Production API",
        version="1.0.0",
        description="Production-ready endpoints",
    )

    dev_spec = generate_openapi_spec(
        registry=dev_registry,
        title="Development API",
        version="1.0.0-dev",
        description="Development API with debugging features",
    )

    print(f"Production API: {len(prod_spec['paths'])} endpoints")
    print(f"Development API: {len(dev_spec['paths'])} endpoints")
    print()

    # Save specs
    with open("openapi_prod.json", "w", encoding="utf-8") as f:
        json.dump(prod_spec, f, indent=2)
    print("✓ Saved: openapi_prod.json")

    with open("openapi_dev.json", "w", encoding="utf-8") as f:
        json.dump(dev_spec, f, indent=2)
    print("✓ Saved: openapi_dev.json")

    print()
    return prod_registry, dev_registry


# ==============================================================================
# Scenario 2: Multi-tenant System
# ==============================================================================


def scenario_multi_tenant():
    """Separate registries for different tenants."""
    print("\nScenario 2: Multi-tenant System")
    print("-" * 70)
    print()

    # Tenant A - Basic features
    tenant_a_registry = Registry()

    @tenant_a_registry.bind("reports.generate")
    def tenant_a_generate_report(month: int, year: int) -> dict:
        """Generate monthly report for Tenant A."""
        return {"tenant": "A", "month": month, "year": year}

    @tenant_a_registry.bind("data.export")
    def tenant_a_export(format: str) -> str:
        """Export data for Tenant A."""
        return f"Exporting data in {format} format for Tenant A"

    # Tenant B - Premium features
    tenant_b_registry = Registry()

    @tenant_b_registry.bind("reports.generate")
    def tenant_b_generate_report(month: int, year: int) -> dict:
        """Generate monthly report for Tenant B (with analytics)."""
        return {
            "tenant": "B",
            "month": month,
            "year": year,
            "analytics": True,
        }

    @tenant_b_registry.bind("data.export")
    def tenant_b_export(format: str) -> str:
        """Export data for Tenant B."""
        return f"Exporting data in {format} format for Tenant B"

    @tenant_b_registry.bind("analytics.advanced")
    def tenant_b_analytics(data: list[float]) -> dict:
        """Advanced analytics (Premium feature)."""
        return {
            "mean": sum(data) / len(data) if data else 0,
            "min": min(data) if data else 0,
            "max": max(data) if data else 0,
        }

    # Generate tenant-specific specs
    tenant_a_spec = generate_openapi_spec(
        registry=tenant_a_registry,
        title="Tenant A API",
        version="1.0.0",
    )

    tenant_b_spec = generate_openapi_spec(
        registry=tenant_b_registry,
        title="Tenant B API (Premium)",
        version="1.0.0",
    )

    print(f"Tenant A: {len(tenant_a_spec['paths'])} endpoints")
    print(
        f"Tenant B: {len(tenant_b_spec['paths'])} endpoints (includes premium)"
    )
    print()

    return tenant_a_registry, tenant_b_registry


# ==============================================================================
# Scenario 3: Plugin Architecture
# ==============================================================================


def scenario_plugin_architecture():
    """Separate registries for core and plugins."""
    print("\nScenario 3: Plugin Architecture")
    print("-" * 70)
    print()

    # Core registry - base functionality
    core_registry = Registry()

    @core_registry.bind("core.authenticate")
    def authenticate(username: str, password: str) -> bool:
        """Authenticate user."""
        return len(username) > 0 and len(password) > 0

    @core_registry.bind("core.get_user")
    def get_user(user_id: int) -> dict:
        """Get user information."""
        return {"id": user_id, "name": f"User{user_id}"}

    # Plugin 1 - Email functionality
    email_plugin_registry = Registry()

    @email_plugin_registry.bind("email.send")
    def send_email(to: str, subject: str, body: str) -> bool:
        """Send email."""
        return len(to) > 0 and "@" in to

    @email_plugin_registry.bind("email.validate")
    def validate_email(email: str) -> bool:
        """Validate email address."""
        return "@" in email and "." in email

    # Plugin 2 - Payment functionality
    payment_plugin_registry = Registry()

    @payment_plugin_registry.bind("payment.process")
    def process_payment(amount: float, currency: str) -> dict:
        """Process payment."""
        return {
            "amount": amount,
            "currency": currency,
            "status": "success",
        }

    @payment_plugin_registry.bind("payment.refund")
    def process_refund(transaction_id: str) -> bool:
        """Process refund."""
        return len(transaction_id) > 0

    # Generate separate specs for each plugin
    core_spec = generate_openapi_spec(
        registry=core_registry,
        title="Core API",
        version="1.0.0",
    )

    email_spec = generate_openapi_spec(
        registry=email_plugin_registry,
        title="Email Plugin API",
        version="1.0.0",
    )

    payment_spec = generate_openapi_spec(
        registry=payment_plugin_registry,
        title="Payment Plugin API",
        version="1.0.0",
    )

    print(f"Core API: {len(core_spec['paths'])} endpoints")
    print(f"Email Plugin: {len(email_spec['paths'])} endpoints")
    print(f"Payment Plugin: {len(payment_spec['paths'])} endpoints")
    print()

    # Save plugin specs
    with open("openapi_core.json", "w", encoding="utf-8") as f:
        json.dump(core_spec, f, indent=2)
    with open("openapi_email_plugin.json", "w", encoding="utf-8") as f:
        json.dump(email_spec, f, indent=2)
    with open("openapi_payment_plugin.json", "w", encoding="utf-8") as f:
        json.dump(payment_spec, f, indent=2)

    print("✓ Saved plugin specs")
    print()

    return core_registry, email_plugin_registry, payment_plugin_registry


# ==============================================================================
# Scenario 4: Global vs Custom Registry
# ==============================================================================


def scenario_global_vs_custom():
    """Compare global registry with custom registries."""
    print("\nScenario 4: Global vs Custom Registry")
    print("-" * 70)
    print()

    # Functions in global registry (using @monic_bind)
    @monic_bind("global.function1")
    def global_func1(x: int) -> int:
        """Function in global registry."""
        return x * 2

    @monic_bind("global.function2")
    def global_func2(text: str) -> str:
        """Another function in global registry."""
        return text.upper()

    # Custom registry
    custom_registry = Registry()

    @custom_registry.bind("custom.function1")
    def custom_func1(x: int) -> int:
        """Function in custom registry."""
        return x * 3

    # Generate specs
    global_spec = generate_openapi_spec(
        title="Global Registry API",
        version="1.0.0",
    )

    custom_spec = generate_openapi_spec(
        registry=custom_registry,
        title="Custom Registry API",
        version="1.0.0",
    )

    print(f"Global registry: {len(global_spec['paths'])} endpoints")
    print("  (includes runtime modules + global functions)")
    print()
    print(f"Custom registry: {len(custom_spec['paths'])} endpoints")
    print("  (isolated, only custom functions)")
    print()

    return custom_registry


def main():
    """Run all multi-registry scenarios."""
    print("=" * 70)
    print("Monic Framework - Multiple Registries Example")
    print("=" * 70)
    print()

    # Run scenarios
    scenario_environment_separation()
    scenario_multi_tenant()
    scenario_plugin_architecture()
    scenario_global_vs_custom()

    print("=" * 70)
    print("Summary: When to Use Multiple Registries")
    print("=" * 70)
    print()
    print("Use multiple registries when:")
    print("  1. Environment separation (prod/dev/staging)")
    print("  2. Multi-tenant systems (different feature sets)")
    print("  3. Plugin architectures (isolated plugins)")
    print("  4. Testing (isolated test fixtures)")
    print("  5. API versioning (v1/v2 registries)")
    print()
    print("Benefits:")
    print("  • Complete isolation between function sets")
    print("  • Independent OpenAPI specs per registry")
    print("  • No naming conflicts")
    print("  • Granular access control")
    print()


if __name__ == "__main__":
    main()
