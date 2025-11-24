# Make src a proper Python package for relative imports and type checking.
# This resolves mypy error: "No parent module -- cannot perform relative import".
# Future package-level constants or version info can go here.

__all__: list[str] = []
__version__ = "0.1.0"
