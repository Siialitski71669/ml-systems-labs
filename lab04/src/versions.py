import sys
from importlib.metadata import version, PackageNotFoundError

print("PYTHON VERSION")
print(sys.version)

packages = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "joblib",
    "PyYAML"
]

print("\nLIBRARY VERSIONS")

for package in packages:
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
