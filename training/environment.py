"""
E11 -- the software environment as a recorded, checked input of every run.

  * check_pins: installed versions of the packages training imports must
    equal their `==` pins in requirements.txt, or training refuses to
    start (override: --ignore-pin-mismatch, recorded and printed).
  * environment_record: versions that decide the computation (compared on
    resume) and everything else installed (recorded only).

torch is pinned to the build Colab ships on a T4 (Step 6, 2026-10-28),
local CUDA label included (e.g. 2.11.0+cu130). A CUDA build must match it
exactly; a CPU-only build (torch.version.cuda is None, as on a laptop
running the tests) passes when the public version matches, since there is
no CPU build carrying a CUDA label. environment_record still records the
exact build and CUDA version either way.
"""

import hashlib
import importlib.metadata
import os
import platform
import re

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS_PATH = os.path.join(REPO_ROOT, "requirements.txt")

# Pinned packages that training imports (directly or through transformers).
# The plotting / analysis pins in requirements.txt are recorded only.
RUNTIME_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "huggingface_hub",
    "pennylane",
    "pennylane-lightning",
)

# A change in any of these makes a resumed run a different computation.
RESUME_COMPARED_PACKAGES = ("torch", "pennylane", "pennylane-lightning")

_PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*==\s*([^\s#;]+)")


class EnvironmentMismatchError(RuntimeError):
    """Installed package versions differ from the requirements.txt pins."""


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(path=None):
    """{normalized name: version} for every `name==version` line."""

    pins = {}

    with open(path or REQUIREMENTS_PATH) as f:
        for line in f:
            match = _PIN.match(line)
            if match:
                pins[normalize(match.group(1))] = match.group(3)

    return pins


def installed_version(name):

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def matches_pin(name, pinned, installed):
    """installed == pinned, except a CPU-only torch build of the pinned public version."""

    if installed == pinned:
        return True

    return (
        normalize(name) == "torch" and installed is not None and "+" in pinned
        and torch.version.cuda is None
        and installed.split("+")[0] == pinned.split("+")[0]
    )


def check_pins(override=False, path=None):
    """
    Raises EnvironmentMismatchError when a pinned runtime package is missing
    or at another version, unless override. Returns the check result that
    goes into the provenance.
    """

    pins = parse_pins(path)

    mismatches = [
        {"package": name, "pinned": pins[normalize(name)],
         "installed": installed_version(name)}
        for name in RUNTIME_PACKAGES
        if normalize(name) in pins
        and not matches_pin(name, pins[normalize(name)], installed_version(name))
    ]

    torch_pin = pins.get("torch")
    if torch_pin and installed_version("torch") != torch_pin and not mismatches:
        print(f"[ENV] torch {installed_version('torch')} is the CPU-only build of the "
              f"pinned {torch_pin}")

    unpinned = [n for n in RUNTIME_PACKAGES if normalize(n) not in pins]

    if unpinned:
        print(f"[ENV] not pinned in requirements.txt (recorded only): {unpinned}")

    if mismatches and not override:
        raise EnvironmentMismatchError(
            "installed versions differ from requirements.txt:\n"
            + "\n".join(
                f"  {m['package']}: pinned {m['pinned']}, installed {m['installed']}"
                for m in mismatches
            )
            + "\nInstall the pins, or pass --ignore-pin-mismatch (recorded in "
              "the provenance and printed with every result)."
        )

    if mismatches:
        print(f"[ENV][PIN CHECK OVERRIDDEN] {mismatches}")

    return {"mismatches": mismatches, "overridden": bool(override)}


def installed_packages():
    """Sorted `name==version` for every installed distribution (pip-freeze-like)."""

    seen = {}

    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            seen[normalize(name)] = f"{normalize(name)}=={dist.version}"

    return [seen[k] for k in sorted(seen)]


def environment_record():
    """
    (compared, recorded): `compared` must match for a resume; `recorded`
    is for the reader.
    """

    packages = installed_packages()

    compared = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "pennylane": installed_version("pennylane"),
        "pennylane-lightning": installed_version("pennylane-lightning"),
    }

    recorded = {
        "python": platform.python_version(),
        "pennylane-lightning-gpu": installed_version("pennylane-lightning-gpu"),
        **{n: installed_version(n) for n in RUNTIME_PACKAGES
           if n not in RESUME_COMPARED_PACKAGES},
        "numpy": installed_version("numpy"),
        "packages_sha256": hashlib.sha256("\n".join(packages).encode()).hexdigest(),
    }

    return compared, recorded


def write_package_list(path):
    """The full installed-package list behind packages_sha256."""

    with open(path, "w") as f:
        f.write("\n".join(installed_packages()) + "\n")


def describe(provenance):
    """
    One line for every printed result: versions, the pin-check state and
    the total / trainable parameter counts (A2).
    """

    env = provenance["environment"]
    pin_check = provenance["pin_check"]

    pins = (
        "pins OVERRIDDEN " + str(pin_check["mismatches"])
        if pin_check["overridden"] and pin_check["mismatches"]
        else "pin check overridden (no mismatch)" if pin_check["overridden"]
        else "pins ok"
    )

    counts = provenance["parameter_counts"]

    return (
        f"[ENV] torch {env['torch']} cuda {env['cuda']} "
        f"pennylane {env['pennylane']} lightning {env['pennylane-lightning']} | {pins}"
        f" | params total {counts['total']:,} trainable {counts['trainable']:,}"
    )
