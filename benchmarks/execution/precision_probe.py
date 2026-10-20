"""Diagnose preservation of eager reduced-dtype rounding under Inductor fusion."""

import torch
from benchmark import main

original_compile = torch.compile


def preserving_compile(function, **options):
    options.pop("mode", None)
    backend_options = dict(options.pop("options", {}))
    backend_options["emulate_precision_casts"] = True
    return original_compile(function, options=backend_options, **options)


if __name__ == "__main__":
    torch.compile = preserving_compile
    main()
