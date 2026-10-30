"""Run the web-based long-context inference gateway."""

from __future__ import annotations

import logging

from .application import serve
from .settings import Settings
from .vllm_adapter import serve_adapter


def gateway_main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    serve(Settings.from_env())


def adapter_main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    serve_adapter()


def train() -> None:
    from .train import main as train_main

    train_main()


def infer() -> None:
    from .distributed_infer import main as infer_main

    infer_main()


def capacity() -> None:
    from .capacity import main as capacity_main

    capacity_main()


def import_model() -> None:
    from .import_model import main as import_main

    import_main()


def main() -> None:
    gateway_main()


if __name__ == "__main__":
    main()
