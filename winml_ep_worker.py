"""Discover and install the AMD NPU execution provider through Windows ML."""

import json
import sys


def main() -> int:
    from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
        InitializeOptions,
        initialize,
    )
    import winui3.microsoft.windows.ai.machinelearning as winml

    result = {"providers": [], "errors": []}
    with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
        catalog = winml.ExecutionProviderCatalog.get_default()
        for provider in catalog.find_all_providers():
            item = {
                "name": provider.name,
                "state_before": str(provider.ready_state),
                "library_path": provider.library_path or "",
            }
            if provider.name == "VitisAIExecutionProvider" and not item["library_path"]:
                try:
                    ready = provider.ensure_ready_async().get()
                    item["ensure_status"] = str(ready.status)
                    item["library_path"] = provider.library_path or ""
                except Exception as exc:  # Windows ML returns rich platform exceptions.
                    result["errors"].append(f"{provider.name}: {exc}")
            result["providers"].append(item)

    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
