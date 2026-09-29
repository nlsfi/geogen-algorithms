#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


import pytest


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int | pytest.ExitCode,
    config: pytest.Config,
):
    passed = terminalreporter.stats.get("passed", []), {"green": True}
    failed = terminalreporter.stats.get("failed", []), {"red": True}

    if not passed[0] and not failed[0]:
        return

    terminalreporter.section(
        "Algorithm Throughput Report",
        sep="-",
        cyan=True,
    )

    for reports, kwargs in (passed, failed):
        for report in reports:
            for key, value in getattr(report, "user_properties", []):
                if key == "features_per_sec":
                    test_name = report.nodeid.split("::")[-1]
                    terminalreporter.write_line(
                        f"{test_name:<40} {value:>12} features/sec", **kwargs
                    )

    terminalreporter.write_sep("-", cyan=True)
