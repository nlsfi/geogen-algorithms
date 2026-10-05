# Test documentation

This repository contains three different types of tests: unit, integration and
benchmark tests. Each test type has its own folders.

Unit tests test one function's behavior. Integration tests runs an entire
generalization algorithm and compares its results to control data. Each
integration test has its own test data, in `tests/testdata/algo`. If an
algorithm class has methods, these should have unit tests in
`tests/unit/application`.

Benchmark tests include both tests for individual functions, but also entire
algorithms. Algorithm benchmarks generally use the same data as the
corresponding integration test.

As of writing, benchmark tests are not run in CI.

## Integration test configuration

Integration tests are defined and ran with the `IntegrationTest` class.

A "default" algorithm fixture and test input data is defined in
`tests/conftest.py`. This is done so that integration tests and algorithm
benchmarks can reuse the same algorithm instances and data.

It's possible to configure how integration tests are run by using environment
variables.

By default, each algorithm in an integration test is run twice, first with the
input data geodataframe's geometry column named as "geometry", the second time
as "geom". This causes algorithms to fail if it doesn't take into account the
possibility of different geometry column names. For development convenience one
may disable running each test twice by setting:

```shell
GEOGENALG_TEST_ONE_GEOM_COLUMN=1
```

### Integration test report

It's possible to inspect algorithm results in integration tests by setting:

```shell
GEOGENALG_TEST_REPORT_SAVE=1
```

When this is on and an algorithm fails a "test report" is saved, unless the
algorithm fails to an error before any results can be produced. The test report
is by default saved to a temporary folder which is shown in a UserWarning in
the test terminal output.

It's possible to configure the folder in which the test report is saved:

```shell
GEOGENALG_TEST_REPORT_DIR="path/to/folder"
```

Now the test report will always be saved into its own folder inside this set
folder. In this case, f.e. a `GeneralizeBuildingAreas` failure would save the
report to `path/to/folder/GeneralizeBuildingAreas_YYYY_MM_DD-HH_MM_SS`.

It's possible to set a specific folder as the target folder. In this case all
test report contents are subject for being overwritten:

```shell
GEOGENALG_TEST_REPORT_DIR_SPECIFIC=1
```

The test report consists of five possible files:

```shell
result.gpkg
result_features_not_in_control.gpkg # Features in result whose index was not found in control
control_features_not_in_result.gpkg # Features in control whose index was not found in result
geomdiff.gpkg # In case there are only geometric differences, this shows those
attributediff.csv # In case there are differences in attribute values, this shows those
```

## Running benchmarks

This repository uses `pytest-benchmark` for benchmarks. You can run benchmark tests
normally with pytest. However, benchmarks are disincluded by default so you *have*
to give the benchmark folder as an argument:

```shell
pytest tests/bench
```

If you wish to compare benchmark runs to another, first establish the baseline
(optionally choose specific test functions with the -k option):

```shell
pytest tests/bench --benchmark-save=<name>
```

Take note of the number of the saved .json file.

Make your changes to whatever you're benchmarking and compare to the baseline:

```shell
pytest tests/bench --benchmark-compare=<benchmark-number>
```

Note that by default (defined in `pyproject.toml`) benchmarks are grouped by
parameter. This can be configured manually:

```shell
pytest tests/bench --benchmark-group-by=func
```

One may find it useful to only show some result columns:

```shell
pytest test/bench --benchmark-columns=mean,max
```

### Algorithm throughput

Each algorithm has its own benchmark, run by the `AlgorithmBenchmark` class.
For each algorithm run, an extra test report is added to the pytest terminal
output:

```shell
------------------- Algorithm Throughput Report -------------------

test_generalize_contours                     11570.14 features/sec
-------------------------------------------------------------------
```

This is calculated from the number of features in the input data / mean
algorithm duration in seconds.

Note that many things affect this metric, and it can't be expected that each
algorithm can process as many features / sec as others, f.e. if their
geometries are significantly larger etc. It is however somewhat indicative and
a useful comparison point.

### Unit test benchmarks

Any time optimization work is done, preferably it would be done through unit
benchmark tests. I.e. if you're optimizing a specific function, add a new
benchmark test for it first (if one does not exist) and test your optimizations
through this benchmark, not just through an algorithm benchmark. This keeps the
results more focused and accurate.

Unit benchmark tests may use testdata, which is placed in
`tests/testdata/bench`. If a test needs a large amount of test data, add it to
this folder or you may also create synthetic test inputs via pytest fixtures
which are defined in `tests/bench/conftest.py`.

## Profiling code

`pytest-profiling` and `snakeviz` is included as a dev dependancy. You can
profile f.e. an algorithm as such:

```shell
pytest tests/integration -k contours --profile
```

Take note of the profiling file path. Then you can inspect the results
visually with snakeviz:

```shell
snakeviz path/to/prof/test.prof
```

This opens a local http server and you can look at the results in a web
browser.
