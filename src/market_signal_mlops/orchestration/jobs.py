from dagster import Definitions, job, op

from market_signal_mlops.inference.run import (
    main as run_batch_inference,
)
from market_signal_mlops.monitoring.run import (
    main as run_monitoring,
)


@op
def batch_inference_op() -> None:
    run_batch_inference()


@job
def batch_inference_job() -> None:
    batch_inference_op()


@op
def monitoring_op() -> None:
    run_monitoring()


@job
def monitoring_job() -> None:
    monitoring_op()


defs = Definitions(
    jobs=[
        batch_inference_job,
        monitoring_job,
    ],
)
