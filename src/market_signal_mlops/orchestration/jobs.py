from dagster import Definitions, job, op

from market_signal_mlops.inference.run import (
    main as run_batch_inference,
)


@op
def batch_inference_op() -> None:
    run_batch_inference()


@job
def batch_inference_job() -> None:
    batch_inference_op()


defs = Definitions(
    jobs=[batch_inference_job],
)
