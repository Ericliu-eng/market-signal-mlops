# ADR-001: Repository Boundary Between Project 1 and Project 2

## Status

Accepted

## Context

Project 1 is a data lakehouse pipeline that produces reliable market data.

Project 2 is a Market Signal MLOps platform that trains, evaluates, registers, deploys, and monitors models using market data.

If Project 2 imports Project 1 internal Python modules, the two projects become tightly coupled and Project 2 cannot be tested independently.

## Decision

Project 2 will not import Project 1 internal Python code.

Project 2 will depend only on:

- exported CSV or Parquet snapshots
- stable database views
- documented data contracts

## Consequences

Benefits:

- Project 2 can run tests without starting Project 1.
- The model platform has a clear input boundary.
- The data contract can be tested independently.
- The project is easier to explain in interviews.

Tradeoff:

- Some export or mapping logic is needed between Project 1 and Project 2.