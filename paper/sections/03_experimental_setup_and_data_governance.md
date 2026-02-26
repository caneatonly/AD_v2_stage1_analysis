# 3. Experimental Setup and Data Governance

## 3.1 Evidence Objective and Overview

Free-decay experiments provide the primary evidence for parameter identification and model validation in this study. The experimental design is aligned with the mission phase of interest: each segment represents a single excitation followed by an unforced decay during the horizontal-launch to vertical-stabilization transition. The goal of this section is to define the experimental setup and the data governance rules that make the subsequent identification (Section~5) and validation (Section~6) reproducible and interpretable.

## 3.2 Free-Decay Protocol and Run Definition

Each run follows a protocolized sequence consisting of initial stabilization, recording start, a single manual excitation, free decay, and final stabilization. One raw file is required to contain only one excitation--decay event. This one-event-per-file rule is essential for deterministic preprocessing and segmentation and directly supports downstream condition-level split governance.

## 3.3 Data Schema, Preprocessing, and Deterministic Segmentation

Raw files are stored under the pipeline data contract and must satisfy a fixed column schema with monotonic timestamps. Preprocessing is designed to be deterministic so that repeated execution yields the same segment boundaries and derived signals, which is required to make model calibration and validation comparable across revisions. The resulting segments constitute the calibrated and validated evidence objects used throughout the remainder of the paper.

## 3.4 Condition Matrix and Anti-Leakage Split Strategy

The planned experimental program targets a condition matrix spanning initial pitch angle \(\theta_0\) and pitch rate \(q_0\) levels with repeated trials. In the current repository stage, the baseline dataset contains three free-decay segments, and the manuscript structure is written to scale directly to the expanded matrix.

A key data-governance decision is that splits are defined by operating-condition blocks rather than random samples. This avoids leakage between calibration and validation when segments from the same condition would otherwise share strong waveform similarity. The manifest-driven split topology is summarized in Fig.~3.

[Fig. 3. Experimental condition matrix, repeat policy, and anti-leakage split strategy. Insert here.]

## 3.5 Summary of Evidence Contracts

Sections~5--7 rely on three contracts established in this section: (i) each run represents exactly one excitation--decay event, (ii) preprocessing and segmentation are deterministic and protocolized, and (iii) calibration/validation splits are condition-aware to prevent leakage. These contracts define the evidence boundary for identification, validation, and design guidance.
