# Changelog

## [0.3.0](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.2.1...v0.3.0) (2025-02-15)


### Features

* cpu-function altered to support cpus-per-gpu, too ([#28](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/issues/28)) ([30fecc3](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/30fecc3f96b238ad5a4cfdc82d55b5490b8b1524))


### Bug Fixes

* gres ([#31](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/issues/31)) ([bfa338e](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/bfa338ed2e226db9980d954a0aef6c0518daa6fe))

## [0.2.1](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.2.0...v0.2.1) (2024-04-11)


### Bug Fixes

* cgroup confinement ([#23](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/issues/23)) ([f8a0bfc](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/f8a0bfc3d44ff818a99b3f935c58360f485def97))

## [0.2.0](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.11...v0.2.0) (2024-04-06)


### Features

* improved debug messages ([#21](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/issues/21)) ([344ca68](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/344ca68a23e3bb3703a83738163f88df144fac82))

## [0.1.11](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.10...v0.1.11) (2024-03-11)


### Bug Fixes

* avoid redundant steps related to shared fs usage ([#19](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/issues/19)) ([a3379bb](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/a3379bb76933703f77fae13d2a850eef053bc396))

## [0.1.10](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.9...v0.1.10) (2024-01-16)


### Bug Fixes

* fix group job execution ([#14](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/issues/14)) ([a23e8f5](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/a23e8f5bfe53a6d0db93025761faa3d61b112865))

## [0.1.9](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.8...v0.1.9) (2024-01-13)


### Bug Fixes

* fix parallelism in case of group jobs ([#12](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/issues/12)) ([6c06d14](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/6c06d14078d1c9a1e002b1d9643fc9d3a9c056d1))

## [0.1.8](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.7...v0.1.8) (2023-12-08)


### Documentation

* update metadata ([903b2ea](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/903b2eaf89f3091d47f981250fe759aa976de3cb))

## [0.1.7](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.6...v0.1.7) (2023-12-06)


### Documentation

* update author encoding ([bafb5f1](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/bafb5f1153ab66c35261572d17217803207c28f4))

## [0.1.6](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.5...v0.1.6) (2023-11-20)


### Bug Fixes

* adapt to interface changes ([01cb97d](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/01cb97dc253ffb4d803477c73b89a8f2f0ab0e14))

## [0.1.5](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.4...v0.1.5) (2023-10-27)


### Bug Fixes

* fix release ci ([7134199](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/7134199cb61d34268af5bd43a17c41ba4aa6d24e))

## [0.1.4](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.3...v0.1.4) (2023-10-27)


### Bug Fixes

* adapt to changes in main branch of snakemake ([967095a](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/967095a05e1e82759608a9d9570714c6bb46c82b))

## [0.1.3](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.2...v0.1.3) (2023-09-21)


### Bug Fixes

* update dependencies ([ae5e7b0](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/ae5e7b07c46569a8ecabada42ead97b8e991a8c7))

## [0.1.2](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.1...v0.1.2) (2023-09-20)


### Bug Fixes

* adapt to changes in snakemake-interface-executor-plugins ([e8ef5e7](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/e8ef5e74b09806a2b916da86a6bccf9469b17b36))

## [0.1.1](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/compare/v0.1.0...v0.1.1) (2023-09-11)


### Bug Fixes

* adapt to API changes ([00698a8](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/00698a8548db27034da3dd9dee5fd67f32656042))

## 0.1.0 (2023-09-08)


### Bug Fixes

* adapt to API changes ([844f3ee](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/844f3ee68b54d9ec1eb5e6ef7395171851d912d8))
* fix dependency versions ([ab4426f](https://github.com/snakemake/snakemake-executor-plugin-slurm-jobstep/commit/ab4426f3d0fe6a0a82c1985737ea1da093f5dd9e))
