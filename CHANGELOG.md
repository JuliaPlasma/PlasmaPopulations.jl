# Changelog

## [0.3.0] - 2026-11-27

### Changed

- **Breaking**: `isgyrotropic` is public but no longer exported.
- **Breaking**: with a vector `B0`, `current_density(plasma)` now treats bare gyrotropic drifts as parallel to the field.
- **Breaking**: `vth_perp_para` and `temperature_perp_para` now require a field except for bare gyrotropic models.
- **Breaking**: `plasma_beta(plasma)` now measures `(β⊥, β∥)` relative to `B0` (lab `+ẑ` for scalar fields).
- **Breaking**: a zero field for a `(⊥, ∥)` split or a zero axis for `Oriented` now raises `DomainError`.

## [0.2.0] - 2026-11-15

### Changed

- **Breaking**: `Electron`, `Positron` and `Proton` are constants: `Electron()` → `Electron`.
