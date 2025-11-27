# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Version strategy and development workflow documentation
- Changelog standards and commit message conventions
- Professional GitHub Issues management system

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Nothing yet

## [0.0.1-dev.1] - 2024-11-27

### Added
- Initial Home Assistant add-on structure and configuration
- Physics-based machine learning heating control system
- Real-time dashboard with overview, control, and performance panels
- Comprehensive configuration schema with entity validation
- InfluxDB integration for data storage and retrieval
- Multi-architecture support (amd64, arm64, armv7, armhf, i386)
- Backup and restore functionality for ML models
- Development API for external access (Jupyter notebooks)
- Advanced learning features with seasonal adaptation
- External heat source detection (PV, fireplace, TV)
- Blocking detection for DHW, defrost, and maintenance cycles
- Physics validation and safety constraints
- Professional project documentation and issue templates

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Home Assistant add-on discovery issue by implementing proper semantic versioning
- Add-on configuration validation and schema structure

### Security
- Secure API key authentication for development access
- InfluxDB token-based authentication
- AppArmor disabled for system-level heat pump control access

---

## Version History Notes

This changelog started with version 0.0.1-dev.1 as the project transitions from internal development to structured release management. Previous development history is captured in the Git commit log and project documentation.

### Versioning Strategy
- **0.0.x-dev.N**: Development builds for testing and iteration
- **0.0.x**: Development releases for broader beta testing  
- **0.x.0**: Beta releases with feature-complete functionality
- **x.0.0**: Production releases for general use

See `memory-bank/versionStrategy.md` for complete versioning guidelines.
