# Active Context - Current Work & Decision State

## Current Work Focus

### Current Phase: Complete Dev Branch Merge & Documentation Consolidation (In Progress)
**Status**: 🔄 **IN PROGRESS** - Merging complete dev branch with comprehensive documentation and dual-channel build system

**What is being accomplished**:

**Complete Dev Branch Integration**:
- Merging sophisticated tag-only, branch-based dual-channel build system from dev
- Integrating comprehensive GitHub issue templates and contributor workflows
- Consolidating development documentation with GitHub CLI management workflows
- Bringing in enhanced release automation and container build processes

**Documentation Consolidation**:
- **From Dev Branch**: Dual-channel release system, contributor workflows, issue templates
- **From Main Branch**: GitHub CLI integration, development workflow documentation
- **Result**: Comprehensive development ecosystem with professional build processes

**Release System Benefits**:
- **Stable Channel** (main branch): Auto-update enabled, production-ready releases
- **Dev Channel** (dev/feature branches): Auto-update disabled, manual updates for safety
- **Version Processing**: v0.1.0 → stable, v0.1.0-dev.1 → dev with validation
- **Container Strategy**: Separate tagging for ghcr.io/helgeerbe/ml_heating builds

### Previous Phase: Branch-Based Dual-Channel Release System (Complete)
**Status**: ✅ **COMPLETED** - Implemented sophisticated tag-only, branch-based dual-channel build system

**What was accomplished**:

**Dual-Channel Release Architecture**:
- Created branch-based build type detection system (main = stable, others = dev)
- Modified GitHub Actions to trigger only on tags, not commits
- Implemented automatic version and auto_update configuration based on build type
- Built comprehensive container tagging strategy for stable vs dev releases
- Added dev branch for development channel testing

**Release Automation Features**:
- **Stable Channel** (main branch): Auto-update enabled, production-ready releases
- **Dev Channel** (dev/feature branches): Auto-update disabled, manual updates for safety
- **Version Processing**: v0.1.0 → stable, v0.1.0-dev.1 → dev with validation
- **Container Strategy**: Separate tagging for ghcr.io/helgeerbe/ml_heating:v0.1.0 vs :v0.1.0-dev.1
- **Release Notes**: Dynamic release descriptions based on build type with prerelease flags

**Documentation & Workflow**:
- Created comprehensive CONTRIBUTOR_WORKFLOW.md with practical examples
- Documented branch workflows, version naming rules, and troubleshooting
- Established clear contributor guidelines for feature development and hotfixes
- Implemented safety validations preventing incorrect dev tag formats

### Previous Phase: GitHub Issues Transition & Project Management Setup (Complete)
**Status**: ✅ **COMPLETED** - Transitioned from progress.md tracking to professional GitHub Issues management

**What was accomplished**:

**GitHub Issues Infrastructure Setup**:
- Created comprehensive issue templates: Feature Request, Bug Report, Documentation
- Designed 4 detailed issues ready for GitHub creation:
  - Issue #1: Design Custom Logo/Icon for Home Assistant Add-on
  - Issue #2: Implement Automatic Backup Scheduling System
  - Issue #3: Validate Enhanced Add-on Configuration Features
  - Issue #4: Enhance Documentation with Visual Guides and User Examples
- Established project management structure with labels, milestones, and organization
- Created instruction guide for creating and managing GitHub issues

**Memory Bank Role Evolution**:
- **Previous Role**: Active development tracking via progress.md
- **New Role**: Historical context and architectural documentation for AI assistant sessions
- **GitHub Issues**: Now handle active development planning, community collaboration, and feature tracking
- **Benefit**: Enables community visibility and contribution while maintaining internal context

### Previous Phase: Complete PV Entity Migration & Configuration Optimization (Complete)
**Status**: ✅ **COMPLETED** - Successfully migrated from multiple PV entities to single entity across entire project

**What was accomplished**:

**Core Project PV Entity Migration**:
- Migrated from 3 separate PV entities (`PV1_POWER_ENTITY_ID`, `PV2_POWER_ENTITY_ID`, `PV3_POWER_ENTITY_ID`) to single `PV_POWER_ENTITY_ID`
- Updated all core files: `config.py`, `physics_features.py`, `influx_service.py`, `physics_calibration.py`
- Updated configuration files: `.env_sample` and `.env` to use `sensor.power_pv`
- Simplified PV power calculation from summing 3 entities to reading 1 aggregated entity
- Maintained backward compatibility for existing trained models (no recalibration needed)

**Add-on Configuration Optimization**:
- Added missing core entity parameters with Home Assistant entity autocomplete
- Removed unnecessary parameters (influxdb username/password, simplified ML params)
- Implemented comprehensive entity selector schemas for all Home Assistant entities
- Updated examples to match actual user configuration
- Aligned add-on config with core project's .env_sample parameters
- Enhanced user experience with autocomplete dropdowns for entity selection
- Confirmed `training_lookback_hours` parameter is properly exposed for new user calibration

**Technical Benefits**:
- **Simplified Configuration**: Single PV parameter instead of three separate entities
- **Consistent Architecture**: Core project and add-on now use identical single-entity approach
- **Cleaner Code**: Eliminated redundant summing logic across multiple files
- **Better User Experience**: Entity autocomplete and reduced configuration complexity
- **Future-Proof**: Easier to maintain and extend with additional features

### Previous Phase: Live Performance Tracking Enhancement (Complete)
**Status**: ✅ **COMPLETED** - Major enhancement to real-time performance monitoring system

**What was accomplished**:
- Fixed critical AttributeError in physics model calibration
- Implemented comprehensive live performance tracking system
- Enhanced error handling and model robustness
- Upgraded confidence calculation from static to dynamic real-time tracking
- Verified production deployment and operation

### Previous Phase: Memory Bank Initialization (Complete)
**Status**: ✅ **COMPLETED** - Comprehensive memory bank documentation created

**What was accomplished**:
- Reviewed complete codebase including main modules, notebooks, and configuration
- Analyzed sophisticated physics-based ML architecture with multi-lag learning
- Documented system patterns, technology decisions, and implementation details
- Created structured memory bank with 6 core files covering all aspects

### Memory Bank Structure Created

**Core Documentation Files**:
1. **`projectbrief.md`** ✅ - Foundation document defining scope, goals, and technical approach
2. **`productContext.md`** ✅ - Problem analysis, solution benefits, and user scenarios  
3. **`systemPatterns.md`** ✅ - Architecture patterns, design decisions, and implementation details
4. **`techContext.md`** ✅ - Technology stack, dependencies, and deployment architecture
5. **`activeContext.md`** ✅ - This file: current state and decision tracking
6. **`progress.md`** 🔄 - Next: Development status and what's working/remaining

## Recent Discoveries & Key Insights

### November 27, 2025 - Learning Parameters Export with Historical Timestamps

**Refinement Implemented**: Physics calibration now exports learning parameters with historical timestamps to InfluxDB.
- **Impact**: Improves traceability and analysis of model learning over time.
- **Details**: `src/influx_service.py` was modified to accept an optional timestamp, and `src/physics_calibration.py` was updated to utilize this for historical exports during calibration.

### November 26, 2025 - Major Live Performance Tracking Enhancement

**Critical Issue Resolved**: Fixed AttributeError that was preventing physics model calibration:
- **Root Cause**: Missing `pv_warming_coefficient`, `fireplace_heating_rate`, and `tv_heat_contribution` attributes in `RealisticPhysicsModel`
- **Impact**: Physics calibration (`--calibrate-physics`) was completely broken
- **Resolution**: Added all missing attributes with appropriate initial values in `__init__` method

**Major Enhancement Implemented**: **Live Performance Tracking System**
- **Real-time Sigma Calculation**: Dynamic uncertainty based on rolling 50-sample prediction error window
- **Adaptive Confidence**: Confidence now reflects actual recent prediction accuracy (0.02°C to 0.5°C range)
- **Live MAE/RMSE Updates**: Performance metrics update every control cycle, not just at startup
- **Prediction Error Tracking**: `track_prediction_error()` method captures actual vs predicted changes
- **Production Verified**: System confirmed working in live operation with excellent results

**Enhanced Error Handling**: Resolved multiple numpy RuntimeWarning issues:
- Added robust zero-division checks in correlation calculations
- Implemented `nan_to_num` handling for edge cases
- Improved stability of multi-lag learning functions (`_learn_pv_lags`, `_learn_fireplace_lags`, `_learn_tv_lags`, `_learn_seasonal_variations`)

### November 27, 2025 - Alpine/Scikit-learn Deployment Resolution

**Critical Deployment Issue Resolved**: Fixed Alpine Linux scikit-learn compilation problems in GitHub Actions:
- **Root Cause**: Home Assistant uses Alpine Linux base images where scikit-learn compilation fails due to missing build dependencies
- **Impact**: GitHub Actions container builds were failing, preventing deployment
- **Resolution**: Implemented `src/utils_metrics.py` with pure NumPy implementations replacing scikit-learn

**Complete Utils Metrics Implementation**: Created comprehensive metrics module:
- **Batch Functions**: `mae(y_true, y_pred)` and `rmse(y_true, y_pred)` for direct calculations
- **Incremental Classes**: `MAE` and `RMSE` classes with `.update()` and `.get()` methods for streaming data
- **Helper Functions**: `rolling_sigma()` and `confidence_from_sigma()` for confidence calculations
- **Pure NumPy**: No external ML library dependencies, compatible with Alpine Linux

**Notebook Compatibility Fixed**: Resolved notebook import issues:
- **Updated Imports**: `notebook_imports.py` now uses `src.utils_metrics as metrics`
- **Backward Compatibility**: Maintained identical API to previous River/scikit-learn metrics
- **Complete RMSE Fix**: Fixed incomplete `RMSE.get()` method that was missing return statement
- **Full Testing**: All notebook imports verified working correctly

**Production Impact**:
- **GitHub Actions**: Container builds now succeed on Alpine Linux
- **Notebooks**: All analysis notebooks work correctly with new metrics module
- **Deployment**: Home Assistant add-on can build successfully across all architectures
- **Performance**: Equivalent functionality to scikit-learn with lighter dependencies

### System Sophistication Level
This is a **production-grade, highly sophisticated** heating control system with:

**Advanced ML Architecture**:
- Physics-based model combining thermodynamics with online learning
- Multi-lag learning capturing thermal mass effects (PV: 120min delays, Fireplace: 90min)
- Automatic seasonal adaptation via trigonometric modulation (±30-50% variation)
- **Live performance tracking with adaptive confidence** (NEW: Nov 2025)
- **Real-time uncertainty calculation** based on actual prediction accuracy (NEW: Nov 2025)
- Shadow mode for safe testing and quantitative comparison

**Safety & Robustness**:
- 7-stage prediction pipeline with physics validation
- Multi-layer safety (absolute bounds, rate limiting, blocking detection, grace periods)
- Comprehensive error handling and network resilience
- DHW/defrost blocking with intelligent grace period recovery

**Advanced Features**:
- Online learning from every 30-min heating cycle
- **Dynamic MAE/RMSE tracking** updated every cycle (NEW: Nov 2025)
- **Real-time confidence bounds** adapting to performance (NEW: Nov 2025)
- External heat source integration (solar PV, fireplace, TV/electronics)
- Weather and PV forecast integration for proactive control
- Feature importance analysis and learning metrics export to InfluxDB
- **Enhanced error handling** with robust numpy correlation calculations (NEW: Nov 2025)

### Technical Architecture Excellence
**Key Patterns Identified**:
- **Physics-based ML**: Domain knowledge + data-driven learning
- **Multi-modal operation**: Active control vs passive observation (shadow mode)
- **Online learning**: River framework for continuous adaptation
- **Multi-lag learning**: Ring buffers + correlation-based coefficient learning
- **Seasonal adaptation**: Cos/sin modulation learned from summer HVAC-off data

### Deployment Maturity
**Production-Ready Features**:
- Systemd service with auto-restart and dependency management
- Comprehensive monitoring via Home Assistant sensors
- 6 Jupyter notebooks for analysis and debugging
- Detailed logging and error state reporting
- Shadow mode for risk-free testing

## Current System State

### Codebase Analysis Results

**Core Modules Reviewed**:
- ✅ `main.py` - Main control loop with blocking detection and grace periods
- ✅ `physics_model.py` - Advanced RealisticPhysicsModel with multi-lag learning  
- ✅ `config.py` - Comprehensive configuration management
- ✅ Notebooks - Learning dashboard and analysis tools
- ✅ `.env_sample` - Production deployment configuration template

**Architecture Understanding**:
- ✅ 7-stage prediction pipeline fully documented
- ✅ Multi-lag learning pattern explained (PV, fireplace, TV time delays)
- ✅ Seasonal adaptation mechanism understood (cos/sin modulation)
- ✅ Safety mechanisms mapped (5 layers of protection)
- ✅ Shadow vs active mode operation clarified

## Active Decisions & Considerations

### Development Workflow Documentation Decisions

**Documentation Approach Chosen**:
- **Comprehensive GitHub CLI Integration**: Complete command reference with examples
- **Issue-Driven Development**: Local drafting → GitHub issue → commit reference workflow
- **Template Standards**: Consistent issue content structure for feature requests and bugs
- **Workflow Integration**: Git, GitHub, and project-specific processes in single document
- **Troubleshooting Focus**: Common issues and solutions documented for efficiency

**Memory Bank Integration**:
- **Standalone Workflow Document**: `developmentWorkflow.md` as dedicated reference
- **Cross-Reference Architecture**: Links to other memory bank files for complete context
- **Version Control Integration**: Document git commands and branch strategies
- **Tool Requirements**: Complete CLI tool setup and authentication procedures

### Live Performance Tracking Implementation Decisions

**Architecture Approach Chosen**:
- **Rolling Window**: 50-sample prediction error history for stability
- **Adaptive Bounds**: Dynamic sigma between 0.02°C and 0.5°C based on performance
- **Real-time Updates**: MAE/RMSE recalculated every cycle instead of static values
- **Backward Compatibility**: Old models automatically initialize new tracking features
- **Production Integration**: Seamless integration with existing model wrapper and control loop

**Error Handling Strategy**:
- **Robust Correlation Calculations**: Added `nan_to_num` and zero-division protection
- **Graceful Degradation**: System continues operation even with correlation calculation issues
- **Comprehensive Testing**: Verified functionality with synthetic prediction error scenarios

### Memory Bank Design Decisions

**Documentation Approach Chosen**:
- **Structured Hierarchy**: Files build upon each other (brief → context → patterns → tech)
- **Technical Depth**: Captured sophisticated ML architecture and safety mechanisms
- **Practical Focus**: Emphasized deployment, configuration, and operational aspects
- **Code Examples**: Included actual implementation patterns for key concepts
- **Enhancement Tracking**: Document major improvements and their production impact

**Content Organization**:
- **projectbrief.md**: High-level scope and success criteria
- **productContext.md**: User problems and solution benefits
- **systemPatterns.md**: Architecture patterns and design decisions
- **techContext.md**: Technology stack and implementation details
- **activeContext.md**: Current state tracking and decisions
- **progress.md**: Status of development and remaining work

## Next Steps & Priorities

### Immediate Next Action
**Complete Dev Branch Merge**: Finalize merging process and consolidate documentation ecosystem

### Recently Completed

**Development Workflow Documentation Enhancement**:
- ✅ Created comprehensive `memory-bank/developmentWorkflow.md` documentation
- ✅ Documented GitHub CLI issue management workflows and commands
- ✅ Added issue templates and content standards for consistency
- ✅ Integrated git workflow, branch strategy, and semantic versioning
- ✅ Documented Home Assistant add-on development specific processes
- ✅ Added troubleshooting guides for common development issues
- ✅ Established quality assurance checklists and best practices

**Live Performance Tracking Enhancement**: 
- ✅ Fixed AttributeError preventing physics calibration
- ✅ Implemented real-time sigma calculation with rolling window
- ✅ Added dynamic confidence tracking based on actual performance
- ✅ Enhanced MAE/RMSE to update every cycle instead of being static
- ✅ Improved error handling for robust correlation calculations
- ✅ Verified production deployment and excellent performance

## Important
