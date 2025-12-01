# Development Workflow - Alpha-Based Multi-Add-on Architecture

## Overview

This document captures the comprehensive alpha-based dual-channel development workflow implemented for the ML Heating project. The system uses a t0bst4r-inspired approach with separate stable and alpha channels, each with distinct add-ons and deployment strategies.

## Alpha-Based Multi-Add-on Architecture

### Dual Channel Strategy

**Stable Channel** (`ml_heating`):
- **Git Tags**: `v*` (e.g., `v0.1.0`, `v0.2.0`) - excludes alpha releases
- **Add-on Config**: `ml_heating_addons/ml_heating/config.yaml`
- **Container**: `ghcr.io/helgeerbe/ml_heating:{version}`
- **Auto-Updates**: ✅ Enabled for production reliability
- **Target Users**: Production deployments, general public

**Alpha Channel** (`ml_heating_dev`):
- **Git Tags**: `v*-alpha.*` (e.g., `v0.1.0-alpha.1`, `v0.1.0-alpha.8`)
- **Add-on Config**: `ml_heating_addons/ml_heating_dev/config.yaml`
- **Container**: `ghcr.io/helgeerbe/ml_heating:{alpha-version}`
- **Auto-Updates**: ❌ Disabled (manual testing)
- **Target Users**: Beta testers, developers, early adopters

### Branch Strategy
- **`main`** - Primary development branch, source for both channels
- **Feature branches** - Temporary branches for specific development work
- **All releases built from `main`** - Single source of truth for releases

## Alpha Development Workflow

### Alpha Release Cycle

#### Alpha Development Process
```bash
# 1. Make changes and commit to main
git add .
git commit -m "feat(physics): improve seasonal learning algorithm"
git push origin main

# 2. Create alpha release for community testing
git tag v0.1.0-alpha.9
git push origin v0.1.0-alpha.9

# 3. GitHub Actions automatically:
#    - Validates alpha add-on configuration with HA linter
#    - Updates version dynamically: "dev" → "0.1.0-alpha.9"
#    - Updates add-on name: "ML Heating Control (Alpha 0.1.0-alpha.9)"
#    - Builds multi-platform containers (amd64, aarch64, armhf)
#    - Creates alpha release with development warnings

# 4. Community tests alpha release
# 5. Iterate with more alpha builds as needed
```

#### Alpha Tag Examples
- `v0.1.0-alpha.1` - First alpha build of v0.1.0
- `v0.1.0-alpha.8` - Latest testing iteration
- `v0.2.0-alpha.1` - New feature development cycle

### Stable Release Cycle

#### When to Create Stable Releases
- Alpha releases tested extensively by community
- No critical bugs reported for 1+ weeks
- All planned features for version complete
- Documentation updated and comprehensive
- Automated workflows tested and validated

#### Stable Release Process
```bash
# When alpha testing is complete
git tag v0.1.0
git push origin v0.1.0

# GitHub Actions automatically:
# - Detects stable release (not alpha)
# - Updates stable add-on version: "0.1.0" → "0.1.0"
# - Builds production containers
# - Creates production release with comprehensive documentation
# - Enables auto-updates for stable users
```

### Dynamic Version Management (t0bst4r-inspired)

#### Alpha Build Version Updates
The development workflow automatically updates add-on configuration during build:

```bash
# Extract version from git tag
TAG_NAME=${GITHUB_REF#refs/tags/}
VERSION=${TAG_NAME#v}  # v0.1.0-alpha.8 → 0.1.0-alpha.8

# Update alpha add-on configuration dynamically
yq eval ".version = \"$VERSION\"" -i ml_heating_addons/ml_heating_dev/config.yaml
yq eval ".name = \"ML Heating Control (Alpha $VERSION)\"" -i ml_heating_addons/ml_heating_dev/config.yaml
```

#### Stable Build Version Updates
```bash
# Update stable add-on configuration
sed -i "s/^version: .*/version: \"$VERSION\"/" ml_heating_addons/ml_heating/config.yaml
```

This ensures:
- **Alpha add-ons** show specific version (e.g., "0.1.0-alpha.8") and include version in name
- **Stable add-ons** show clean semantic version (e.g., "0.1.0") with standard name
- **No manual configuration updates** needed - all handled by workflow automation

## Workflow Trigger Architecture

### Alpha Development Workflow (`.github/workflows/build-dev.yml`)

```yaml
name: Build Development Release

on:
  push:
    tags: ['v*-alpha.*']  # Only alpha tags trigger this workflow
  workflow_dispatch:

jobs:
  validate:     # HA linter validation for alpha add-on
  build-addon:  # Dynamic version update + multi-platform build  
  release:      # Alpha release with development warnings
```

**Key Features**:
- **Alpha-only triggering**: Only `v*-alpha.*` tags activate this workflow
- **Dynamic configuration**: Version and name updated automatically during build
- **Development warnings**: Release notes emphasize experimental nature
- **Disabled auto-updates**: Manual updates required for safety

### Stable Release Workflow (`.github/workflows/build-stable.yml`)

```yaml
name: Build Stable Release

on:
  push:
    tags: ['v*']         # All version tags
    branches-ignore: ['**']

jobs:
  check-release-type:    # Skip if alpha/dev tags detected
  validate:             # HA linter validation (if stable)
  build-addon:          # Version update + production build
  release:              # Production release
```

**Key Features**:
- **Smart filtering**: Automatically skips alpha releases using condition checks
- **Production configuration**: Enables auto-updates and optimized settings
- **Comprehensive releases**: Full feature documentation and upgrade guides
- **Multi-platform builds**: Same architecture support as alpha channel

## Development Best Practices

### Alpha Development Guidelines

#### When to Create Alpha Releases
- **New feature development**: Initial implementation ready for testing
- **Bug fixes**: Community-reported issues requiring validation
- **Performance improvements**: Optimizations needing real-world measurement
- **Configuration changes**: Add-on setting modifications requiring testing
- **Integration enhancements**: InfluxDB, Home Assistant, or external service improvements

#### Alpha Release Frequency
- **Active development**: Multiple alpha releases per week
- **Feature completion**: Alpha series until feature is stable
- **Community feedback**: Iterate based on tester reports
- **No time pressure**: Release when ready, not on schedule

#### Alpha Naming Convention
```bash
# Feature development progression
v0.1.0-alpha.1  →  Initial feature implementation
v0.1.0-alpha.2  →  Bug fixes from testing
v0.1.0-alpha.3  →  Performance improvements
v0.1.0-alpha.4  →  Final polish and documentation
v0.1.0          →  Stable release

# Next feature cycle
v0.2.0-alpha.1  →  New major feature development
# ... iterate through alpha testing
v0.2.0          →  Next stable release
```

### Stable Release Guidelines

#### Version Progression Rules
- **Alpha Build (+alpha.N)**: Testing iterations, experimental features
- **Patch (+0.0.1)**: Bug fixes, minor improvements, no breaking changes
- **Minor (+0.1.0)**: New features, significant improvements, backward compatible
- **Major (+1.0.0)**: Breaking changes, major architecture changes

#### Quality Gates for Stable Release
- [ ] **Community Testing**: Multiple alpha releases tested by users
- [ ] **Documentation Complete**: Installation guides, configuration docs updated
- [ ] **Issue Resolution**: No critical bugs reported in latest alphas
- [ ] **Feature Completeness**: All planned features implemented and tested
- [ ] **Workflow Validation**: CI/CD processes tested and working correctly

## Git Workflow Commands

### Standard Development Commands
```bash
# Check current status and branch
git status
git branch -a

# Standard commit workflow
git add <files>
git commit -m "feat(scope): description following conventional commits"
git push origin main

# Alpha release tagging
git tag v0.1.0-alpha.9
git push origin v0.1.0-alpha.9

# Stable release tagging  
git tag v0.1.0
git push origin v0.1.0
```

### Branch Management
```bash
# Create feature branch for complex development
git checkout -b feature/new-heating-algorithm
git push -u origin feature/new-heating-algorithm

# Merge back to main when complete
git checkout main
git merge feature/new-heating-algorithm
git push origin main
git branch -d feature/new-heating-algorithm
```

### Tag Management
```bash
# List existing tags
git tag -l

# Delete incorrect tag (local and remote)
git tag -d v0.1.0-alpha.9
git push origin :refs/tags/v0.1.0-alpha.9

# Check tag details
git show v0.1.0-alpha.8
```

## Testing Standards and Requirements

### Core Principle: Test-Driven Quality

**All new features and bug fixes MUST include comprehensive unit tests**. The project maintains a **100% test success rate** standard (16/16 tests passing).

### Current Test Coverage Status

**Existing Test Coverage (5 test files):**
- ✅ `test_blocking_polling.py` - Blocking detection and grace periods
- ✅ `test_clamp_baseline.py` - Temperature clamping and baseline logic
- ✅ `test_heat_balance_controller.py` - Heat Balance Controller (7 comprehensive tests)
- ✅ `test_pv_forecast.py` - PV forecast integration
- ✅ `test_state_manager.py` - State persistence

**Critical Gaps Requiring Future Test Coverage:**
- ❌ `physics_model.py` - Core ML logic (multi-lag learning, seasonal adaptation)
- ❌ `model_wrapper.py` - 7-stage prediction pipeline
- ❌ `physics_features.py` - Feature engineering
- ❌ `ha_client.py` - Home Assistant integration
- ❌ `influx_service.py` - InfluxDB data access
- ❌ `physics_calibration.py` - Calibration algorithms

### Testing Requirements for New Development

#### Mandatory Test Coverage
Every new feature or bug fix MUST include:

1. **Unit Tests** - Test individual functions and methods
2. **Edge Case Testing** - Boundary conditions, error handling
3. **Mock External Dependencies** - HA API, InfluxDB, network calls
4. **Integration Tests** (when applicable) - Cross-component interactions
5. **Test Success** - All tests must pass before merging

#### Test Coverage Standards

```python
# Example test structure for new features
import pytest
from unittest.mock import Mock, patch
from src.new_feature import NewFeature

class TestNewFeature:
    """Comprehensive tests for NewFeature functionality."""
    
    def test_basic_functionality(self):
        """Test core feature behavior."""
        feature = NewFeature()
        result = feature.process(input_data)
        assert result == expected_output
    
    def test_edge_cases(self):
        """Test boundary conditions."""
        feature = NewFeature()
        # Test with empty input
        assert feature.process([]) == default_value
        # Test with extreme values
        assert feature.process(large_value) == clamped_value
    
    @patch('src.new_feature.external_api')
    def test_with_mocked_dependencies(self, mock_api):
        """Test with external dependencies mocked."""
        mock_api.return_value = test_data
        feature = NewFeature()
        result = feature.process()
        assert result uses test_data
    
    def test_error_handling(self):
        """Test error conditions and recovery."""
        feature = NewFeature()
        with pytest.raises(ValueError):
            feature.process(invalid_input)
```

### Test File Organization

```
tests/
├── test_blocking_polling.py      # Blocking detection tests
├── test_clamp_baseline.py        # Temperature clamping tests
├── test_heat_balance_controller.py  # Controller tests (comprehensive)
├── test_pv_forecast.py           # PV integration tests
├── test_state_manager.py         # State persistence tests
├── fixtures/                     # Shared test data and mocks
│   ├── __init__.py
│   ├── sample_data.py           # Test data fixtures
│   └── mocks.py                 # Common mock objects
└── conftest.py                   # Pytest configuration
```

### Running Tests Locally

#### Execute Test Suite
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_heat_balance_controller.py

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run with verbose output
pytest tests/ -v

# Run specific test function
pytest tests/test_heat_balance_controller.py::TestHeatBalanceController::test_charging_mode
```

#### Expected Test Output
```bash
$ pytest tests/
============================= test session starts ==============================
collected 16 items

tests/test_blocking_polling.py ....                                      [ 25%]
tests/test_clamp_baseline.py ...                                         [ 43%]
tests/test_heat_balance_controller.py .......                            [ 87%]
tests/test_pv_forecast.py .                                              [ 93%]
tests/test_state_manager.py .                                            [100%]

============================== 16 passed in 2.34s ===============================
```

### Mocking Strategies

#### Mock External Dependencies
```python
# Mock Home Assistant API calls
@patch('src.ha_client.HAClient.get_state')
def test_with_ha_mock(mock_get_state):
    mock_get_state.return_value = 21.5
    result = function_using_ha()
    assert result is correct

# Mock InfluxDB queries
@patch('src.influx_service.InfluxService.fetch_history')
def test_with_influx_mock(mock_fetch):
    mock_fetch.return_value = [21.0, 21.2, 21.5]
    result = function_using_influx()
    assert result processes mock_data

# Mock time-dependent functions
@patch('src.main.datetime')
def test_with_time_mock(mock_datetime):
    mock_datetime.now.return_value = fixed_time
    result = time_dependent_function()
    assert result is deterministic
```

### CI/CD Integration

#### Automated Testing
All tests run automatically on:
- **Pull Requests** - Tests must pass before merge
- **Push to main** - Validates code quality
- **Alpha releases** - Ensures stability before tagging

#### Test Failure Protocol
If tests fail:
1. **Local debugging** - Run tests locally to reproduce
2. **Fix or skip** - Fix the issue or skip flaky tests temporarily
3. **Never merge failing tests** - Maintain 100% success rate
4. **Document skipped tests** - Create issue to fix properly

### Test-Driven Development (TDD) Workflow

#### Recommended Approach
```bash
# 1. Write test first (it will fail - that's expected!)
# tests/test_new_feature.py
def test_new_functionality():
    result = new_function(input)
    assert result == expected

# 2. Run test (should fail)
pytest tests/test_new_feature.py
# FAILED: NameError: new_function not defined

# 3. Implement minimal code to pass test
# src/new_feature.py
def new_function(input):
    return expected  # Simplest implementation

# 4. Run test (should pass)
pytest tests/test_new_feature.py
# PASSED

# 5. Refactor and improve
# Enhance implementation while keeping tests passing

# 6. Add more test cases
# Cover edge cases, error conditions, etc.
```

### Testing Checklist for Issues

Include in every issue's task plan:
```markdown
## Task Plan
- [ ] Design architecture/approach
- [ ] Implement core functionality
- [ ] **Write unit tests (aim for comprehensive coverage)**
- [ ] **Test edge cases and error handling**
- [ ] **Mock external dependencies properly**
- [ ] **Verify all tests pass locally**
- [ ] Update configuration examples
- [ ] Update documentation
- [ ] Create monitoring/dashboard updates
- [ ] Test in development environment
- [ ] Update CHANGELOG.md
- [ ] Create alpha release for testing
```

### Future Test Development Priorities

When expanding test coverage, prioritize:

**Phase 1: Core ML Logic (Critical)**
1. `test_physics_model.py` - Multi-lag learning, seasonal adaptation, predictions
2. `test_model_wrapper.py` - 7-stage pipeline, optimization, monotonic enforcement
3. `test_physics_features.py` - Feature engineering and aggregation

**Phase 2: Integration Points**
4. `test_ha_client.py` - Home Assistant API with comprehensive mocking
5. `test_influx_service.py` - InfluxDB queries and data persistence

**Phase 3: Supporting Systems**
6. `test_physics_calibration.py` - Calibration algorithms and metrics
7. `test_utils_metrics.py` - Utility functions and calculations

### Testing Best Practices

#### DO's ✅
- ✅ Write tests for ALL new features
- ✅ Test edge cases and boundary conditions
- ✅ Mock external dependencies (HA, InfluxDB)
- ✅ Run tests locally before committing
- ✅ Keep tests fast (under 5 seconds total)
- ✅ Make tests deterministic (no randomness)
- ✅ Use descriptive test names
- ✅ Maintain 100% test success rate

#### DON'Ts ❌
- ❌ Commit code without tests
- ❌ Skip testing "because it's simple"
- ❌ Write flaky tests that sometimes fail
- ❌ Test implementation details (test behavior)
- ❌ Leave commented-out test code
- ❌ Merge failing tests "to fix later"
- ❌ Ignore test failures in CI/CD

## Issue-Driven Development Workflow

### Core Principle: Issues Before Code

**All development work must start with a GitHub Issue**. This ensures:
- Clear problem definition before implementation
- Detailed task planning and tracking
- Transparent progress visibility
- Proper documentation of decisions

### Issue Creation Requirements

#### Before Starting Any Development
1. **Create GitHub Issue First** - No code without an issue
2. **Write Detailed Task Plan** - Break down work into specific steps
3. **Define Success Criteria** - Clear acceptance criteria
4. **Get Approval (if major)** - Discuss approach before implementation

#### Issue Template Structure

**Bug Report Format:**
```markdown
## Bug Description
Clear description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Observed behavior

## Expected Behavior
What should happen instead

## Environment
- Version: 
- Platform:
- Configuration:

## Task Plan
- [ ] Identify root cause
- [ ] Implement fix
- [ ] **Add/update test coverage (reproduce bug, verify fix)**
- [ ] **Ensure all tests pass (pytest tests/)**
- [ ] **Update documentation**
  - [ ] README.md: Document bug fix
  - [ ] CHANGELOG.md: Add fix to [Unreleased] section
  - [ ] Add-on docs if configuration changed
- [ ] Verify fix in development environment
- [ ] Create alpha release for validation
- [ ] Verify in production
```

**Note**: Bug fixes MUST include tests that reproduce the bug and verify the fix!

**Feature/Enhancement Format:**
```markdown
## Feature Description
Clear description of the new functionality

## Problem Statement
What problem does this solve?

## Proposed Solution
How will this be implemented?

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Task Plan
- [ ] Design architecture/approach
- [ ] Implement core functionality
- [ ] **Write comprehensive unit tests**
  - [ ] Test core functionality
  - [ ] Test edge cases and boundaries
  - [ ] Mock external dependencies
  - [ ] Test error handling
  - [ ] Achieve 100% test success rate
- [ ] **Update documentation (mandatory)**
  - [ ] README.md: Feature list, configuration, usage examples
  - [ ] CHANGELOG.md: Document all changes in [Unreleased]
  - [ ] Add-on docs: ml_heating/README.md and ml_heating_dev/README.md
  - [ ] Config examples: .env_sample, config.yaml files
  - [ ] Memory bank: systemPatterns.md, activeContext.md
- [ ] Create monitoring/dashboard updates (if applicable)
- [ ] Test in development environment
- [ ] **Verify all 16+ tests pass locally**
- [ ] Update CHANGELOG.md
- [ ] Create alpha release for testing

## Technical Considerations
- Dependencies:
- Breaking changes:
- Migration path:
```

### Task Planning Standards

#### Detailed Task Breakdown Required
Every issue must include a **checklist of specific tasks**:

```markdown
## Task Plan
- [ ] Design architecture/approach
- [ ] Implement core functionality
- [ ] **Write unit tests (comprehensive coverage required)**
- [ ] **Test edge cases and error handling**
- [ ] **Mock external dependencies (HA, InfluxDB)**
- [ ] **Verify all tests pass locally (pytest tests/)**
- [ ] Update configuration examples
- [ ] Update documentation
- [ ] Create monitoring/dashboard updates (if applicable)
- [ ] Test in development environment
- [ ] Update CHANGELOG.md
- [ ] Create alpha release for testing
```

**Note**: Testing is mandatory - no feature is complete without tests!

#### Task Update Protocol
**As work progresses, update the issue:**

```bash
# Mark task complete by editing issue description
# Change [ ] to [x] for completed tasks

# Example progression:
- [x] Step 1: Specific, measurable task  ✅ Completed
- [x] Step 2: Another specific task      ✅ Completed
- [ ] Step 3: Testing task               🔄 In Progress
- [ ] Step 4: Documentation task         ⏳ Pending
```

#### Progress Comments
Add comments to track progress and decisions:

```bash
gh issue comment <issue-number> --body "**Progress Update**: 
- Completed architecture design
- Implemented core controller logic
- Next: Add test coverage"
```

### Development Workflow with Issues

#### Complete Issue-Driven Workflow

```bash
# 1. CREATE ISSUE FIRST (Required)
gh issue create \
  --title "feat: Add adaptive trajectory horizon controller" \
  --body "$(cat issue-template.md)" \
  --label "enhancement" \
  --assignee @me

# Issue created: #16

# 2. REFERENCE ISSUE IN COMMITS
git commit -m "feat(controller): implement base trajectory logic

- Add trajectory prediction framework
- Implement horizon calculation
- Add configuration parameters

Related to #16"

# 3. UPDATE ISSUE PROGRESS
# Edit issue description to check off completed tasks
# Or add progress comments:
gh issue comment 16 --body "✅ Completed: Base trajectory logic implemented"

# 4. LINK COMMITS TO ISSUE
# All commits should reference the issue number
git commit -m "test: add trajectory controller tests

- Comprehensive test suite for new controller
- 100% coverage achieved
- All edge cases covered

Closes #16"

# 5. CLOSE ISSUE WHEN COMPLETE
# Using "Closes #16" in commit message auto-closes when merged
# Or manually close:
gh issue close 16 --comment "Completed and merged to main. Released in v0.2.0-alpha.1"
```

#### Commit Message Format with Issue References

```bash
# Standard format:
<type>(<scope>): <description>

<body>

Related to #<issue-number>
# or
Closes #<issue-number>

# Examples:
git commit -m "feat(physics): improve seasonal adaptation algorithm

- Enhanced cos/sin modulation
- Better convergence for edge cases
- Reduced learning time by 30%

Related to #15"

git commit -m "fix(controller): correct trajectory stability calculation

- Fixed oscillation penalty weight application
- Improved final destination scoring
- Added bounds checking

Closes #14"
```

### Issue Lifecycle Management

#### Issue States and Labels

**Status Labels:**
- `planning` - Issue created, planning in progress
- `in-progress` - Active development underway
- `testing` - Implementation complete, testing phase
- `review-needed` - Ready for code review
- `blocked` - Waiting on external dependency
- `completed` - Work finished and merged

**Type Labels:**
- `bug` - Production bugs requiring fixes
- `enhancement` - New features or improvements
- `documentation` - Documentation updates
- `refactoring` - Code quality improvements
- `alpha-testing` - Alpha release testing

**Priority Labels:**
- `priority-critical` - Blocking production issues
- `priority-high` - Important improvements
- `priority-medium` - Standard enhancements
- `priority-low` - Nice-to-have features

#### Issue Lifecycle Flow

```
1. CREATE ISSUE (with detailed task plan)
   ↓
2. LABEL & ASSIGN (planning, priority, type)
   ↓
3. UPDATE TO in-progress (when development starts)
   ↓
4. COMMIT WITH REFERENCES (all commits reference issue)
   ↓
5. UPDATE TASK CHECKLIST (as tasks complete)
   ↓
6. UPDATE TO testing (when code complete)
   ↓
7. CREATE ALPHA RELEASE (for community testing)
   ↓
8. CLOSE ISSUE (with completion comment)
```

### Best Practices

#### DO's ✅
- ✅ Create issue BEFORE starting any code
- ✅ Include detailed task breakdown
- ✅ Update issue as work progresses
- ✅ Reference issue in ALL related commits
- ✅ Add progress comments for visibility
- ✅ Close with summary of what was accomplished
- ✅ Link to related PRs, releases, or documentation

#### DON'Ts ❌
- ❌ Start coding without an issue
- ❌ Create vague issues without task plans
- ❌ Forget to update issue progress
- ❌ Commit without referencing issue number
- ❌ Close issues without completion comment
- ❌ Leave issues open after work is merged

### GitHub CLI Commands Reference

#### Issue Creation
```bash
# Create bug report
gh issue create --title "bug: Description" --label "bug,priority-high"

# Create feature request
gh issue create --title "feat: Description" --label "enhancement"

# Create with body from file
gh issue create --title "feat: New feature" --body-file issue-template.md
```

#### Issue Management
```bash
# List all open issues
gh issue list

# List by label
gh issue list --label "in-progress"

# View issue details
gh issue view 16

# Add comment
gh issue comment 16 --body "Progress update..."

# Close issue
gh issue close 16 --comment "Completed in v0.2.0-alpha.1"

# Reopen if needed
gh issue reopen 16
```

#### Issue Search and Filtering
```bash
# Search issues
gh issue list --search "trajectory"

# Filter by assignee
gh issue list --assignee @me

# Filter by state
gh issue list --state closed

# Complex search
gh issue list --label "enhancement" --search "controller" --state open
```

## GitHub Issue Management

### GitHub CLI Setup
Ensure GitHub CLI is installed and authenticated for efficient issue management:

```bash
# Check authentication
gh auth status

# Set default repository
gh repo set-default helgeerbe/ml_heating
```

### Alpha Release Issue Workflow

#### Creating Alpha Testing Issues
```bash
# Create alpha testing issue
gh issue create \
  --title "Alpha Testing: v0.1.0-alpha.8 - New Seasonal Learning" \
  --body "## Testing Request

**Alpha Version**: v0.1.0-alpha.8
**Focus Areas**: Seasonal learning improvements, PV lag optimization

### What's New
- Enhanced seasonal adaptation algorithm
- Improved PV lag coefficient learning
- Better error handling for edge cases

### Testing Instructions
1. Install alpha add-on from repository
2. Monitor learning progress for 24-48 hours
3. Report any unusual behavior or errors
4. Check dashboard metrics for improvements

### Feedback Requested
- Learning convergence speed
- Prediction accuracy changes
- Any error messages or issues
- Performance vs previous version

**⚠️ Alpha Warning**: This is experimental software for testing only." \
  --label "alpha-testing,community" \
  --assignee helgeerbe

# Link to alpha release
gh issue comment <issue-number> --body "Released as [v0.1.0-alpha.8](https://github.com/helgeerbe/ml_heating/releases/tag/v0.1.0-alpha.8)"
```

#### Community Feedback Management
```bash
# List alpha testing issues
gh issue list --label "alpha-testing"

# Update issue with feedback
gh issue comment <issue-number> --body "**Update**: Fixed reported issue with PV coefficient learning"

# Close resolved alpha issues
gh issue close <issue-number> --comment "Resolved in v0.1.0-alpha.9. Thank you for testing!"
```

### Issue Labels for Multi-Add-on Project
```bash
# Alpha channel labels
alpha-testing     # Community testing of alpha releases
alpha-feedback    # User feedback on alpha features
alpha-bug        # Bugs found in alpha releases

# Stable channel labels  
stable-release   # Stable release planning
production-bug   # Issues in stable releases
enhancement      # Feature requests for stable

# Component labels
workflow         # CI/CD and build process issues
documentation    # Docs updates for dual-channel setup
multi-addon      # Issues related to dual add-on architecture
```

## Container and Deployment Workflow

### Multi-Platform Building
Both alpha and stable channels support all Home Assistant platforms:
- **linux/amd64** - Standard x86_64 systems (Intel/AMD)
- **linux/aarch64** - Raspberry Pi 4, Apple Silicon, newer ARM64
- **linux/arm/v7** - Raspberry Pi 3, older ARM systems

Home Assistant Builder automatically handles cross-compilation.

### Container Tagging Strategy

#### Alpha Containers
```bash
# Each alpha gets specific container tag
v0.1.0-alpha.1  →  ghcr.io/helgeerbe/ml_heating:0.1.0-alpha.1
v0.1.0-alpha.8  →  ghcr.io/helgeerbe/ml_heating:0.1.0-alpha.8
v0.2.0-alpha.1  →  ghcr.io/helgeerbe/ml_heating:0.2.0-alpha.1
```

#### Stable Containers
```bash
# Stable versions get semantic tags plus latest
v0.1.0  →  ghcr.io/helgeerbe/ml_heating:0.1.0
v0.2.0  →  ghcr.io/helgeerbe/ml_heating:0.2.0, :latest
```

### Deployment Validation

#### Alpha Testing Workflow
```bash
# 1. Monitor GitHub Actions build
gh run list --workflow="Build Development Release"

# 2. Verify container publication
# Check packages at: https://github.com/helgeerbe/ml_heating/pkgs/container/ml_heating

# 3. Test Home Assistant discovery
# Add repository in HA: https://github.com/helgeerbe/ml_heating
# Verify "ML Heating Control (Development)" appears

# 4. Installation testing
# Install alpha add-on and verify functionality
# Check logs for errors or warnings

# 5. Community notification
gh issue create --title "Alpha Testing Available: v0.1.0-alpha.X" --body "..."
```

#### Stable Release Validation
```bash
# 1. Final alpha testing complete
# Ensure latest alpha has been thoroughly tested

# 2. Pre-release checklist
# [ ] Documentation updated
# [ ] CHANGELOG.md entries complete  
# [ ] No critical alpha issues reported
# [ ] Community feedback incorporated

# 3. Create stable release
git tag v0.1.0
git push origin v0.1.0

# 4. Post-release validation
# [ ] Both add-ons available in HA
# [ ] Auto-updates working for stable channel
# [ ] Release notes complete and accurate
# [ ] Community announcement posted
```

## Documentation Requirements

### Core Principle: Documentation is Mandatory

**All new features and significant changes MUST include documentation updates**. Incomplete documentation prevents users from benefiting from new functionality and creates support burden.

### Documentation Files That Require Updates

#### When Adding New Features

**Always Update:**
1. **README.md** (Project root)
   - Add new feature to feature list
   - Update configuration examples if needed
   - Add usage examples for new functionality
   - Update system requirements if changed

2. **CHANGELOG.md** (Project root)
   - Document all changes in unreleased section
   - Follow semantic versioning guidelines
   - Include breaking changes prominently
   - Reference related GitHub issue numbers

3. **Add-on README Files**
   - `ml_heating/README.md` (Stable channel)
   - `ml_heating_dev/README.md` (Alpha channel)
   - Update feature descriptions
   - Add new configuration parameters
   - Include usage examples
   - Update screenshots if UI changed

**Update When Applicable:**
4. **docs/INSTALLATION_GUIDE.md**
   - New dependencies or requirements
   - Changed installation procedures
   - New configuration steps

5. **docs/QUICK_START.md**
   - New quick start examples
   - Updated workflow diagrams
   - Changed default configurations

6. **Configuration Examples**
   - `.env_sample` - New environment variables
   - Config YAML files - New parameters with descriptions
   - Home Assistant integration examples

7. **Memory Bank Files**
   - `memory-bank/systemPatterns.md` - New architecture patterns
   - `memory-bank/activeContext.md` - Current development state
   - `memory-bank/progress.md` - Feature completion status

### Documentation Update Checklist

#### For Each New Feature
```markdown
## Documentation Updates Required
- [ ] **README.md**: Add feature to feature list with description
- [ ] **README.md**: Update configuration section if parameters added
- [ ] **README.md**: Add usage examples
- [ ] **CHANGELOG.md**: Document changes in [Unreleased] section
- [ ] **ml_heating/README.md**: Update stable add-on documentation
- [ ] **ml_heating_dev/README.md**: Update alpha add-on documentation
- [ ] **.env_sample**: Add new environment variables with descriptions
- [ ] **Config files**: Document new parameters with defaults and descriptions
- [ ] **docs/**: Update relevant guide documents
- [ ] **Memory bank**: Update systemPatterns.md with new architecture
```

### Documentation Standards

#### README.md Structure
```markdown
# ML Heating Control

## Features
- Existing feature 1
- Existing feature 2
- **NEW: Your new feature description**

## Configuration

### New Feature Configuration
\```yaml
new_feature_parameter: value  # Description of what this does
another_parameter: default    # When and why to change this
\```

### Usage Example
\```python
# How to use the new feature
example_code_here()
\```

## Changelog
See [CHANGELOG.md](CHANGELOG.md) for detailed version history.
```

#### CHANGELOG.md Format
```markdown
# Changelog

## [Unreleased]

### Added
- New feature name: Brief description (#issue-number)
- Another feature: What it does (#issue-number)

### Changed
- Modified behavior: What changed and why (#issue-number)

### Fixed
- Bug description: What was fixed (#issue-number)

### Breaking Changes
- ⚠️ Important change that breaks compatibility
- Migration steps: How to upgrade

## [0.1.0] - 2025-12-01
Previous release notes...
```

#### Configuration Documentation
```yaml
# .env_sample or config.yaml
# New Parameter Name
NEW_FEATURE_ENABLED: true
# Description: Enable/disable the new feature
# Default: true
# When to change: Set to false if you experience issues
# Related: See docs/NEW_FEATURE.md for details

NEW_FEATURE_THRESHOLD: 0.5
# Description: Threshold value for feature activation
# Default: 0.5
# Range: 0.0 to 1.0
# Impact: Higher values = more conservative behavior
```

### Documentation Update Workflow

#### Step-by-Step Process

```bash
# 1. IMPLEMENT FEATURE
git add src/new_feature.py
git commit -m "feat(feature): implement new functionality"

# 2. UPDATE MAIN README
# Edit README.md - add feature, configuration, examples
git add README.md
git commit -m "docs(readme): document new feature"

# 3. UPDATE CHANGELOG
# Edit CHANGELOG.md - add to [Unreleased] section
git add CHANGELOG.md
git commit -m "docs(changelog): add new feature entry"

# 4. UPDATE ADD-ON DOCUMENTATION
# Edit ml_heating/README.md and ml_heating_dev/README.md
git add ml_heating/README.md ml_heating_dev/README.md
git commit -m "docs(addon): update add-on documentation"

# 5. UPDATE CONFIGURATION EXAMPLES
# Edit .env_sample and config files
git add .env_sample ml_heating/config.yaml ml_heating_dev/config.yaml
git commit -m "docs(config): add new configuration parameters"

# 6. UPDATE MEMORY BANK
# Edit relevant memory-bank files
git add memory-bank/systemPatterns.md memory-bank/activeContext.md
git commit -m "docs(memory-bank): document new architecture patterns"

# 7. CREATE ALPHA RELEASE
git tag v0.2.0-alpha.1
git push origin v0.2.0-alpha.1
```

### Documentation Quality Standards

#### DO's ✅
- ✅ Update documentation in the same PR as code changes
- ✅ Include code examples for new features
- ✅ Document all configuration parameters
- ✅ Explain WHY a feature exists, not just WHAT it does
- ✅ Include screenshots for UI changes
- ✅ Update CHANGELOG.md for every change
- ✅ Reference GitHub issue numbers
- ✅ Use clear, concise language

#### DON'Ts ❌
- ❌ Commit code without documentation updates
- ❌ Use technical jargon without explanation
- ❌ Leave configuration parameters undocumented
- ❌ Skip CHANGELOG.md entries
- ❌ Forget to update add-on README files
- ❌ Assume users understand implementation details
- ❌ Leave outdated documentation in place

### Documentation Review Checklist

Before creating alpha release, verify:
```markdown
## Documentation Completeness
- [ ] README.md updated with new features
- [ ] CHANGELOG.md includes all changes
- [ ] Add-on documentation synchronized
- [ ] Configuration examples updated
- [ ] Code examples provided and tested
- [ ] Memory bank reflects current architecture
- [ ] No outdated information remains
- [ ] All links and references work
- [ ] Screenshots updated if UI changed
- [ ] Breaking changes clearly marked
```

## Documentation Maintenance

### Memory Bank Updates

#### When to Update Memory Bank
- **After major alpha releases**: Document new features and architecture changes
- **Before stable releases**: Ensure all documentation reflects current state
- **When user requests**: **"update memory bank"** trigger comprehensive review
- **After workflow changes**: Update development processes and CI/CD changes

#### Memory Bank Update Process
```bash
# 1. Review all memory bank files (required for triggered updates)
git status memory-bank/

# 2. Update key files based on recent changes
# - activeContext.md: Current development phase and recent accomplishments
# - systemPatterns.md: Architecture changes and new patterns
# - versionStrategy.md: Release strategy updates
# - developmentWorkflow.md: Process improvements

# 3. Commit memory bank updates
git add memory-bank/
git commit -m "docs(memory-bank): update for alpha architecture v0.1.0-alpha.8

- Document successful multi-add-on implementation
- Update workflow processes for alpha/stable channels  
- Capture lessons learned from t0bst4r-inspired approach"
```

### Project Documentation Updates

#### Documentation Files Requiring Updates
- **README.md**: Dual-channel installation instructions
- **docs/INSTALLATION_GUIDE.md**: Separate alpha vs stable installation
- **docs/CONTRIBUTOR_WORKFLOW.md**: Alpha development process
- **CHANGELOG.md**: Track both alpha and stable releases

#### Documentation Standards
```markdown
# Example installation section structure

## Installation

### Stable Channel (Recommended for Production)
Use this for production heating control with automatic updates.

### Alpha Channel (Testing and Development)  
Use this to test latest features and provide feedback to development.
```

## Troubleshooting Common Issues

### Workflow Build Failures

#### Alpha Workflow Issues
```bash
# Check workflow status
gh run list --workflow="Build Development Release"

# View specific run details
gh run view <run-id>

# Common issues:
# - Missing files in build context
# - HA linter validation failures  
# - Docker tag format problems
# - Platform build failures
```

#### Stable Workflow Issues
```bash
# Verify stable workflow triggers correctly
gh run list --workflow="Build Stable Release"

# Common issues:
# - Alpha release accidentally triggering stable
# - Version format problems
