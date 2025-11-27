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
