# Development Workflow and Build Processes

## Overview

This document describes the established development workflow for the ML Heating project, including the distinction between development builds and release builds.

## Current Status

**First Development Build Created**: `v0.0.1-dev.1`
- **Committed**: c4954fe (release: v0.0.1-dev.1 development build)  
- **Tagged**: v0.0.1-dev.1
- **Pushed**: 2024-11-27 13:47 UTC
- **Purpose**: Establish proper development workflow and fix Home Assistant add-on discovery

## Development Build Process (Testing Phase)

### **When to Create Development Builds**
- Bug fixes that need immediate testing
- Experimental features requiring feedback
- Configuration changes needing validation
- Performance improvements requiring measurement
- Any changes affecting add-on functionality

### **Development Build Workflow**
```bash
# 1. Make changes to code/configuration
# 2. Update version in ml_heating_addon/config.yaml
version: "0.0.1-dev.2"  # Increment dev build number

# 3. Update CHANGELOG.md [Unreleased] section
### Added
- New feature or fix description

# 4. Commit with conventional message
git add .
git commit -m "feat(addon): add new feature description

### Added
- Detailed description of changes
- Impact on functionality

Addresses testing requirement for XYZ."

# 5. Create and push development tag
git tag v0.0.1-dev.2
git push origin main && git push origin v0.0.1-dev.2

# 6. Monitor GitHub Actions for build success
# 7. Test deployment in Home Assistant
```

### **Development Build Naming Convention**
- **Format**: `0.0.x-dev.N`
- **Examples**: 
  - `0.0.1-dev.1` - First development build
  - `0.0.1-dev.2` - Second iteration with fixes
  - `0.0.2-dev.1` - New minor version development
- **Git Tags**: `v0.0.1-dev.1`, `v0.0.1-dev.2`

## Release Build Process (Stable Versions)

### **When to Create Release Builds**
- Multiple dev builds tested and stable
- Feature milestones completed
- Community feedback incorporated
- Documentation updated
- Ready for broader user testing

### **Release Build Workflow**
```bash
# 1. Ensure all dev builds are tested and stable
# 2. Update version to remove -dev suffix
version: "0.0.1"  # Clean version number

# 3. Move CHANGELOG.md [Unreleased] entries to versioned section
## [0.0.1] - 2024-XX-XX
### Added
- Consolidated features from dev builds
- Tested and validated functionality

# 4. Commit release
git commit -m "release: v0.0.1 development release

Stable development release incorporating tested changes from:
- v0.0.1-dev.1: Initial development build
- v0.0.1-dev.2: Bug fixes and improvements

Ready for broader beta testing community."

# 5. Create release tag
git tag v0.0.1
git push origin main && git push origin v0.0.1

# 6. Monitor build and container publication
# 7. Announce to community for testing
```

### **Release Build Naming Convention**
- **Development Releases**: `0.0.x` (e.g., 0.0.1, 0.0.2)
- **Beta Releases**: `0.x.0` (e.g., 0.1.0, 0.2.0)  
- **Production Releases**: `x.0.0` (e.g., 1.0.0, 2.0.0)
- **Git Tags**: `v0.0.1`, `v0.1.0`, `v1.0.0`

## GitHub Actions Integration

### **Expected Workflow Triggers**
- **Dev Builds**: Tags matching `v*-dev.*` pattern
- **Releases**: Tags matching `v*` pattern (without -dev)

### **Build Process**
1. **Multi-architecture container build**
2. **Publishing to GitHub Container Registry (GHCR)**
3. **Container tagging**:
   - Dev: `ghcr.io/helgeerbe/ml_heating:v0.0.1-dev.1`
   - Release: `ghcr.io/helgeerbe/ml_heating:v0.0.1`, `:latest`

### **Monitoring Builds**
- Check GitHub Actions tab after pushing tags
- Verify container publication in repository packages
- Test Home Assistant add-on discovery and installation

## Home Assistant Add-on Discovery

### **Repository Configuration**
- **Repository URL**: `https://github.com/helgeerbe/ml_heating`
- **Add-on Path**: `ml_heating_addon/`
- **Discovery Requirements**:
  - Valid `config.yaml` with proper semantic versioning
  - Proper `build.json` for container builds
  - `repository.json` at root level

### **Version Discovery Process**
1. Home Assistant scans repository for add-on configurations
2. Validates semantic versioning format
3. Lists available versions for installation
4. Users can choose dev builds or stable releases

### **Expected Add-on Appearance**
- **Name**: "ML Heating Control"
- **Current Version**: "0.0.1-dev.1" 
- **Description**: "Physics-based machine learning heating control system with online learning"

## Quality Assurance

### **Before Every Development Build**
- [ ] Code changes tested locally
- [ ] Configuration syntax validated
- [ ] Version number incremented properly
- [ ] CHANGELOG.md updated in [Unreleased]
- [ ] Commit message follows conventional format

### **Before Every Release**
- [ ] All related dev builds tested successfully
- [ ] Community feedback reviewed and incorporated
- [ ] Documentation updated and accurate
- [ ] CHANGELOG.md entries moved to versioned section
- [ ] Version number updated to clean format (no -dev)

### **After Every Tag Push**
- [ ] Monitor GitHub Actions for build success
- [ ] Verify container publication to GHCR
- [ ] Test Home Assistant add-on discovery
- [ ] Validate installation and basic functionality

## Troubleshooting

### **Common Issues**

#### **Add-on Not Appearing in Home Assistant**
1. Check semantic versioning format in config.yaml
2. Verify repository.json structure
3. Refresh Home Assistant add-on store
4. Check GitHub Actions build status

#### **Build Failures**
1. Review GitHub Actions logs
2. Validate YAML syntax in configuration files
3. Check container build process
4. Verify multi-architecture compatibility

#### **Version Conflicts**
1. Ensure version increments are logical
2. Check for duplicate tags
3. Validate changelog consistency
4. Review commit message format

## Benefits of This Workflow

### **Development Efficiency**
- **Rapid iteration** with frequent dev builds
- **Clear progression** from experimental to stable
- **Community feedback** integration at multiple stages
- **Risk mitigation** through staged releases

### **Project Quality**
- **Structured versioning** prevents confusion
- **Professional appearance** builds community trust
- **Documentation standards** ensure maintainability
- **Automated processes** reduce human error

### **User Experience**
- **Version transparency** sets clear expectations
- **Choice of stability** (dev vs release builds)
- **Predictable updates** following semantic versioning
- **Quality assurance** through testing phases

This workflow establishes professional development practices while maintaining flexibility for rapid iteration and community collaboration.
