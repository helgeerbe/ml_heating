# Development Workflow - ML Heating Project

## Overview

This document captures the comprehensive development workflows, tools, and processes used in the ML Heating project. It serves as a reference for maintaining consistent development practices, efficient project management, and professional build processes.

## Git Workflow

### Branch Strategy
- **`main`** - Production-ready code, Home Assistant add-on discovery source
- **`dev`** - Development branch for new features and testing
- **`ha-addon`** - Home Assistant add-on specific development
- **`refactor`** - Large-scale refactoring work

### Semantic Versioning Strategy
- **Development versions**: `0.x.x-dev.x` (e.g., `0.1.0-dev.3`)
- **Stable versions**: `x.x.x` (e.g., `1.0.0`) 
- **Container tags**: Match version exactly (e.g., `v0.1.0-dev.3`)

### Key Git Commands
```bash
# Check current status and branch
git status
git branch -a

# Switch between branches
git checkout main
git checkout dev

# Standard commit workflow
git add <files>
git commit -m "type(scope): description"
git push origin <branch>

# Version tagging for releases
git tag v0.1.0-dev.3
git push origin v0.1.0-dev.3
```

## GitHub Issue Management

### GitHub CLI Setup
The project uses GitHub CLI (`gh`) for efficient issue management. Ensure `gh` is installed and authenticated.

### Issue Creation Workflow

#### Method 1: Direct CLI Issue Creation
```bash
# Create issue from markdown file
gh issue create --title "Feature Request: Title Here" --body-file <filename.md> --label "enhancement"

# Create simple issue with inline content
gh issue create --title "Bug: Issue Title" --body "Issue description" --label "bug"

# List available labels first (to avoid label errors)
gh label list
```

#### Method 2: Issue Template Approach
```bash
# Create issue using repository template
gh issue create --template feature_request.md
gh issue create --template bug_report.md
```

### Issue Management Commands
```bash
# List issues
gh issue list
gh issue list --label "enhancement"
gh issue list --state "open"

# View specific issue
gh issue view 10

# Update issue
gh issue edit 10 --title "New Title"
gh issue edit 10 --add-label "priority:high"

# Close issue
gh issue close 10 --comment "Resolved in commit abc123"
```

### Local Documentation → GitHub Issue Process
1. **Draft locally** - Create detailed markdown file for complex issues
2. **Review content** - Ensure completeness and clarity
3. **Create issue** - Use `gh issue create --body-file` command
4. **Clean up** - Remove local draft file after successful creation
5. **Reference** - Link to issue in commits: `Closes #10` or `Refs #10`

## Development Build Process

### Current Status
**Latest Development Build**: `v0.1.0-dev.3`
- **Purpose**: Fix Home Assistant add-on discovery with proper semantic versioning
- **Container**: `ghcr.io/helgeerbe/ml_heating:v0.1.0-dev.3`

### When to Create Development Builds
- Bug fixes that need immediate testing
- Experimental features requiring feedback
- Configuration changes needing validation
- Performance improvements requiring measurement
- Any changes affecting add-on functionality

### Development Build Workflow
```bash
# 1. Make changes to code/configuration
# 2. Update version in ml_heating_addon/config.yaml
version: "0.1.0-dev.4"  # Increment dev build number

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
git tag v0.1.0-dev.4
git push origin main && git push origin v0.1.0-dev.4

# 6. Monitor GitHub Actions for build success
# 7. Test deployment in Home Assistant
```

### Development Build Naming Convention
- **Format**: `0.x.x-dev.N`
- **Examples**: 
  - `0.1.0-dev.3` - Current development build
  - `0.1.0-dev.4` - Next iteration with fixes
  - `0.2.0-dev.1` - New minor version development
- **Git Tags**: `v0.1.0-dev.3`, `v0.1.0-dev.4`

## Release Build Process

### When to Create Release Builds
- Multiple dev builds tested and stable
- Feature milestones completed
- Community feedback incorporated
- Documentation updated
- Ready for broader user testing

### Release Build Workflow
```bash
# 1. Ensure all dev builds are tested and stable
# 2. Update version to remove -dev suffix
version: "0.1.0"  # Clean version number

# 3. Move CHANGELOG.md [Unreleased] entries to versioned section
## [0.1.0] - 2025-XX-XX
### Added
- Consolidated features from dev builds
- Tested and validated functionality

# 4. Commit release
git commit -m "release: v0.1.0 development release

Stable development release incorporating tested changes from:
- v0.1.0-dev.3: Home Assistant discovery fix
- v0.1.0-dev.4: Additional improvements

Ready for broader beta testing community."

# 5. Create release tag
git tag v0.1.0
git push origin main && git push origin v0.1.0

# 6. Monitor build and container publication
# 7. Announce to community for testing
```

### Release Build Naming Convention
- **Development Releases**: `0.x.0` (e.g., 0.1.0, 0.2.0)
- **Beta Releases**: `0.x.0` with broader testing (e.g., 0.5.0, 0.8.0)  
- **Production Releases**: `x.0.0` (e.g., 1.0.0, 2.0.0)
- **Git Tags**: `v0.1.0`, `v0.5.0`, `v1.0.0`

## GitHub Actions Integration

### Workflow Triggers
- **Dev Builds**: Tags matching `v*-dev.*` pattern
- **Releases**: Tags matching `v*` pattern (without -dev)

### Build Process
1. **Multi-architecture container build**
2. **Publishing to GitHub Container Registry (GHCR)**
3. **Container tagging**:
   - Dev: `ghcr.io/helgeerbe/ml_heating:v0.1.0-dev.3`
   - Release: `ghcr.io/helgeerbe/ml_heating:v0.1.0`, `:latest`

### Monitoring Builds
- Check GitHub Actions tab after pushing tags
- Verify container publication in repository packages
- Test Home Assistant add-on discovery and installation

## Home Assistant Add-on Development

### Add-on Discovery Requirements
**Critical Configuration**:
- Version must use valid semantic versioning (not `"dev"`)
- Add-on files must be on `main` branch (Home Assistant only reads default branch)
- Container image must exist and be accessible
- `config.yaml` structure must be valid

### Repository Configuration
- **Repository URL**: `https://github.com/helgeerbe/ml_heating`
- **Add-on Path**: `ml_heating_addon/`
- **Discovery Requirements**:
  - Valid `config.yaml` with proper semantic versioning
  - Proper `build.json` for container builds
  - `repository.json` at root level

### Add-on Version Management
```bash
# Fix version for HA discovery
# Edit ml_heating_addon/config.yaml
version: "0.1.0-dev.3"  # Valid semantic version

# Commit and push to main
git add ml_heating_addon/config.yaml
git commit -m "fix(addon): update version to 0.1.0-dev.3 for HA discovery"
git push origin main
```

### Testing Add-on Discovery
1. **Remove repository** from HA (Settings → Add-ons → Add-on Store → ⋮ → Repositories)
2. **Wait 2-3 minutes** for cache clearance
3. **Re-add repository**: `https://github.com/helgeerbe/ml_heating`
4. **Verify add-on appears** with correct version

### Expected Add-on Appearance
- **Name**: "ML Heating Control"
- **Current Version**: "0.1.0-dev.3" 
- **Description**: "Physics-based machine learning heating control system with online learning"

## Documentation Standards

### Memory Bank Updates
**When to Update**:
- After implementing significant features
- When user requests with **"update memory bank"**
- After architectural changes
- During troubleshooting sessions that reveal new insights

**Update Process**:
1. **Review ALL memory bank files** (required when triggered)
2. **Focus on `activeContext.md`** - current state and decisions
3. **Update `progress.md`** - development status
4. **Document patterns** in relevant files
5. **Preserve technical accuracy** and implementation details

### Commit Message Standards
```bash
# Format: type(scope): description
git commit -m "feat(addon): add dual channel architecture"
git commit -m "fix(discovery): update version for HA compatibility" 
git commit -m "docs(memory): add GitHub workflow documentation"
git commit -m "refactor(physics): improve error handling"
```

**Types**: feat, fix, docs, refactor, test, chore, style, perf

## Development Tools

### Required CLI Tools
```bash
# Essential tools
git --version          # Version control
gh --version          # GitHub CLI for issue management
docker --version      # Container builds
code --version        # VS Code editor
python --version      # Python development

# Optional but useful
make --version        # Build automation
curl --version        # API testing
wget --version        # File downloads
```

### Development Environment Setup
```bash
# Clone repository
git clone https://github.com/helgeerbe/ml_heating.git
cd ml_heating

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Set up GitHub CLI
gh auth login
gh repo set-default helgeerbe/ml_heating
```

## Quality Assurance

### Pre-commit Checklist
- [ ] **Version consistency** - `config.yaml` matches expected container tag
- [ ] **Branch alignment** - Add-on changes committed to `main` branch
- [ ] **Issue references** - Commits reference relevant GitHub issues
- [ ] **Documentation updates** - Memory bank updated for significant changes
- [ ] **Testing** - Changes tested in appropriate environment

### Before Every Development Build
- [ ] Code changes tested locally
- [ ] Configuration syntax validated
- [ ] Version number incremented properly
- [ ] CHANGELOG.md updated in [Unreleased]
- [ ] Commit message follows conventional format

### Before Every Release
- [ ] All related dev builds tested successfully
- [ ] Community feedback reviewed and incorporated
- [ ] Documentation updated and accurate
- [ ] CHANGELOG.md entries moved to versioned section
- [ ] Version number updated to clean format (no -dev)

### After Every Tag Push
- [ ] Monitor GitHub Actions for build success
- [ ] Verify container publication to GHCR
- [ ] Test Home Assistant add-on discovery
- [ ] Validate installation and basic functionality

## Project-Specific Workflows

### Physics Model Calibration
```bash
# Calibrate physics model with historical data
python src/main.py --calibrate-physics

# Monitor calibration results
tail -f logs/ml_heating.log
```

### Container Development
```bash
# Local container build and test
cd ml_heating_addon
docker build -t ml_heating_test .
docker run -it ml_heating_test

# Push to registry (automated via GitHub Actions)
git tag v0.1.0-dev.4
git push origin v0.1.0-dev.4
```

### Notebook Development
```bash
# Start Jupyter server
cd notebooks
jupyter lab

# Import helper modules
from notebook_imports import *  # Loads all required modules
```

## Troubleshooting Common Issues

### Add-on Discovery Problems
1. **Check version format** - Must be valid semantic versioning
2. **Verify main branch** - HA only reads default branch
3. **Container availability** - Ensure image exists at registry
4. **Clear HA cache** - Remove and re-add repository

### Build Failures
1. Review GitHub Actions logs
2. Validate YAML syntax in configuration files
3. Check container build process
4. Verify multi-architecture compatibility

### Version Conflicts
1. Ensure version increments are logical
2. Check for duplicate tags
3. Validate changelog consistency
4. Review commit message format

### GitHub CLI Issues
```bash
# Authentication problems
gh auth status
gh auth refresh

# Label not found errors
gh label list  # Check available labels first

# Repository context issues
gh repo set-default owner/repo
```

### Git Branch Confusion
```bash
# Check current status
git status
git branch -a

# Sync with remote
git fetch origin
git pull origin main
```

## Best Practices

### Issue Management
- **Draft complex issues locally** first for review
- **Use descriptive titles** that explain the request
- **Include implementation details** for feature requests
- **Reference issues in commits** for traceability
- **Close with commit references** when resolving

### Development Workflow
- **Small, focused commits** with clear messages
- **Test on main branch** before major releases
- **Document architectural decisions** in memory bank
- **Maintain version consistency** across files
- **Use semantic versioning** throughout project

### Documentation
- **Update memory bank** for significant changes
- **Include code examples** in documentation
- **Maintain accuracy** of technical details
- **Structure hierarchically** across files
- **Focus on practical operations** and deployment

## Benefits of This Workflow

### Development Efficiency
- **Rapid iteration** with frequent dev builds
- **Clear progression** from experimental to stable
- **Community feedback** integration at multiple stages
- **Risk mitigation** through staged releases

### Project Quality
- **Structured versioning** prevents confusion
- **Professional appearance** builds community trust
- **Documentation standards** ensure maintainability
- **Automated processes** reduce human error

### User Experience
- **Version transparency** sets clear expectations
- **Choice of stability** (dev vs release builds)
- **Predictable updates** following semantic versioning
- **Quality assurance** through testing phases

---

**Last Updated**: November 27, 2025  
**Next Review**: When implementing new development tools or major workflow changes

This workflow documentation ensures consistent development practices and efficient project management for the sophisticated ML Heating control system, combining comprehensive GitHub CLI integration with professional build and release processes.
