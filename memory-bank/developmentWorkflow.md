# Development Workflow - ML Heating Project

## Overview

This document captures the development workflows, tools, and processes used in the ML Heating project. It serves as a reference for maintaining consistent development practices and efficient project management.

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

### Label Strategy
**Standard Labels Available**:
- `enhancement` - Feature requests and improvements
- `bug` - Bug reports and fixes
- `documentation` - Documentation updates
- `question` - Support questions

**Note**: Check available labels with `gh label list` before creating issues to avoid "label not found" errors.

### Issue Documentation Process

#### When to Create GitHub Issues
1. **Feature Requests**: New functionality or significant enhancements
2. **Bug Reports**: Issues with current functionality
3. **Documentation**: Missing or incorrect documentation
4. **Architecture Changes**: Major system modifications
5. **User Experience**: UI/UX improvements

#### Issue Content Standards
**Feature Request Template**:
```markdown
# Feature Request: [Title]

## Current State
- Brief description of current functionality

## Requested Enhancement
- Detailed description of desired feature
- Benefits and use cases

## Implementation Plan
- Technical approach
- Breaking changes (if any)
- Migration strategy

## Success Criteria
- [ ] Acceptance criteria
- [ ] Testing requirements
- [ ] Documentation updates

## Priority: [High/Medium/Low]
**Rationale**: Why this feature is needed

## Labels
- enhancement
- [additional relevant labels]
```

**Bug Report Template**:
```markdown
# Bug: [Title]

## Problem Description
- What's happening vs what should happen

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Environment
- Version: 
- Platform:
- Configuration:

## Expected vs Actual Behavior
**Expected**: 
**Actual**: 

## Error Messages
```
[error logs or screenshots]
```

## Impact
- Severity level and user impact

## Labels
- bug
- [additional relevant labels]
```

### Issue Workflow Integration

#### Local Documentation → GitHub Issue Process
1. **Draft locally** - Create detailed markdown file for complex issues
2. **Review content** - Ensure completeness and clarity
3. **Create issue** - Use `gh issue create --body-file` command
4. **Clean up** - Remove local draft file after successful creation
5. **Reference** - Link to issue in commits: `Closes #10` or `Refs #10`

#### Example: Feature Enhancement Process
```bash
# 1. Create detailed enhancement document
nano enhancement_proposal.md

# 2. Create GitHub issue from document  
gh issue create --title "Feature Request: Enhancement Title" --body-file enhancement_proposal.md --label "enhancement"

# 3. Note the issue URL (e.g., https://github.com/user/repo/issues/10)

# 4. Clean up local file
rm enhancement_proposal.md

# 5. Reference in commits
git commit -m "feat: implement new feature (refs #10)"
```

## Home Assistant Add-on Development

### Add-on Discovery Requirements
**Critical Configuration**:
- Version must use valid semantic versioning (not `"dev"`)
- Add-on files must be on `main` branch (Home Assistant only reads default branch)
- Container image must exist and be accessible
- `config.yaml` structure must be valid

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

### Container Build Process
**Tag-based Builds**:
- Git tags trigger GitHub Actions container builds
- Container name: `ghcr.io/helgeerbe/ml_heating:v{version}`
- Ensure tag matches `config.yaml` version exactly

### Testing Add-on Discovery
1. **Remove repository** from HA (Settings → Add-ons → Add-on Store → ⋮ → Repositories)
2. **Wait 2-3 minutes** for cache clearance
3. **Re-add repository**: `https://github.com/helgeerbe/ml_heating`
4. **Verify add-on appears** with correct version

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

### Release Process
1. **Tag version** - Create semantic version tag
2. **Build containers** - Verify GitHub Actions success
3. **Test add-on** - Verify HA discovery and installation
4. **Update documentation** - Memory bank and README updates
5. **Close issues** - Reference resolved issues in commits

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

---

**Last Updated**: November 27, 2025  
**Next Review**: When implementing new development tools or major workflow changes

This workflow documentation ensures consistent development practices and efficient project management for the sophisticated ML Heating control system.
