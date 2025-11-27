# ML Heating Contributor Workflow Guide

## 🎯 Dual-Channel Release Strategy

The ML Heating project uses a **branch-based dual-channel release system** that automatically determines build types and auto-update behavior based on which branch a tag is created from.

## 📋 Channel Overview

| Channel | Branch | Auto-Update | Tag Format | Use Case |
|---------|--------|-------------|------------|----------|
| **🎯 Stable** | `main` | ✅ **Enabled** | `v0.1.0`, `v1.0.0` | Production releases |
| **🚧 Dev** | `dev`, `feature/*` | ❌ **Disabled** | `v0.1.0-dev.1`, `v0.2.0-dev.1` | Development/testing |

## 🔄 Workflow Process

### 🎯 Stable Release Workflow
```bash
# 1. Work is completed and tested on dev/feature branches
git checkout main
git merge dev  # Or merge via PR

# 2. Create stable version tag on main branch
git tag v0.1.0
git push origin v0.1.0

# ✅ Result: Stable build with auto-update enabled
```

### 🚧 Development Release Workflow  
```bash
# 1. Create feature branch or use dev branch
git checkout -b feature/new-model
# or
git checkout dev

# 2. Make changes and commit
git add .
git commit -m "Add new predictive model"

# 3. Create dev version tag (must include -dev.N suffix)
git tag v0.2.0-dev.1
git push origin v0.2.0-dev.1

# ❌ Result: Dev build with auto-update disabled
```

## 🤖 Build System Behavior

### Branch Detection Logic
The GitHub Actions workflow automatically:

1. **Detects which branch** contains the tagged commit
2. **Determines build type** based on branch:
   - `origin/main` → **Stable** build
   - Any other branch → **Dev** build
3. **Sets version and auto-update** accordingly

### Version Processing
```bash
# Stable builds (from main)
v0.1.0 → version: "0.1.0", auto_update: true

# Dev builds (from other branches)  
v0.2.0-dev.1 → version: "dev", auto_update: false
```

### Container Tagging
- **Stable**: `ghcr.io/helgeerbe/ml_heating:v0.1.0`, `:latest`
- **Dev**: `ghcr.io/helgeerbe/ml_heating:v0.2.0-dev.1`

## 📝 Version Naming Rules

### ✅ Valid Tag Examples
```bash
# Stable versions (from main branch)
v0.1.0    # Initial release
v0.2.0    # Feature release
v1.0.0    # Major release
v1.2.3    # Patch release

# Dev versions (from dev/feature branches)  
v0.1.0-dev.1    # First dev build toward v0.1.0
v0.1.0-dev.2    # Second dev build toward v0.1.0
v0.2.0-dev.1    # First dev build toward v0.2.0
v1.0.0-dev.3    # Third dev build toward v1.0.0
```

### ❌ Invalid Tag Examples
```bash
# Dev tags without -dev suffix (will fail build)
v0.1.0-beta     ❌ Must use -dev.N format
v0.2.0-alpha    ❌ Must use -dev.N format
v1.0.0-rc1      ❌ Must use -dev.N format

# Stable tags with dev suffix on main branch (confusing)
v0.1.0-dev.1    ❌ Don't use -dev on main branch
```

## 🔧 Practical Workflows

### Feature Development
```bash
# 1. Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/dashboard-improvements

# 2. Develop and test
# ... make changes ...
git add .
git commit -m "Improve dashboard performance metrics"

# 3. Create dev release for testing
git tag v0.1.1-dev.1
git push origin feature/dashboard-improvements
git push origin v0.1.1-dev.1

# 4. Test the dev release in Home Assistant (manual update required)

# 5. When ready, merge to main for stable release
git checkout main
git merge feature/dashboard-improvements
git tag v0.1.1
git push origin main
git push origin v0.1.1
```

### Hotfix Workflow
```bash
# 1. Create hotfix branch from main
git checkout main
git checkout -b hotfix/critical-fix

# 2. Make critical fix
git add .
git commit -m "Fix critical temperature calculation bug"

# 3. Test with dev release first (optional but recommended)
git tag v0.1.1-dev.1
git push origin hotfix/critical-fix  
git push origin v0.1.1-dev.1

# 4. After testing, merge to main for immediate stable release
git checkout main
git merge hotfix/critical-fix
git tag v0.1.1
git push origin main
git push origin v0.1.1
```

### Ongoing Development
```bash
# Use dev branch for ongoing development
git checkout dev

# Make incremental changes
git add .
git commit -m "Improve seasonal learning algorithm"

# Create dev releases as needed
git tag v0.2.0-dev.1
git push origin dev
git push origin v0.2.0-dev.1

# Continue development
# ... more changes ...
git tag v0.2.0-dev.2
git push origin v0.2.0-dev.2

# When dev work is complete, merge to main
git checkout main  
git merge dev
git tag v0.2.0
git push origin main
git push origin v0.2.0
```

## 🏠 Home Assistant User Experience

### Stable Channel Users
- ✅ **Automatic updates** when new stable releases are published
- 🎯 **Production-ready** code only
- 📧 **Release notifications** for major updates

### Dev Channel Users
- ❌ **Manual updates** required for safety
- 🚧 **Early access** to new features
- 🧪 **Help test** new functionality before stable release
- 🔄 **Switch to stable** anytime by changing repository reference

## 🛠️ Troubleshooting

### Build Fails on Tag Creation
```bash
# Check tag format for dev builds
git tag -l  # List all tags
git tag -d v0.1.0-dev.1  # Delete incorrect tag if needed
git tag v0.1.0-dev.1     # Create correct tag
git push origin v0.1.0-dev.1
```

### Wrong Auto-Update Setting
The build system automatically sets `auto_update` based on branch context:
- No manual configuration needed
- Issue likely due to tagging wrong branch
- Check which branch contains your tag: `git branch -r --contains <tag>`

### Dev Build Not Updating
This is **expected behavior**:
- Dev builds have `auto_update: false` for safety
- Users must manually update dev versions
- Prevents unstable code from auto-installing

## 📊 Release Dashboard

Monitor releases at:
- **GitHub Releases**: https://github.com/helgeerbe/ml_heating/releases
- **Container Registry**: https://github.com/helgeerbe/ml_heating/pkgs/container/ml_heating
- **Actions**: https://github.com/helgeerbe/ml_heating/actions

## 🎉 Benefits

### For Contributors
- ✅ **Safe testing** with dev channel
- 🚀 **Fast iteration** without affecting stable users  
- 🔒 **Automatic safeguards** prevent accidental stable releases
- 📈 **Clear deployment path** from dev to stable

### For Users
- 🎯 **Stable experience** with auto-updates (stable channel)
- 🧪 **Early access** to new features (dev channel)
- 🔒 **Safety** from unstable auto-updates
- 🎛️ **Choice** between stability and features

This dual-channel system ensures both rapid development and stable production deployments! 🚀
