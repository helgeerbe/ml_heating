# ML Heating Contributor Workflow Guide

## 🎯 Alpha-Based Dual-Channel Release Strategy

The ML Heating project uses an **alpha-based dual-channel release system** inspired by the t0bst4r approach. This professional deployment strategy provides separate stable and alpha channels with dynamic version management during builds.

## 📋 Channel Overview

| Channel | Add-on Name | Auto-Update | Tag Format | Container | Use Case |
|---------|-------------|-------------|------------|-----------|----------|
| **🎯 Stable** | "ML Heating Control" | ✅ **Enabled** | `v0.1.0`, `v1.0.0` | `ghcr.io/helgeerbe/ml_heating:0.1.0` | Production releases |
| **🚧 Alpha** | "ML Heating Control (Alpha {version})" | ❌ **Disabled** | `v0.1.0-alpha.1` | `ghcr.io/helgeerbe/ml_heating:0.1.0-alpha.1` | Testing/development |

## 🏗️ Multi-Add-on Architecture

### Dual Add-on Structure
```
ml_heating_addons/
├── ml_heating/          # Stable channel add-on
│   ├── config.yaml      # Production configuration
│   └── build.json       # Stable build context
└── ml_heating_dev/      # Alpha channel add-on
    ├── config.yaml      # Development configuration  
    └── build.json       # Alpha build context
```

### Dynamic Version Management (t0bst4r-inspired)
The build workflows automatically update add-on configurations during builds:

**Alpha Builds:**
```bash
# Extract version from git tag: v0.1.0-alpha.8 → 0.1.0-alpha.8
yq eval ".version = \"$VERSION\"" -i ml_heating_addons/ml_heating_dev/config.yaml
yq eval ".name = \"ML Heating Control (Alpha $VERSION)\"" -i ml_heating_addons/ml_heating_dev/config.yaml
```

**Stable Builds:**
```bash
# Update stable configuration: v0.1.0 → 0.1.0
sed -i "s/^version: .*/version: \"$VERSION\"/" ml_heating_addons/ml_heating/config.yaml
```

## 🔄 Workflow Process

### 🚧 Alpha Release Workflow (Main Development Flow)
```bash
# 1. Make changes and commit to main branch
git add .
git commit -m "feat(physics): improve seasonal learning algorithm"
git push origin main

# 2. Create alpha release for community testing
git tag v0.1.0-alpha.9
git push origin v0.1.0-alpha.9

# ✅ Result: 
# - "ML Heating Control (Alpha 0.1.0-alpha.9)" add-on available
# - Auto-updates disabled (manual testing)
# - Development warnings in release notes
# - Multi-platform containers built
```

### 🎯 Stable Release Workflow
```bash
# 1. Alpha releases tested extensively by community
# 2. No critical bugs reported for 1+ weeks
# 3. Documentation updated and complete

# 4. Create stable release
git tag v0.1.0
git push origin v0.1.0

# ✅ Result:
# - "ML Heating Control" add-on available  
# - Auto-updates enabled for production reliability
# - Comprehensive release notes
# - Both alpha and stable add-ons available simultaneously
```

## 🤖 Build System Behavior

### Workflow Triggers

**Alpha Development Workflow** (`.github/workflows/build-dev.yml`):
```yaml
on:
  push:
    tags: ['v*-alpha.*']  # Only alpha tags
```

**Stable Release Workflow** (`.github/workflows/build-stable.yml`):
```yaml
on:
  push:
    tags: ['v*']         # All version tags
    branches-ignore: ['**']

jobs:
  check-release-type:    
    if: ${{ !contains(github.ref, 'alpha') }}  # Skip alpha releases
```

### Version Processing Examples
```bash
# Alpha builds
v0.1.0-alpha.1 → 
  - Alpha add-on version: "0.1.0-alpha.1"
  - Alpha add-on name: "ML Heating Control (Alpha 0.1.0-alpha.1)"
  - Container: ghcr.io/helgeerbe/ml_heating:0.1.0-alpha.1

# Stable builds  
v0.1.0 →
  - Stable add-on version: "0.1.0"
  - Stable add-on name: "ML Heating Control"
  - Container: ghcr.io/helgeerbe/ml_heating:0.1.0
  - Latest tag: ghcr.io/helgeerbe/ml_heating:latest
```

## 📝 Version Naming Rules

### ✅ Valid Alpha Tag Examples
```bash
# Alpha development cycle toward v0.1.0
v0.1.0-alpha.1    # First alpha build
v0.1.0-alpha.2    # Bug fixes from community testing
v0.1.0-alpha.3    # Feature additions
v0.1.0-alpha.8    # Final testing iteration

# Next feature development cycle
v0.2.0-alpha.1    # New major features
v0.2.0-alpha.5    # After multiple iterations

# Patch development
v0.1.1-alpha.1    # Patch improvements testing
```

### ✅ Valid Stable Tag Examples
```bash
v0.1.0    # First stable release (after alpha testing)
v0.2.0    # Major feature release  
v0.1.1    # Patch release
v1.0.0    # Production-ready major release
```

### ❌ Invalid Tag Examples
```bash
# Old "dev" format (no longer used)
v0.1.0-dev.1     ❌ Use -alpha.N instead
v0.2.0-dev.2     ❌ Use -alpha.N instead

# Other prerelease formats (not supported)
v0.1.0-beta.1    ❌ Use -alpha.N format
v0.2.0-rc1       ❌ Use -alpha.N format
v1.0.0-pre       ❌ Use -alpha.N format
```

## 🔧 Practical Workflows

### Feature Development Workflow
```bash
# 1. Develop new feature on main branch
git checkout main
git pull origin main

# ... implement feature ...
git add .
git commit -m "feat(heating): add multi-zone temperature control"
git push origin main

# 2. Create alpha release for community testing
git tag v0.2.0-alpha.1
git push origin v0.2.0-alpha.1

# 3. Community tests alpha release
# - Alpha add-on appears in Home Assistant
# - Users install and provide feedback
# - Issues reported via GitHub

# 4. Iterate based on feedback
git add .
git commit -m "fix(heating): resolve multi-zone configuration issue"
git push origin main

git tag v0.2.0-alpha.2
git push origin v0.2.0-alpha.2

# 5. When alpha testing complete and stable:
git tag v0.2.0
git push origin v0.2.0
# Now both alpha and stable channels available
```

### Hotfix Workflow
```bash
# 1. Critical bug found in stable release
git checkout main

# 2. Create fix
git add .
git commit -m "fix(physics): critical temperature calculation error"
git push origin main

# 3. Optional: Test with alpha first
git tag v0.1.1-alpha.1  
git push origin v0.1.1-alpha.1
# Community can test alpha before stable release

# 4. Create stable hotfix release
git tag v0.1.1
git push origin v0.1.1
# Stable users get automatic update
```

### Feature Branch Development (Advanced)
```bash
# 1. Create feature branch for complex development
git checkout -b feature/advanced-forecasting
git push -u origin feature/advanced-forecasting

# 2. Develop feature
# ... multiple commits ...

# 3. Test with alpha release from feature branch
git tag v0.3.0-alpha.1
git push origin v0.3.0-alpha.1

# 4. When feature complete, merge to main
git checkout main
git merge feature/advanced-forecasting
git push origin main

# 5. Continue alpha testing on main
git tag v0.3.0-alpha.2
git push origin v0.3.0-alpha.2

# 6. Final stable release
git tag v0.3.0
git push origin v0.3.0
```

## 🏠 Home Assistant User Experience

### Stable Channel Users (Production)
- ✅ **Automatic updates** for stable releases
- 🎯 **Production-ready** features only
- 🔒 **Reliability** prioritized over latest features
- 📧 **Release notifications** for major updates

### Alpha Channel Users (Testing Community)
- ❌ **Manual updates** required for safety
- 🧪 **Early access** to new features and improvements
- 🔄 **Frequent releases** during active development
- 💬 **Direct feedback** to development team
- 🚧 **Development warnings** about experimental nature

### Dual Installation (Advanced Users)
- 🎛️ **Both channels** can be installed simultaneously
- 🔄 **A/B testing** between stable and alpha
- 📊 **Performance comparison** between versions
- 🔀 **Easy switching** between channels

## 🛠️ Troubleshooting

### Alpha Build Fails
```bash
# Check tag format
git tag -l | grep alpha

# Delete incorrect tag if needed
git tag -d v0.1.0-alpha.1
git push origin :refs/tags/v0.1.0-alpha.1

# Create correct alpha tag
git tag v0.1.0-alpha.1
git push origin v0.1.0-alpha.1
```

### Stable Build Accidentally Triggered by Alpha
- **Automatic prevention**: Stable workflow skips alpha tags
- **Condition check**: `${{ !contains(github.ref, 'alpha') }}`
- **Result**: No stable build created for alpha tags

### Alpha Add-on Not Appearing
1. **Check build success**: GitHub Actions completed
2. **Verify container**: Published to GitHub Container Registry
3. **Clear HA cache**: Remove and re-add repository
4. **Check version format**: Must be valid semantic version

### Version Confusion
```bash
# Check what version containers exist
gh api repos/helgeerbe/ml_heating/packages/container/ml_heating/versions

# Check which tags triggered which workflows
gh run list --workflow="Build Development Release"
gh run list --workflow="Build Stable Release"
```

## 📊 Release Dashboard & Monitoring

### GitHub Resources
- **Releases**: https://github.com/helgeerbe/ml_heating/releases
- **Container Registry**: https://github.com/helgeerbe/ml_heating/pkgs/container/ml_heating
- **Alpha Workflow**: https://github.com/helgeerbe/ml_heating/actions/workflows/build-dev.yml
- **Stable Workflow**: https://github.com/helgeerbe/ml_heating/actions/workflows/build-stable.yml

### Release Monitoring
```bash
# Monitor workflow runs
gh run list --limit 20

# Check specific workflow run
gh run view <run-id>

# List recent releases
gh release list --limit 10

# View release details
gh release view v0.1.0-alpha.8
```

## 📈 Alpha Testing Guidelines

### For Contributors Creating Alphas
- 🎯 **Clear focus**: Each alpha should test specific features
- 📝 **Detailed notes**: Explain what's new and needs testing
- 🔄 **Iteration ready**: Expect to create multiple alphas
- 📧 **Community engagement**: Monitor feedback and respond to issues

### For Alpha Testers
- 🚨 **Backup first**: Alpha is experimental software
- 📊 **Monitor performance**: Compare against previous stable
- 🐛 **Report issues**: Use GitHub issues for bug reports
- 💡 **Provide feedback**: Suggest improvements and share experience

### Alpha Testing Process
1. **Install alpha add-on** from repository
2. **Monitor for 24-48 hours** for stability
3. **Report any issues** via GitHub
4. **Test new features** and provide feedback
5. **Compare performance** against stable if possible

## 🎉 Benefits of Alpha Architecture

### For Contributors
- ✅ **Safe iteration**: Alpha testing protects stable users
- 🚀 **Fast deployment**: Quick alpha releases for community feedback
- 📊 **Real-world testing**: Community validates changes in production environments
- 🔧 **Professional CI/CD**: Automated multi-platform builds

### For Community
- 🧪 **Early access**: Test features before stable release
- 🤝 **Collaborative development**: Direct feedback shapes final product
- 🔒 **Safety**: Alpha channel clearly marked as experimental
- 🎛️ **Choice**: Pick stability (stable) or features (alpha)

### For Project
- 📈 **Quality assurance**: Community testing improves release quality
- 🔄 **Rapid feedback**: Issues found and fixed before stable
- 🌍 **Broader testing**: Multiple environments and configurations
- 📊 **Professional appearance**: Matches industry-standard practices

This alpha-based dual-channel system provides professional release management while enabling rapid development and comprehensive community testing! 🚀
