# Version Strategy and Multi-Add-on Deployment

## Project Maturity and Versioning

### Current Phase: **Multi-Channel Deployment Phase**
The ML Heating project now uses a professional dual-channel strategy with alpha and stable releases, inspired by the t0bst4r approach.

## Dual Channel Strategy

### **Stable Channel** (`ml_heating`)
- **Version Format**: `MAJOR.MINOR.PATCH` (semantic versioning)
- **Git Tags**: `v*` (e.g., `v0.1.0`, `v0.2.0`, `v1.0.0`)
- **Container**: `ghcr.io/helgeerbe/ml_heating:{version}`
- **Auto-Updates**: ✅ Enabled for production reliability
- **Target Audience**: Production users, stable deployments
- **Release Frequency**: When features are tested and stable

### **Alpha Channel** (`ml_heating_dev`)
- **Version Format**: `MAJOR.MINOR.PATCH-alpha.BUILD`
- **Git Tags**: `v*-alpha.*` (e.g., `v0.1.0-alpha.1`, `v0.1.0-alpha.8`)
- **Container**: `ghcr.io/helgeerbe/ml_heating:{alpha-version}`
- **Auto-Updates**: ❌ Disabled for safety (manual testing)
- **Target Audience**: Testers, early adopters, developers
- **Release Frequency**: Frequent testing deployments

## Version Number Format: `MAJOR.MINOR.PATCH[-alpha.BUILD]`

### **Alpha Releases (Development Channel)**
- **Format**: `0.1.0-alpha.1`, `0.1.0-alpha.2`, `0.2.0-alpha.1`
- **Purpose**: Testing new features, bug fixes, experimental functionality
- **Frequency**: Multiple per week during active development
- **Audience**: Beta testers, developers, community contributors
- **Add-on Name**: "ML Heating Control (Alpha {version})"
- **Configuration**: DEBUG logging, development API enabled

### **Stable Releases (Production Channel)**
- **Format**: `0.1.0`, `0.2.0`, `1.0.0`
- **Purpose**: Tested, stable functionality for production use
- **Frequency**: When alpha releases are thoroughly tested
- **Audience**: End users, production deployments
- **Add-on Name**: "ML Heating Control"
- **Configuration**: INFO logging, development API disabled

### **Version Progression Examples**

```
Development Cycle Example:
v0.1.0-alpha.1  →  First alpha of v0.1.0
v0.1.0-alpha.2  →  Bug fixes for v0.1.0
v0.1.0-alpha.3  →  Feature additions
v0.1.0-alpha.4  →  Final testing
v0.1.0          →  Stable release

Next Feature Cycle:
v0.1.1-alpha.1  →  Patch improvements
v0.1.1          →  Stable patch

Major Feature Cycle:
v0.2.0-alpha.1  →  New major features
v0.2.0-alpha.5  →  After multiple iterations
v0.2.0          →  Stable feature release
```

## Git Tag Strategy and Workflow Triggers

### **Alpha Development Workflow**
```yaml
# .github/workflows/build-dev.yml
on:
  push:
    tags: ['v*-alpha.*']  # Only alpha tags
```

**Alpha Tag Examples**:
- `v0.1.0-alpha.1` - First alpha build
- `v0.1.0-alpha.8` - Latest testing build
- `v0.2.0-alpha.1` - New feature alpha

### **Stable Release Workflow**
```yaml
# .github/workflows/build-stable.yml
on:
  push:
    tags: ['v*']         # All version tags
    branches-ignore: ['**']

jobs:
  check-release-type:    # Skip alpha releases
    if: ${{ !contains(github.ref, 'alpha') }}
```

**Stable Tag Examples**:
- `v0.1.0` - First stable release
- `v0.2.0` - Major feature release
- `v1.0.0` - Production ready

## Dynamic Version Management (t0bst4r-inspired)

### **Alpha Build Process**
```bash
# Workflow automatically updates version during build
yq eval ".version = \"$VERSION\"" -i ml_heating_addons/ml_heating_dev/config.yaml
yq eval ".name = \"ML Heating Control (Alpha $VERSION)\"" -i ml_heating_addons/ml_heating_dev/config.yaml
```

### **Stable Build Process**
```bash
# Workflow updates stable configuration
sed -i "s/^version: .*/version: \"$VERSION\"/" ml_heating_addons/ml_heating/config.yaml
```

## Development Workflow

### **Alpha Development Cycle**
```bash
# 1. Make changes and test locally
git add .
git commit -m "feat: new heating optimization algorithm"
git push origin main

# 2. Create alpha release for testing
git tag v0.1.0-alpha.9
git push origin v0.1.0-alpha.9

# 3. Workflow automatically:
#    - Validates add-on configuration  
#    - Updates version dynamically
#    - Builds multi-platform containers
#    - Creates alpha release with warnings

# 4. Test alpha release in Home Assistant
# 5. Iterate with more alpha builds if needed
```

### **Stable Release Cycle**
```bash
# When alpha testing is complete
git tag v0.1.0
git push origin v0.1.0

# Workflow automatically:
# - Validates it's not an alpha release
# - Builds stable containers
# - Creates production release
# - Enables auto-updates for users
```

## Version Progression Rules

### **Increment Guidelines**
- **Alpha Build (+alpha.1)**: Bug fixes, small changes, testing iterations
- **Patch (+0.0.1)**: Bug fixes, minor improvements, no breaking changes
- **Minor (+0.1.0)**: New features, significant improvements, backward compatible
- **Major (+1.0.0)**: Breaking changes, major architecture changes

### **When to Create Alpha Releases**
- New feature development and testing
- Bug fixes requiring community validation
- Performance optimizations needing measurement
- Configuration changes requiring testing
- Integration improvements
- Dashboard enhancements

### **When to Create Stable Releases**
- Alpha releases tested and validated by community
- No critical bugs reported for 1+ weeks
- Documentation updated and complete
- All planned features for version complete
- Automated tests passing

## Container Strategy

### **Alpha Containers**
```
ghcr.io/helgeerbe/ml_heating:0.1.0-alpha.1
ghcr.io/helgeerbe/ml_heating:0.1.0-alpha.8
```

### **Stable Containers**  
```
ghcr.io/helgeerbe/ml_heating:0.1.0
ghcr.io/helgeerbe/ml_heating:latest   # Points to latest stable
```

### **Multi-Platform Support**
- `linux/amd64` - Standard x86_64 systems
- `linux/aarch64` - Raspberry Pi 4, newer ARM systems
- `linux/arm/v7` - Raspberry Pi 3, older ARM systems

## Current Version Status

**Latest Alpha**: `v0.1.0-alpha.8` (completed multi-add-on architecture)
**Latest Stable**: Not yet released (pending alpha testing completion)
**Phase**: Alpha Testing and Validation
**Next Steps**: Community testing of alpha builds → stable release

## Multi-Add-on Benefits

### **For Users**
- **Stable Channel**: Reliable, auto-updating production experience
- **Alpha Channel**: Early access to features, contribute to testing
- **Dual Installation**: Run both channels for A/B testing
- **Clear Expectations**: Channel name indicates stability level

### **For Development**
- **Safe Testing**: Alpha channel isolates experimental features
- **Rapid Iteration**: Quick alpha deployment for community feedback
- **Quality Control**: Thorough testing before stable release
- **Professional CI/CD**: Automated multi-platform builds

### **For Project Management**
- **Clear Roadmap**: Alpha → stable progression path
- **Community Engagement**: Alpha testers contribute to quality
- **Risk Management**: Stable users protected from experimental changes
- **Release Planning**: Structured approach with clear milestones

## Release Notes Strategy

### **Alpha Release Notes**
- Include development warnings
- Highlight experimental features
- List known issues and limitations
- Provide testing instructions

### **Stable Release Notes**
- Focus on proven features and improvements
- Include upgrade instructions
- Provide production configuration guidance
- Highlight stability and performance gains

This dual-channel strategy provides professional release management while maintaining rapid development velocity and community engagement through comprehensive alpha testing.
