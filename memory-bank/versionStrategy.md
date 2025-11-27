# Version Strategy and Development Workflow

## Project Maturity and Versioning

### Current Phase: **Development/Testing Phase**
The ML Heating project is in active development and testing. We use a two-track deployment system for controlled releases.

## Semantic Versioning Strategy

### Version Number Format: `MAJOR.MINOR.PATCH[-PRERELEASE][.BUILD]`

#### **Development Builds (Testing Phase)**
- **Format**: `0.0.x-dev.N`
- **Examples**: `0.0.1-dev.1`, `0.0.1-dev.2`, `0.0.2-dev.1`
- **Purpose**: Frequent testing deployments, experimental features, bug fixes
- **Frequency**: Multiple per day/week during active development
- **Audience**: Internal testing, early adopters willing to test unstable builds
- **Git Tags**: `v0.0.1-dev.1`, `v0.0.1-dev.2`

#### **Development Releases (Stable Testing)**
- **Format**: `0.0.x`
- **Examples**: `0.0.1`, `0.0.2`, `0.0.3`
- **Purpose**: Tested features, more stable for broader testing
- **Frequency**: When dev builds are tested and features are ready
- **Audience**: Beta testers, community feedback
- **Git Tags**: `v0.0.1`, `v0.0.2`

#### **Beta Releases (Feature Complete)**
- **Format**: `0.x.0`
- **Examples**: `0.1.0`, `0.2.0`
- **Purpose**: Feature-complete versions with all planned functionality
- **Frequency**: Major feature milestones
- **Audience**: Broader testing community
- **Git Tags**: `v0.1.0`, `v0.2.0`

#### **Production Releases (Stable)**
- **Format**: `x.0.0`
- **Examples**: `1.0.0`, `2.0.0`
- **Purpose**: Production-ready, fully supported releases
- **Frequency**: When confident in stability and feature completeness
- **Audience**: General public, production use
- **Git Tags**: `v1.0.0`, `v2.0.0`

## Development Workflow

### **Phase 1: Dev Builds (Current)**
```bash
# Development iteration cycle
1. Make changes/fixes
2. Update version: 0.0.1-dev.1, 0.0.1-dev.2, etc.
3. Commit with conventional message
4. Tag: git tag v0.0.1-dev.N
5. Push: git push origin main && git push origin v0.0.1-dev.N
6. Test deployment
7. Repeat
```

### **Phase 2: Development Release**
```bash
# When dev builds are stable
1. Final testing of dev builds
2. Update version: 0.0.1 (remove -dev suffix)
3. Update changelog
4. Commit: "release: v0.0.1 development release"
5. Tag: git tag v0.0.1
6. Push and announce
```

### **Phase 3: Beta Release**
```bash
# When feature set is complete
1. Version: 0.1.0
2. Comprehensive testing
3. Documentation updates
4. Community feedback integration
```

### **Phase 4: Production Release**
```bash
# When ready for general use
1. Version: 1.0.0
2. Full documentation
3. Support processes
4. Stable API commitment
```

## Git Tag Strategy

### **Tag Patterns**
- **Dev Builds**: `v0.0.x-dev.N` (triggers dev build workflow)
- **Releases**: `v0.0.x`, `v0.x.0`, `vx.0.0` (triggers release workflow)

### **GitHub Actions Integration**
```yaml
# Example workflow triggers
on:
  push:
    tags:
      - 'v*-dev.*'    # Dev builds
      - 'v*'          # Releases (without -dev)
```

### **Container Tagging**
- **Dev builds**: `ghcr.io/helgeerbe/ml_heating:v0.0.1-dev.1`
- **Releases**: `ghcr.io/helgeerbe/ml_heating:v0.0.1`, `ghcr.io/helgeerbe/ml_heating:latest`

## Version Progression Rules

### **Increment Guidelines**
- **Build number (+dev.N)**: Bug fixes, small changes, testing iterations
- **Patch (+0.0.1)**: Bug fixes, minor improvements, no breaking changes
- **Minor (+0.1.0)**: New features, significant improvements, backward compatible
- **Major (+1.0.0)**: Breaking changes, major architecture changes, API changes

### **When to Create Dev Builds**
- Bug fixes that need immediate testing
- Experimental features for feedback
- Configuration changes requiring validation
- Performance improvements needing measurement
- Any change that affects add-on functionality

### **When to Create Releases**
- Set of dev builds tested and stable
- Feature milestones completed
- Community feedback incorporated
- Documentation updated
- Ready for broader user testing

## Current Version Status

**Current Version**: `1.0.0` (incorrect, needs correction)
**Target Version**: `0.0.1-dev.1` (first development build)
**Phase**: Development/Testing
**Next Steps**: Establish proper dev build workflow

## Benefits of This Strategy

### **For Development**
- **Rapid Iteration**: Quick dev builds for testing
- **Clear Progression**: Obvious path from dev to production
- **Risk Management**: Dev builds clearly marked as experimental
- **Feedback Integration**: Multiple opportunities for community input

### **For Users**
- **Clear Expectations**: Version clearly indicates stability level
- **Choice**: Can choose stable releases or help with dev testing
- **Transparency**: Development progress is visible
- **Safety**: Won't accidentally install unstable versions

### **For Project Management**
- **Milestone Tracking**: Clear version milestones
- **Release Planning**: Structured approach to releases
- **Quality Control**: Multiple testing phases before production
- **Documentation**: Changelog tracks all changes

This strategy provides professional project management while maintaining flexibility for rapid development and testing iterations.
