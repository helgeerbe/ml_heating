# Changelog and Commit Standards

## Changelog Format (Keep a Changelog Standard)

We follow the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format for consistent, readable changelogs.

### Changelog File Structure

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Features in development, not yet released

## [0.0.1-dev.1] - 2024-XX-XX
### Added
- Initial Home Assistant add-on structure
- Basic ML heating control functionality
- Dashboard with overview and control panels

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Fixed add-on discovery issue with proper semantic versioning

### Security
- Nothing yet
```

### Section Definitions

#### **Added**
- New features, capabilities, or functionality
- New configuration options
- New dashboard components
- New API endpoints
- New documentation sections

#### **Changed**
- Changes in existing functionality
- Modified behavior of existing features
- Updated dependencies
- Performance improvements
- UI/UX improvements

#### **Deprecated**
- Features that are still available but marked for future removal
- Configuration options that will be removed
- API endpoints that will be discontinued
- Legacy functionality warnings

#### **Removed**
- Features that have been completely removed
- Configuration options no longer supported
- Discontinued API endpoints
- Deleted files or components

#### **Fixed**
- Bug fixes
- Error corrections
- Performance issue resolutions
- Security vulnerability patches
- Documentation corrections

#### **Security**
- Security-related improvements
- Vulnerability fixes
- Access control updates
- Authentication/authorization changes
- Data protection enhancements

## Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) for consistent, parseable commit messages.

### Format: `type(scope): description`

#### **Types**
- **feat**: New features or functionality
- **fix**: Bug fixes
- **docs**: Documentation changes only
- **style**: Code style changes (formatting, semicolons, etc.)
- **refactor**: Code changes that neither fix bugs nor add features
- **test**: Adding or updating tests
- **chore**: Maintenance tasks (dependencies, build processes, etc.)
- **release**: Version releases and changelog updates

#### **Scopes** (Optional but Recommended)
- **addon**: Home Assistant add-on specific changes
- **dashboard**: Dashboard/UI related changes
- **ml**: Machine learning model or algorithm changes
- **physics**: Physics model or calibration changes
- **config**: Configuration system changes
- **api**: API or external interface changes
- **docs**: Documentation updates
- **ci**: Continuous integration/deployment changes
- **deps**: Dependency updates

#### **Description Guidelines**
- Use imperative mood ("add feature" not "added feature")
- Start with lowercase letter
- No period at the end
- Maximum 50 characters for the first line
- Be specific and descriptive

### Commit Message Examples

#### **Good Examples**
```bash
feat(addon): add entity autocomplete in configuration
fix(physics): resolve calibration drift in cold weather
docs(readme): update installation instructions for HAOS
style(dashboard): improve responsive layout on mobile
refactor(ml): optimize feature extraction pipeline
test(physics): add unit tests for thermal modeling
chore(deps): update numpy to v1.24.0
release: bump version to 0.0.1-dev.2
```

#### **Bad Examples**
```bash
update stuff                    # Too vague
Fixed bug.                      # Not descriptive enough
Added new feature for ML model  # Too long, should use scope
WIP: working on dashboard       # Work in progress shouldn't be committed
feat: Added awesome feature!!!  # Wrong mood, unnecessary punctuation
```

### Multi-line Commit Messages

For complex changes, use multi-line commit messages:

```bash
feat(ml): implement seasonal learning adaptation

- Add cosine/sine seasonal modulation features
- Implement adaptive learning rate for seasonal parameters
- Add configuration options for seasonal learning control
- Include validation for minimum sample requirements

Closes #23
```

## Version Release Commit Messages

### **Development Builds**
```bash
release: v0.0.1-dev.1 development build

### Added
- Initial add-on structure and configuration
- Basic ML heating control functionality
- Dashboard with overview panel

### Fixed
- Add-on discovery issue with semantic versioning
```

### **Development Releases**
```bash
release: v0.0.1 development release

Stable development release incorporating:
- Tested dev builds from v0.0.1-dev.1 to v0.0.1-dev.5
- Community feedback integration
- Documentation updates
- Performance improvements

Ready for broader beta testing.
```

### **Production Releases**
```bash
release: v1.0.0 production release

First stable production release of ML Heating Control.

### Highlights
- Production-ready ML heating control system
- Full Home Assistant integration
- Comprehensive dashboard and monitoring
- Complete documentation and support

This release is recommended for production use.
```

## Workflow Integration

### **Before Every Commit**
1. **Update changelog**: Add entry to [Unreleased] section
2. **Follow commit convention**: Use proper type(scope): description format
3. **Be descriptive**: Explain what and why, not just what

### **Before Every Release**
1. **Review changelog**: Move [Unreleased] entries to versioned section
2. **Add release notes**: Summarize major changes and impact
3. **Update version**: In config.yaml and any other version files
4. **Tag properly**: Follow Git tag strategy from version standards

### **Changelog Automation**
Future consideration: Use tools like `conventional-changelog` to auto-generate changelogs from commit messages.

## Quality Guidelines

### **Changelog Quality**
- **User-focused**: Write for users, not developers
- **Specific**: Include relevant details without being too technical
- **Grouped**: Related changes should be grouped together
- **Prioritized**: Most important changes first in each section

### **Commit Quality**
- **Atomic**: One logical change per commit
- **Complete**: Each commit should represent a working state
- **Tested**: Changes should be tested before committing
- **Reviewable**: Commits should be easy to review and understand

This standard ensures professional project management and helps both developers and users understand project evolution.
