# GitHub Label Strategy for ML Heating Project

This document defines the complete labeling strategy for GitHub issues in the ML Heating project, ensuring consistent categorization and efficient project management.

## 🏷️ **Complete Label Set**

### **Type Labels** (What kind of issue?)
- **`enhancement`** - New features and improvements
- **`bug`** - Something isn't working correctly
- **`documentation`** - Improvements or additions to documentation

### **Component Labels** (Which part of the system?)
- **`core`** - Issues related to core ML heating algorithm, models, and physics engine
- **`addon`** - Issues related to Home Assistant addon packaging, configuration, and integration
- **`workflow`** - GitHub Actions, CI/CD, release processes
- **`backup`** - Backup and recovery system functionality

### **Domain Labels** (What area of expertise?)
- **`configuration`** - Configuration options, settings, environment variables
- **`testing`** - Test coverage, validation, quality assurance
- **`automation`** - Automated processes and scheduling
- **`user-experience`** - User interface, usability, accessibility
- **`development`** - Development tools, debugging, developer experience
- **`design`** - Visual design, branding, icons, UI/UX

## 📋 **Label Usage Guidelines**

### **Every Issue Should Have:**
1. **One Type Label**: `enhancement`, `bug`, or `documentation`
2. **One Component Label**: `core` or `addon` (or `workflow` for CI/CD issues)
3. **Zero or More Domain Labels**: As relevant to the specific issue

### **Label Combinations Examples**

#### Core ML Engine Issues
```
enhancement + core + configuration
bug + core + testing
enhancement + core + development
```

#### Home Assistant Addon Issues
```
enhancement + addon + user-experience
bug + addon + configuration
enhancement + addon + backup
```

#### Documentation Issues
```
documentation + core
documentation + addon + user-experience
documentation + development
```

#### Workflow/CI Issues
```
enhancement + workflow + automation
bug + workflow + testing
```

## 🎯 **Specific Issue Type Guidelines**

### **Core ML Engine Issues** (`core`)
Use for issues affecting:
- ML model training and prediction (`src/` directory)
- Physics model and calibration
- Feature engineering and data processing
- Algorithm improvements
- Model performance and accuracy
- InfluxDB integration for data storage
- State management and persistence

**Examples:**
- Physics model calibration improvements
- New feature engineering techniques
- Model performance optimization
- InfluxDB optional mode
- Shadow mode for testing

### **Home Assistant Addon Issues** (`addon`)
Use for issues affecting:
- Home Assistant integration (`ml_heating_addon/` directory)
- Dashboard interface and UI
- Addon configuration and setup
- Container builds and deployment
- Home Assistant entity management
- User-facing features and interface

**Examples:**
- Dashboard enhancements
- Addon configuration improvements
- Installation and setup issues
- User interface improvements
- Backup system features

### **Workflow Issues** (`workflow`)
Use for issues affecting:
- GitHub Actions and CI/CD (`.github/` directory)
- Release processes and versioning
- Container builds and publishing
- Automated testing and validation

**Examples:**
- Dual-channel release system
- Container build optimizations
- Automated testing improvements
- Release workflow enhancements

## 🔄 **Issue Lifecycle and Labels**

### **Priority Indicators** (Optional)
While not formal labels, issues can be marked with priority in the title or description:
- **High Priority**: Critical bugs, security issues, major features
- **Medium Priority**: Important enhancements, non-critical bugs
- **Low Priority**: Nice-to-have features, minor improvements

### **Status Tracking** (Through GitHub features)
- **Open Issues**: Active development targets
- **Assigned Issues**: Currently being worked on
- **Milestones**: Group issues for release planning
- **Projects**: Track progress across multiple issues

## 📊 **Current Issues by Label Category**

### **Type Distribution:**
- `enhancement`: Feature requests and improvements
- `bug`: Issues and problems to fix
- `documentation`: Documentation improvements

### **Component Focus:**
- `core`: ML engine and algorithm development
- `addon`: Home Assistant integration and UI
- `workflow`: Development and deployment processes

### **Domain Expertise:**
- `configuration`: Setup and customization
- `testing`: Quality assurance and validation
- `development`: Developer tools and experience
- `user-experience`: End-user interface and usability

## 🔍 **Label Usage Examples**

### **Recent Issues with Proper Labels:**

1. **"Make InfluxDB optional for simplified setup without calibration"**
   - Labels: `enhancement`, `core`, `configuration`, `user-experience`
   - Rationale: New feature (enhancement) affecting core engine (core) with configuration changes (configuration) that improves user experience (user-experience)

2. **"Add changelog content to GitHub releases"**
   - Labels: `enhancement`, `workflow`, `documentation`
   - Rationale: New feature (enhancement) for release process (workflow) that improves documentation (documentation)

3. **"Design Custom Logo/Icon for Home Assistant Add-on"**
   - Labels: `enhancement`, `addon`, `design`
   - Rationale: New feature (enhancement) for addon interface (addon) involving visual design (design)

4. **"Add shadow mode for testing without affecting real heating control"**
   - Labels: `enhancement`, `core`, `development`, `testing`
   - Rationale: New feature (enhancement) for core engine (core) that helps development (development) and testing (testing)

## 🎯 **Best Practices**

### **When Creating Issues:**
1. **Always include a type label** (`enhancement`, `bug`, `documentation`)
2. **Always include a component label** (`core`, `addon`, `workflow`)
3. **Add relevant domain labels** based on the area of expertise required
4. **Keep labels focused** - use 2-4 labels maximum per issue
5. **Be consistent** - follow the established patterns

### **When Working on Issues:**
1. **Check labels before starting** to understand the scope
2. **Update labels if the scope changes** during development
3. **Use labels for filtering** and project planning
4. **Reference labels in PR descriptions** to maintain traceability

---

## 📝 **Label Maintenance**

This label strategy should be reviewed and updated as the project evolves. New labels should be discussed before creation to maintain consistency and avoid overlap.

**Last Updated**: November 27, 2025
**Next Review**: When significant new components are added to the project
