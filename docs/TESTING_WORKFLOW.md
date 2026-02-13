# ML Heating Testing Workflow

## 📋 Overview

This document outlines the testing workflow for the ML heating system, which is built on a strict **Test-Driven Development (TDD)** methodology. All new features and bug fixes must begin with writing tests. The project maintains a 100% test success rate, with all 236 tests currently passing.

## 🏗️ Test Architecture

### **Unit and Integration Tests (`tests/` directory)**

- **Structure**: The `tests/` directory is organized into `unit/` and `integration/` subdirectories.
    - **`tests/unit`**: Contains tests for individual components, algorithms, and functions in isolation. These tests are fast and use mocked data to ensure deterministic execution.
    - **`tests/integration`**: Contains tests that verify the interaction between multiple components.
- **Purpose**: To provide a comprehensive suite of fast, automated tests for CI/CD pipelines and local development.
- **Runtime**: The entire suite runs in seconds for quick feedback.

### **Validation Scripts (`validation/` directory)**
- **Purpose**: End-to-end validation with real data.
- **Runtime**: Minutes for comprehensive analysis.
- **Scope**: Complete workflows, system integration, and performance benchmarking against real InfluxDB historical data.

### **Container Validation (`validate_container.py`)**
- **Purpose**: Validates the Home Assistant Add-on deployment, including the dashboard, configuration files, Dockerfile, and build system.
- **Usage**: Part of the CI/CD deployment pipeline to ensure a valid and installable add-on.

## 🚀 **TDD Development Workflow**

All development in this project follows a strict Test-Driven Development (TDD) cycle:

1.  **Write a Failing Test**: Before writing any implementation code, write a test that captures the requirements of the new feature or bug fix. Run the test to ensure it fails as expected.
2.  **Write Code to Pass the Test**: Write the minimum amount of code required to make the failing test pass.
3.  **Refactor**: With the safety of a passing test suite, refactor the code to improve its design, readability, and performance.

This TDD approach ensures that the codebase remains robust, maintainable, and fully tested.

## 🛠️ **Running Tests**

### **Execute the Full Test Suite**
```bash
# Run all 236+ tests
pytest tests/
```

### **Run a Specific Test File**
```bash
# Run all tests in a specific file
pytest tests/unit/test_heating_controller.py
```

### **Run a Specific Test**
```bash
# Run a single test method from a test class
pytest tests/unit/test_heating_controller.py::TestHeatBalanceController::test_charging_mode
```

### **Run Tests with Coverage**
```bash
# Run tests and generate an HTML coverage report
pytest tests/ --cov=src --cov-report=html
```

## 🧪 **Testing Best Practices**

- **Write Tests First**: Adhere to the TDD workflow for all changes.
- **Keep Unit Tests Fast**: Unit tests should be small, focused, and fast.
- **Mock External Dependencies**: Use `unittest.mock` to mock external services like Home Assistant and InfluxDB to ensure tests are deterministic and fast.
- **Test for Edge Cases**: Write tests for boundary conditions, invalid inputs, and error states.
- **Maintain 100% Test Success**: No code will be merged if it breaks any existing tests.

By following this TDD workflow, we ensure the ML Heating project maintains the highest standards of quality and reliability.