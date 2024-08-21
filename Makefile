# Makefile for automating documentation, testing, and coverage tasks

# Directories and files
PROJECT_ROOT := .
SPHINX_SOURCE := $(PROJECT_ROOT)/docs/source
SPHINX_BUILD := $(PROJECT_ROOT)/docs/build
MODULE := physped
EXCLUDE_DIRS := $(MODULE)/conf/ $(MODULE)/tests/
REPORTS_DIR := $(SPHINX_BUILD)/html/reports
JUNIT_REPORT := $(REPORTS_DIR)/junit
COVERAGE_REPORT := $(REPORTS_DIR)/coverage

# Default target
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  apidocs          Generate API documentation using Sphinx"
	@echo "  html             Build the HTML documentation"
	@echo "  test-report      Run tests and generate a test report including a badge"
	@echo "  coverage-report  Generate a coverage report and a badge"
	@echo "  all-reports      Generate both test and coverage reports"
	@echo "  build            Build the project using Poetry"
	@echo "  publish          Publish the project using Poetry"
	@echo "  clean            Clean the build directory"

.PHONY: apidocs
apidocs:
	@echo "Generating API documentation..."
	sphinx-apidoc -f -o $(SPHINX_SOURCE) $(MODULE) $(EXCLUDE_DIRS)
	@echo "API documentation generated in $(SPHINX_SOURCE)"

.PHONY: html
html: apidocs
	@echo "Building HTML documentation..."
	make -C docs/ html
	@echo "HTML documentation built in $(SPHINX_BUILD)/html"

.PHONY: test-report
test-report:
	@echo "Running tests and generating reports..."
	coverage run -m pytest $(MODULE)/tests/ --junitxml=$(JUNIT_REPORT)/junit.xml --html=$(JUNIT_REPORT)/report.html
	genbadge tests -v -n "Unit tests" -i $(JUNIT_REPORT)/junit.xml -o $(JUNIT_REPORT)/tests-badge.svg
	@echo "Test reports and badge generated in $(JUNIT_REPORT)"

.PHONY: coverage-report
coverage-report:
	@echo "Generating coverage report..."
	coverage xml -o $(COVERAGE_REPORT)/coverage.xml
	coverage html -d $(COVERAGE_REPORT)
	genbadge coverage -i $(COVERAGE_REPORT)/coverage.xml -o $(COVERAGE_REPORT)/coverage-badge.svg
	@echo "Coverage report and badge generated in $(COVERAGE_REPORT)"
	@echo "Copying reports to $(REPORTS_DIR)..."
	cp -r $(REPORTS_DIR) .

.PHONY: all-reports
all-reports: test-report coverage-report
	@echo "All reports generated."

.PHONY: build
build:
	@echo "Building the project using Poetry..."
	poetry build
	@echo "Build complete."

.PHONY: publish
publish:
	@echo "Publishing the project using Poetry..."
	poetry publish
	@echo "Publish complete."

.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(SPHINX_BUILD)
	rm -rf $(REPORTS_DIR)
	rm README.html
	@echo "Build and reports directories cleaned"
