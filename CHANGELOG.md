# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive spam detection system combining AI and user feedback
- User management and password reset functionality
- Web dashboard with frontend and backend components
- Comment analysis capabilities with sentiment analysis
- Bulk spam marking functionality
- Video import functionality
- Metrics tracking and analysis features
- Detailed toxicity analysis system
- User registration and authentication system

### Fixed
- Resolved ModuleNotFoundError for utils.logger
- Fixed UI issues when marking comments as spam
- Corrected import statements and package structure
- Fixed spam detection and prediction methods
- Resolved database connection handling issues
- Fixed feature consistency in training and prediction
- Corrected logger implementation and error handling
- Fixed frontend/backend mismatches
- Resolved issues with boolean handling in spam marking

### Improved
- Enhanced spam detector metrics tracking
- Improved error detection and handling
- Enhanced database schema and initialization
- Updated comment analysis with better toxicity scoring
- Improved prediction method with better thresholds
- Enhanced spam detection features and whitelist system
- Optimized database operations and connection management
- Improved UI components for spam management
- Enhanced metrics calculation and history tracking

### Changed
- Updated to use new OpenAI API client
- Simplified classification categories
- Restructured application with blueprints
- Modified spam detection threshold for more conservative detection
- Updated schema to include new required fields
- Reorganized code for better testability
- Updated requirements and dependencies

### Added Infrastructure
- Added unit tests
- Created database initialization scripts
- Added comprehensive error logging
- Implemented proper database indices
- Added .gitignore and environment configuration
- Set up proper package structure

### Documentation
- Added initial README
- Updated requirements documentation
- Added environment configuration examples
