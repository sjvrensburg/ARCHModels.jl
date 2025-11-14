# CRUSH.md - Agent Guide for ARCHModels.jl

This document helps agents work effectively with the ARCHModels.jl Julia package.

## Project Overview

ARCHModels.jl is a Julia package for ARCH (Autoregressive Conditional Heteroskedasticity) models, designed to capture volatility clustering in financial returns data. The package provides efficient routines for simulating, estimating, and testing various GARCH models.

## Essential Commands

### Testing
- **Run tests**: `julia --project=. -e "using Pkg; Pkg.test()"`
- **Run specific test file**: `julia --project=. test/runtests.jl`
- **Note**: Package may have precompilation issues in some Julia environments

### Documentation
- **Build docs**: `julia --project=docs docs/make.jl`
- **Local docs preview**: Built documentation generates HTML output

### Package Management
- **Activate environment**: `julia --project=.`
- **Install dependencies**: `using Pkg; Pkg.instantiate()`
- **Build package**: `using Pkg; Pkg.build()`

## Code Organization

### Main Source Structure
```
src/
├── ARCHModels.jl              # Main module file with exports and includes
├── univariatearchmodel.jl     # Core univariate ARCH model types and functionality
├── multivariatearchmodel.jl   # Core multivariate ARCH model types
├── general.jl                 # Common utilities and abstract types
├── meanspecs.jl               # Mean specifications (Intercept, ARMA, Regression, etc.)
├── univariatestandardizeddistributions.jl  # Distribution implementations
├── multivariatestandardizeddistributions.jl  # Multivariate distributions
├── EGARCH.jl                  # EGARCH model implementation
├── TGARCH.jl                  # TGARCH model implementation
├── DCC.jl                     # DCC multivariate model implementation
├── tests.jl                   # Statistical tests (ARCH LM, DQ tests)
├── utils.jl                   # Utility functions
└── data/                      # Sample datasets
    ├── bollerslev_ghysels.txt # BG96 dataset
    └── dow29.csv               # DOW29 dataset
```

### Key Abstract Types
- `UnivariateVolatilitySpec{T}` - Abstract supertype for volatility specifications
- `StandardizedDistribution{T}` - Abstract supertype for standardized distributions
- `MeanSpec{T}` - Abstract supertype for mean specifications
- `ARCHModel` - Base type for all ARCH models
- `MultivariateVolatilitySpec{T}` - Abstract supertype for multivariate volatility specs

## Key Patterns and Conventions

### Type System Design
- All models use parameterized types with `T<:AbstractFloat` for numerical precision
- Concrete types follow naming pattern: `Model{p, q, T}` (e.g., `GARCH{1, 1, Float64}`)
- Model specifications are separate from fitted models

### Model Construction Pattern
```julia
# Create specification
spec = GARCH{1, 1}([ω, β₁, α₁])

# Fit to data
am = fit(spec, data)

# Or fit directly
am = fit(GARCH{1, 1}, data)
```

### Data Loading
- Sample datasets available as constants: `BG96`, `DOW29`
- Data is loaded from `src/data/` directory using `readdlm`

## Testing Approach

### Test Structure
- Main test file: `test/runtests.jl`
- Tests use `@testset` for organization
- Stable RNGs used for reproducible tests: `StableRNG(1)`
- Common test data size: `T = 10^4`

### Test Data
- Primary test datasets: `BG96` (Bollerslev & Ghysels), `DOW29` (Dow 29 stocks)
- Simulation-based tests with known parameters for validation

### Common Test Patterns
- Parameter estimation accuracy tests with `isapprox` and `rtol`
- Model selection tests using `selectmodel`
- Statistical tests for model validation
- Exception handling tests with `@test_throws`

## Important Gotchas

### Numerical Stability
- Uses custom `lgamma` implementation to work around ForwardDiff issues
- ForwardDiff compatibility handled through special methods

### Parameter Counting
- `nparams()` function returns number of parameters for specifications
- `NumParamError` thrown for incorrect parameter counts

### Precompilation
- Package uses PrecompileTools.jl for faster loading
- Precompile block in main module with common operations

### Version Compatibility
- Minimum Julia version: 1.0.0
- Uses version-specific conditional compilation for Julia 1.9+

## Dependencies and External Packages

### Key Dependencies
- `Optim` - Parameter estimation via optimization
- `ForwardDiff` - Automatic differentiation
- `Distributions` - Probability distributions
- `StatsBase` - Statistical model interface
- `GLM` - Linear models for regression mean specifications

### Data Dependencies
- Sample datasets included in package (no external data fetching)

## Development Guidelines

### Code Style
- Follow Julia style conventions
- Comprehensive docstrings for all public APIs
- Type annotations required for all public functions

### Adding New Models
- Inherit from appropriate abstract types
- Implement required interface methods: `nparams`, `presample`, `coefnames`
- Add comprehensive tests covering estimation and simulation

### Documentation Updates
- Update reference.md for new public APIs
- Add examples to usage.md for user-facing features
- Maintain type hierarchy documentation

## Build System

### CI/CD
- Uses GitHub Actions for CI
- Tests on Linux, macOS, Windows across multiple Julia versions
- Automatic documentation deployment
- Code coverage reporting via Codecov

### Package Registration
- Registered Julia package in General registry
- Semantic versioning used
- Compatibility bounds defined in Project.toml