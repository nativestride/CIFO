# Fantasy League Optimization - Documentation

## Overview
This documentation covers the simplified Fantasy League Optimization website that visualizes optimization algorithms (Hill Climbing, Simulated Annealing, Genetic Algorithm) for team formation in a sports league.

## Features

### Core Features
- **Algorithm Selection**: Choose between Genetic Algorithm, Hill Climbing, and Simulated Annealing
- **Team Visualization**: View optimized teams on a football pitch layout
- **Real-time Metrics**: Monitor optimization progress and fitness values
- **Parameter Controls**: Adjust algorithm parameters before optimization
- **Algorithm Explanation**: Learn about each optimization algorithm

### Simplified Interface
- Read-only view of teams and players
- Direct use of CSV data without database modifications
- Validation of problem constraints (e.g., required number of players per position)
- Clear error messages for unsolvable configurations

## Usage Instructions

### Starting Optimization
1. Select an algorithm from the dropdown menu
2. Adjust algorithm parameters if desired
3. Click "Start Optimization" to begin the process
4. Watch as teams are formed and metrics update in real-time

### Viewing Results
- Teams will appear on the football pitch visualization
- Player positions are color-coded (GK, DEF, MID, FWD)
- Performance metrics show current and best fitness values
- Algorithm metrics display the current parameter settings

### Handling Errors
If you see an error message like "Not enough GK players: 0 available, 5 required", it means:
- The current player data doesn't have enough players of a specific position
- The optimization cannot proceed until this is resolved
- You would need to update the CSV data with sufficient players

## Technical Implementation

### Architecture
- **Frontend**: React-based UI with real-time visualization
- **Backend**: Flask API serving CSV data and running optimization algorithms
- **Data Source**: Direct CSV file access without database

### Optimization Algorithms
- **Genetic Algorithm**: Population-based approach with crossover and mutation
- **Hill Climbing**: Iterative improvement through local search
- **Simulated Annealing**: Probabilistic technique for approximating global optimum

### Data Flow
1. Frontend requests player and configuration data from backend
2. User selects algorithm and parameters
3. Optimization request is sent to backend
4. Backend runs algorithm and streams results back to frontend
5. Frontend visualizes teams and updates metrics in real-time

## Limitations
- Cannot add or remove players through the interface
- Configuration changes require updating the backend data
- Optimization can only run with valid player distributions

## Future Enhancements
- Database integration for dynamic player management
- Additional optimization algorithms
- More detailed visualization of algorithm operations
- Comparative analysis between different algorithms
