import React, { useState, useEffect } from 'react';
import TeamVisualization from './components/TeamVisualization';
import AlgorithmExplanation from './components/AlgorithmExplanation';
import RealTimeMetrics from './components/RealTimeMetrics';
import ParameterControls from './components/ParameterControls';
import './styles.css';

function App() {
  // State for algorithm selection and parameters
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('GeneticAlgorithm');
  const [algorithmParams, setAlgorithmParams] = useState({});
  const [algorithms, setAlgorithms] = useState({});
  
  // State for optimization status
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationStatus, setOptimizationStatus] = useState({
    iteration: 0,
    totalIterations: 0,
    currentFitness: null,
    bestFitness: null
  });
  
  // State for teams and players
  const [teams, setTeams] = useState([]);
  const [players, setPlayers] = useState([]);
  const [configuration, setConfiguration] = useState(null);
  const [configValid, setConfigValid] = useState(true);
  const [configMessage, setConfigMessage] = useState('');
  
  // State for active tab
  const [activeTab, setActiveTab] = useState('visualization');
  
  // State for player movements
  const [playerMovements, setPlayerMovements] = useState([]);
  
  // Event source for real-time updates
  const [eventSource, setEventSource] = useState(null);
  
  // Fetch algorithms on component mount
  useEffect(() => {
    fetchAlgorithms();
    fetchPlayers();
    fetchConfiguration();
    validateConfiguration();
  }, []);
  
  // Fetch algorithms from API
  const fetchAlgorithms = async () => {
    try {
      const response = await fetch('/api/algorithms');
      const data = await response.json();
      
      if (data.success) {
        setAlgorithms(data.algorithms);
        
        // Set default parameters for selected algorithm
        if (data.algorithms[selectedAlgorithm]) {
          const defaultParams = {};
          Object.entries(data.algorithms[selectedAlgorithm].parameters).forEach(([key, param]) => {
            defaultParams[key] = param.default;
          });
          setAlgorithmParams(defaultParams);
        }
      }
    } catch (error) {
      console.error('Error fetching algorithms:', error);
    }
  };
  
  // Fetch players from API
  const fetchPlayers = async () => {
    try {
      const response = await fetch('/api/players');
      const data = await response.json();
      
      if (data.success) {
        setPlayers(data.players);
      }
    } catch (error) {
      console.error('Error fetching players:', error);
    }
  };
  
  // Fetch configuration from API
  const fetchConfiguration = async () => {
    try {
      const response = await fetch('/api/configuration');
      const data = await response.json();
      
      if (data.success) {
        setConfiguration(data.configuration);
      }
    } catch (error) {
      console.error('Error fetching configuration:', error);
    }
  };
  
  // Validate configuration
  const validateConfiguration = async () => {
    try {
      const response = await fetch('/api/configuration/validate');
      const data = await response.json();
      
      if (data.success) {
        setConfigValid(data.is_valid);
        setConfigMessage(data.message);
      }
    } catch (error) {
      console.error('Error validating configuration:', error);
      setConfigValid(false);
      setConfigMessage('Error validating configuration');
    }
  };
  
  // Start optimization
  const startOptimization = async () => {
    // Validate configuration first
    await validateConfiguration();
    
    if (!configValid) {
      alert(`Cannot start optimization: ${configMessage}`);
      return;
    }
    
    try {
      // Changed from /api/optimize/start to /optimize/start to match the new endpoint
      const response = await fetch('/optimize/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          algorithm: selectedAlgorithm,
          parameters: algorithmParams
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setIsOptimizing(true);
        setTeams([]);
        setPlayerMovements([]);
        
        // Connect to event stream
        connectToEventStream();
        
        // Start polling for status
        pollOptimizationStatus();
      } else {
        alert(`Failed to start optimization: ${data.message}`);
      }
    } catch (error) {
      console.error('Error starting optimization:', error);
      alert('Error starting optimization');
    }
  };
  
  // Connect to event stream for real-time updates
  const connectToEventStream = () => {
    // Close existing event source if any
    if (eventSource) {
      eventSource.close();
    }
    
    // Create new event source - changed from /api/optimize/events to /optimize/events
    const newEventSource = new EventSource('/optimize/events');
    
    // Set up event handlers
    newEventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'iteration') {
        // Update optimization status
        setOptimizationStatus(prevStatus => ({
          ...prevStatus,
          iteration: data.iteration,
          currentFitness: data.current_fitness,
          bestFitness: data.best_fitness
        }));
      } else if (data.type === 'player_movement') {
        // Add player movement
        setPlayerMovements(prevMovements => [...prevMovements, data.movement]);
      } else if (data.type === 'completed') {
        // Optimization completed
        setIsOptimizing(false);
        newEventSource.close();
      } else if (data.type === 'error') {
        // Error occurred
        alert(`Optimization error: ${data.message}`);
        setIsOptimizing(false);
        newEventSource.close();
      }
    };
    
    newEventSource.onerror = () => {
      console.error('Event source error');
      newEventSource.close();
    };
    
    setEventSource(newEventSource);
  };
  
  // Poll for optimization status
  const pollOptimizationStatus = async () => {
    if (!isOptimizing) return;
    
    try {
      // Changed from /api/optimize/status to /optimize/status
      const response = await fetch('/optimize/status');
      const data = await response.json();
      
      if (data.success) {
        const status = data.status;
        
        setOptimizationStatus({
          iteration: status.iteration,
          totalIterations: status.total_iterations,
          currentFitness: status.current_fitness,
          bestFitness: status.best_fitness
        });
        
        if (status.solution) {
          setTeams(status.solution.teams);
        }
        
        if (status.running) {
          // Continue polling
          setTimeout(pollOptimizationStatus, 1000);
        } else {
          setIsOptimizing(false);
          
          // Close event source
          if (eventSource) {
            eventSource.close();
            setEventSource(null);
          }
        }
      }
    } catch (error) {
      console.error('Error polling optimization status:', error);
    }
  };
  
  // Stop optimization
  const stopOptimization = async () => {
    try {
      // Changed from /api/optimize/stop to /optimize/stop
      const response = await fetch('/optimize/stop', {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (data.success) {
        setIsOptimizing(false);
        
        // Close event source
        if (eventSource) {
          eventSource.close();
          setEventSource(null);
        }
      } else {
        alert(`Failed to stop optimization: ${data.message}`);
      }
    } catch (error) {
      console.error('Error stopping optimization:', error);
      alert('Error stopping optimization');
    }
  };
  
  // Reset optimization
  const resetOptimization = async () => {
    try {
      // Changed from /api/optimize/reset to /optimize/reset
      const response = await fetch('/optimize/reset', {
        method: 'POST'
      });
      
      const data = await response.json();
      
      if (data.success) {
        setTeams([]);
        setPlayerMovements([]);
        setOptimizationStatus({
          iteration: 0,
          totalIterations: 0,
          currentFitness: null,
          bestFitness: null
        });
      } else {
        alert(`Failed to reset optimization: ${data.message}`);
      }
    } catch (error) {
      console.error('Error resetting optimization:', error);
      alert('Error resetting optimization');
    }
  };
  
  // Handle algorithm change
  const handleAlgorithmChange = (e) => {
    const algorithm = e.target.value;
    setSelectedAlgorithm(algorithm);
    
    // Set default parameters for selected algorithm
    if (algorithms[algorithm]) {
      const defaultParams = {};
      Object.entries(algorithms[algorithm].parameters).forEach(([key, param]) => {
        defaultParams[key] = param.default;
      });
      setAlgorithmParams(defaultParams);
    }
  };
  
  // Handle parameter change
  const handleParamChange = (name, value) => {
    setAlgorithmParams(prevParams => ({
      ...prevParams,
      [name]: value
    }));
  };
  
  return (
    <div className="app">
      <header className="app-header">
        <h1>Fantasy League Optimizer</h1>
        <div className="algorithm-selector">
          <select value={selectedAlgorithm} onChange={handleAlgorithmChange} disabled={isOptimizing}>
            {Object.entries(algorithms).map(([key, algorithm]) => (
              <option key={key} value={key}>{algorithm.name}</option>
            ))}
          </select>
          <button 
            className="start-button" 
            onClick={isOptimizing ? stopOptimization : startOptimization}
            disabled={!configValid}
          >
            {isOptimizing ? 'Stop Optimization' : 'Start Optimization'}
          </button>
          <button 
            className="reset-button" 
            onClick={resetOptimization}
            disabled={isOptimizing}
          >
            Reset Teams
          </button>
        </div>
      </header>
      
      {!configValid && (
        <div className="error-message">
          {configMessage}
        </div>
      )}
      
      <div className="tabs">
        <button 
          className={activeTab === 'visualization' ? 'active' : ''} 
          onClick={() => setActiveTab('visualization')}
        >
          Team Visualization
        </button>
        <button 
          className={activeTab === 'explanation' ? 'active' : ''} 
          onClick={() => setActiveTab('explanation')}
        >
          Algorithm Explanation
        </button>
      </div>
      
      <div className="content">
        {activeTab === 'visualization' && (
          <div className="visualization-container">
            <TeamVisualization 
              teams={teams} 
              players={players} 
              playerMovements={playerMovements}
              configuration={configuration}
            />
            <div className="sidebar">
              <RealTimeMetrics 
                status={optimizationStatus} 
                isOptimizing={isOptimizing}
                algorithm={selectedAlgorithm}
                algorithms={algorithms}
                params={algorithmParams}
              />
              <ParameterControls 
                algorithm={selectedAlgorithm}
                algorithms={algorithms}
                params={algorithmParams}
                onParamChange={handleParamChange}
                disabled={isOptimizing}
              />
            </div>
          </div>
        )}
        
        {activeTab === 'explanation' && (
          <AlgorithmExplanation algorithm={selectedAlgorithm} />
        )}
      </div>
    </div>
  );
}

export default App;
