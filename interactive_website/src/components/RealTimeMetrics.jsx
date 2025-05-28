import React from 'react';
import PropTypes from 'prop-types';

const RealTimeMetrics = ({ status, isOptimizing, algorithm, algorithms, params }) => {
  const { iteration, totalIterations, currentFitness, bestFitness } = status;
  
  // Calculate progress percentage
  const progress = totalIterations > 0 ? (iteration / totalIterations) * 100 : 0;
  
  // Format fitness values
  const formatFitness = (fitness) => {
    if (fitness === null || fitness === undefined) return 'N/A';
    return fitness.toFixed(2);
  };
  
  // Get algorithm display name
  const getAlgorithmName = () => {
    if (algorithms && algorithms[algorithm]) {
      return algorithms[algorithm].name;
    }
    return algorithm;
  };
  
  return (
    <div className="metrics-card">
      <h3>Performance Metrics</h3>
      
      <div className="metric">
        <span className="metric-label">Iteration:</span>
        <span className="metric-value">{iteration} / {totalIterations || '?'}</span>
      </div>
      
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${progress}%` }}></div>
      </div>
      
      <div className="metric">
        <span className="metric-label">Current Fitness:</span>
        <span className="metric-value">{formatFitness(currentFitness)}</span>
      </div>
      
      <div className="metric">
        <span className="metric-label">Best Fitness:</span>
        <span className="metric-value">{formatFitness(bestFitness)}</span>
      </div>
      
      <h3>Algorithm Metrics</h3>
      
      <div className="metric">
        <span className="metric-label">Algorithm:</span>
        <span className="metric-value">{getAlgorithmName()}</span>
      </div>
      
      {algorithm === 'GeneticAlgorithm' && (
        <>
          <div className="metric">
            <span className="metric-label">Population Size:</span>
            <span className="metric-value">{params.population_size || 'N/A'}</span>
          </div>
          
          <div className="metric">
            <span className="metric-label">Crossover Rate:</span>
            <span className="metric-value">{params.crossover_rate || 'N/A'}</span>
          </div>
          
          <div className="metric">
            <span className="metric-label">Mutation Rate:</span>
            <span className="metric-value">{params.mutation_rate || 'N/A'}</span>
          </div>
          
          <div className="metric">
            <span className="metric-label">Selection Method:</span>
            <span className="metric-value">{params.selection_method || 'N/A'}</span>
          </div>
        </>
      )}
      
      {algorithm === 'IslandGeneticAlgorithm' && (
        <>
          <div className="metric">
            <span className="metric-label">Islands:</span>
            <span className="metric-value">{params.num_islands || 'N/A'}</span>
          </div>
          
          <div className="metric">
            <span className="metric-label">Island Population:</span>
            <span className="metric-value">{params.island_population || 'N/A'}</span>
          </div>
          
          <div className="metric">
            <span className="metric-label">Migration Interval:</span>
            <span className="metric-value">{params.migration_interval || 'N/A'}</span>
          </div>
        </>
      )}
      
      {algorithm === 'SimulatedAnnealing' && (
        <>
          <div className="metric">
            <span className="metric-label">Initial Temperature:</span>
            <span className="metric-value">{params.initial_temperature || 'N/A'}</span>
          </div>
          
          <div className="metric">
            <span className="metric-label">Cooling Rate:</span>
            <span className="metric-value">{params.cooling_rate || 'N/A'}</span>
          </div>
        </>
      )}
      
      {algorithm === 'HillClimbing' && (
        <>
          <div className="metric">
            <span className="metric-label">Random Restarts:</span>
            <span className="metric-value">{params.random_restarts || 'N/A'}</span>
          </div>
          
          <div className="metric">
            <span className="metric-label">Intensive Search:</span>
            <span className="metric-value">{params.intensive_search ? 'Yes' : 'No'}</span>
          </div>
        </>
      )}
      
      <div className="status-message">
        {isOptimizing ? 'Optimization in progress...' : 'Ready to optimize'}
      </div>
    </div>
  );
};

RealTimeMetrics.propTypes = {
  status: PropTypes.shape({
    iteration: PropTypes.number,
    totalIterations: PropTypes.number,
    currentFitness: PropTypes.number,
    bestFitness: PropTypes.number
  }),
  isOptimizing: PropTypes.bool,
  algorithm: PropTypes.string,
  algorithms: PropTypes.object,
  params: PropTypes.object
};

RealTimeMetrics.defaultProps = {
  status: {
    iteration: 0,
    totalIterations: 0,
    currentFitness: null,
    bestFitness: null
  },
  isOptimizing: false,
  algorithm: '',
  algorithms: {},
  params: {}
};

export default RealTimeMetrics;
