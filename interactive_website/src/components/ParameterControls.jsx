import React from 'react';
import PropTypes from 'prop-types';

const ParameterControls = ({ algorithm, algorithms, params, onParamChange, disabled }) => {
  // Get algorithm parameters
  const getAlgorithmParams = () => {
    if (algorithms && algorithms[algorithm] && algorithms[algorithm].parameters) {
      return algorithms[algorithm].parameters;
    }
    return {};
  };
  
  const algorithmParams = getAlgorithmParams();
  
  // Render parameter input based on type
  const renderParamInput = (name, param) => {
    const value = params[name] !== undefined ? params[name] : param.default;
    
    switch (param.type) {
      case 'number':
        return (
          <input
            type="number"
            className="parameter-input"
            value={value}
            min={param.min}
            max={param.max}
            step={param.step}
            onChange={(e) => onParamChange(name, parseFloat(e.target.value))}
            disabled={disabled}
          />
        );
      
      case 'select':
        return (
          <select
            className="parameter-select"
            value={value}
            onChange={(e) => onParamChange(name, e.target.value)}
            disabled={disabled}
          >
            {param.options.map((option) => (
              <option key={option} value={option}>
                {option.charAt(0).toUpperCase() + option.slice(1)}
              </option>
            ))}
          </select>
        );
      
      case 'boolean':
        return (
          <input
            type="checkbox"
            className="parameter-checkbox"
            checked={value}
            onChange={(e) => onParamChange(name, e.target.checked)}
            disabled={disabled}
          />
        );
      
      default:
        return null;
    }
  };
  
  return (
    <div className="parameters-card">
      <h3>Algorithm Parameters</h3>
      
      {Object.entries(algorithmParams).map(([name, param]) => (
        <div key={name} className="parameter">
          <label className="parameter-label">{param.label}:</label>
          {renderParamInput(name, param)}
        </div>
      ))}
    </div>
  );
};

ParameterControls.propTypes = {
  algorithm: PropTypes.string.isRequired,
  algorithms: PropTypes.object,
  params: PropTypes.object,
  onParamChange: PropTypes.func.isRequired,
  disabled: PropTypes.bool
};

ParameterControls.defaultProps = {
  algorithms: {},
  params: {},
  disabled: false
};

export default ParameterControls;
