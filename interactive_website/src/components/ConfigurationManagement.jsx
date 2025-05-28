import React, { useState, useEffect } from 'react';
import '../styles.css';

/**
 * ConfigurationManagement component for managing problem configurations
 * in the Fantasy League optimization website.
 */
const ConfigurationManagement = ({ onConfigurationChange }) => {
  // State for configurations
  const [configurations, setConfigurations] = useState([]);
  // State for selected configuration
  const [selectedConfig, setSelectedConfig] = useState(null);
  // State for new configuration form
  const [newConfig, setNewConfig] = useState({
    name: '',
    description: '',
    num_teams: 4,
    budget: 1000,
    gk_count: 1,
    def_count: 4,
    mid_count: 4,
    fwd_count: 2,
    is_default: false
  });
  // State for editing configuration
  const [editingConfig, setEditingConfig] = useState(null);
  // State for loading status
  const [isLoading, setIsLoading] = useState(true);
  // State for error message
  const [error, setError] = useState(null);
  // State for validation message
  const [validationMessage, setValidationMessage] = useState(null);
  // State for validation status
  const [isValid, setIsValid] = useState(true);

  // Fetch configurations from API
  useEffect(() => {
    fetchConfigurations();
  }, []);

  // Fetch configurations from API
  const fetchConfigurations = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/configurations');
      const data = await response.json();
      
      if (data.success) {
        setConfigurations(data.configurations);
        
        // Select default configuration if available
        const defaultConfig = data.configurations.find(config => config.is_default);
        if (defaultConfig) {
          setSelectedConfig(defaultConfig);
          
          // Notify parent component of configuration change
          if (onConfigurationChange) {
            onConfigurationChange(defaultConfig);
          }
        } else if (data.configurations.length > 0) {
          // Otherwise select the first configuration
          setSelectedConfig(data.configurations[0]);
          
          // Notify parent component of configuration change
          if (onConfigurationChange) {
            onConfigurationChange(data.configurations[0]);
          }
        }
      } else {
        setError(data.message || 'Failed to fetch configurations');
      }
    } catch (error) {
      console.error('Error fetching configurations:', error);
      setError('Failed to fetch configurations. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  // Add a new configuration
  const addConfiguration = async () => {
    // Validate configuration data
    if (!newConfig.name.trim()) {
      setError('Configuration name is required');
      return;
    }
    
    if (newConfig.num_teams < 1) {
      setError('Number of teams must be positive');
      return;
    }
    
    if (newConfig.budget < 1) {
      setError('Budget must be positive');
      return;
    }
    
    if (newConfig.gk_count < 0 || newConfig.def_count < 0 || 
        newConfig.mid_count < 0 || newConfig.fwd_count < 0) {
      setError('Player counts cannot be negative');
      return;
    }
    
    // Validate that at least one player of each position is required
    if (newConfig.gk_count === 0 && newConfig.def_count === 0 && 
        newConfig.mid_count === 0 && newConfig.fwd_count === 0) {
      setError('At least one player position must have a non-zero count');
      return;
    }
    
    try {
      const response = await fetch('/api/configurations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newConfig)
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Add new configuration to state
        setConfigurations([...configurations, data.configuration]);
        
        // Select the new configuration
        setSelectedConfig(data.configuration);
        
        // Notify parent component of configuration change
        if (onConfigurationChange) {
          onConfigurationChange(data.configuration);
        }
        
        // Reset new configuration form
        setNewConfig({
          name: '',
          description: '',
          num_teams: 4,
          budget: 1000,
          gk_count: 1,
          def_count: 4,
          mid_count: 4,
          fwd_count: 2,
          is_default: false
        });
        
        setError(null);
      } else {
        setError(data.message || 'Failed to add configuration');
      }
    } catch (error) {
      console.error('Error adding configuration:', error);
      setError('Failed to add configuration. Please try again later.');
    }
  };

  // Update a configuration
  const updateConfiguration = async () => {
    if (!editingConfig) return;
    
    // Validate configuration data
    if (!editingConfig.name.trim()) {
      setError('Configuration name is required');
      return;
    }
    
    if (editingConfig.num_teams < 1) {
      setError('Number of teams must be positive');
      return;
    }
    
    if (editingConfig.budget < 1) {
      setError('Budget must be positive');
      return;
    }
    
    if (editingConfig.gk_count < 0 || editingConfig.def_count < 0 || 
        editingConfig.mid_count < 0 || editingConfig.fwd_count < 0) {
      setError('Player counts cannot be negative');
      return;
    }
    
    // Validate that at least one player of each position is required
    if (editingConfig.gk_count === 0 && editingConfig.def_count === 0 && 
        editingConfig.mid_count === 0 && editingConfig.fwd_count === 0) {
      setError('At least one player position must have a non-zero count');
      return;
    }
    
    try {
      const response = await fetch(`/api/configurations/${editingConfig.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(editingConfig)
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Update configuration in state
        setConfigurations(configurations.map(config => 
          config.id === editingConfig.id ? data.configuration : config
        ));
        
        // Update selected configuration if it was the one being edited
        if (selectedConfig && selectedConfig.id === editingConfig.id) {
          setSelectedConfig(data.configuration);
          
          // Notify parent component of configuration change
          if (onConfigurationChange) {
            onConfigurationChange(data.configuration);
          }
        }
        
        // Reset editing configuration
        setEditingConfig(null);
        
        setError(null);
      } else {
        setError(data.message || 'Failed to update configuration');
      }
    } catch (error) {
      console.error('Error updating configuration:', error);
      setError('Failed to update configuration. Please try again later.');
    }
  };

  // Delete a configuration
  const deleteConfiguration = async (configId) => {
    // Cannot delete the default configuration
    const config = configurations.find(c => c.id === configId);
    if (config && config.is_default) {
      setError('Cannot delete the default configuration');
      return;
    }
    
    if (!window.confirm('Are you sure you want to delete this configuration?')) {
      return;
    }
    
    try {
      const response = await fetch(`/api/configurations/${configId}`, {
        method: 'DELETE'
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Remove configuration from state
        setConfigurations(configurations.filter(config => config.id !== configId));
        
        // If the deleted configuration was selected, select another one
        if (selectedConfig && selectedConfig.id === configId) {
          const defaultConfig = configurations.find(config => config.is_default && config.id !== configId);
          if (defaultConfig) {
            setSelectedConfig(defaultConfig);
            
            // Notify parent component of configuration change
            if (onConfigurationChange) {
              onConfigurationChange(defaultConfig);
            }
          } else if (configurations.length > 1) {
            const newSelected = configurations.find(config => config.id !== configId);
            setSelectedConfig(newSelected);
            
            // Notify parent component of configuration change
            if (onConfigurationChange) {
              onConfigurationChange(newSelected);
            }
          } else {
            setSelectedConfig(null);
            
            // Notify parent component of configuration change
            if (onConfigurationChange) {
              onConfigurationChange(null);
            }
          }
        }
        
        setError(null);
      } else {
        setError(data.message || 'Failed to delete configuration');
      }
    } catch (error) {
      console.error('Error deleting configuration:', error);
      setError('Failed to delete configuration. Please try again later.');
    }
  };

  // Select a configuration
  const selectConfiguration = (configId) => {
    const config = configurations.find(c => c.id === configId);
    if (config) {
      setSelectedConfig(config);
      
      // Notify parent component of configuration change
      if (onConfigurationChange) {
        onConfigurationChange(config);
      }
      
      // Validate the configuration
      validateConfiguration(configId);
    }
  };

  // Validate a configuration
  const validateConfiguration = async (configId) => {
    try {
      const response = await fetch(`/api/configurations/${configId}/validate`);
      const data = await response.json();
      
      if (data.success) {
        setIsValid(data.is_valid);
        setValidationMessage(data.message);
      } else {
        setIsValid(false);
        setValidationMessage(data.message || 'Failed to validate configuration');
      }
    } catch (error) {
      console.error('Error validating configuration:', error);
      setIsValid(false);
      setValidationMessage('Failed to validate configuration. Please try again later.');
    }
  };

  // Handle new configuration form changes
  const handleNewConfigChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    setNewConfig({
      ...newConfig,
      [name]: type === 'checkbox' ? checked : (type === 'number' ? parseInt(value, 10) : value)
    });
  };

  // Handle editing configuration form changes
  const handleEditingConfigChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    setEditingConfig({
      ...editingConfig,
      [name]: type === 'checkbox' ? checked : (type === 'number' ? parseInt(value, 10) : value)
    });
  };

  // Start editing a configuration
  const startEditing = (config) => {
    setEditingConfig({ ...config });
  };

  // Cancel editing
  const cancelEditing = () => {
    setEditingConfig(null);
  };

  // Render configuration form (add or edit)
  const renderConfigurationForm = (config, isEditing = false) => {
    const handleChange = isEditing ? handleEditingConfigChange : handleNewConfigChange;
    const handleSubmit = isEditing ? updateConfiguration : addConfiguration;
    const buttonText = isEditing ? 'Update Configuration' : 'Add Configuration';
    
    return (
      <div className={`config-form ${isEditing ? 'editing-form' : 'new-form'}`}>
        <h3>{isEditing ? 'Edit Configuration' : 'Add New Configuration'}</h3>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-name`}>Name:</label>
          <input
            type="text"
            id={`${isEditing ? 'edit' : 'new'}-name`}
            name="name"
            value={config.name}
            onChange={handleChange}
            placeholder="Configuration Name"
          />
        </div>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-description`}>Description:</label>
          <textarea
            id={`${isEditing ? 'edit' : 'new'}-description`}
            name="description"
            value={config.description || ''}
            onChange={handleChange}
            placeholder="Configuration Description"
            rows="2"
          />
        </div>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-num-teams`}>Number of Teams:</label>
          <input
            type="number"
            id={`${isEditing ? 'edit' : 'new'}-num-teams`}
            name="num_teams"
            value={config.num_teams}
            onChange={handleChange}
            min="1"
          />
        </div>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-budget`}>Budget per Team (€M):</label>
          <input
            type="number"
            id={`${isEditing ? 'edit' : 'new'}-budget`}
            name="budget"
            value={config.budget}
            onChange={handleChange}
            min="1"
          />
        </div>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-gk-count`}>Goalkeepers per Team:</label>
          <input
            type="number"
            id={`${isEditing ? 'edit' : 'new'}-gk-count`}
            name="gk_count"
            value={config.gk_count}
            onChange={handleChange}
            min="0"
          />
        </div>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-def-count`}>Defenders per Team:</label>
          <input
            type="number"
            id={`${isEditing ? 'edit' : 'new'}-def-count`}
            name="def_count"
            value={config.def_count}
            onChange={handleChange}
            min="0"
          />
        </div>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-mid-count`}>Midfielders per Team:</label>
          <input
            type="number"
            id={`${isEditing ? 'edit' : 'new'}-mid-count`}
            name="mid_count"
            value={config.mid_count}
            onChange={handleChange}
            min="0"
          />
        </div>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-fwd-count`}>Forwards per Team:</label>
          <input
            type="number"
            id={`${isEditing ? 'edit' : 'new'}-fwd-count`}
            name="fwd_count"
            value={config.fwd_count}
            onChange={handleChange}
            min="0"
          />
        </div>
        <div className="form-group checkbox">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-is-default`}>
            <input
              type="checkbox"
              id={`${isEditing ? 'edit' : 'new'}-is-default`}
              name="is_default"
              checked={config.is_default}
              onChange={handleChange}
            />
            Set as Default Configuration
          </label>
        </div>
        <div className="form-actions">
          <button className="primary-button" onClick={handleSubmit}>
            {buttonText}
          </button>
          {isEditing && (
            <button className="secondary-button" onClick={cancelEditing}>
              Cancel
            </button>
          )}
        </div>
      </div>
    );
  };

  // Render configuration summary
  const renderConfigurationSummary = (config) => {
    if (!config) return null;
    
    const totalPlayers = config.gk_count + config.def_count + config.mid_count + config.fwd_count;
    
    return (
      <div className="config-summary">
        <h3>Configuration Summary</h3>
        <div className="summary-item">
          <div className="summary-label">Name:</div>
          <div className="summary-value">{config.name}</div>
        </div>
        {config.description && (
          <div className="summary-item">
            <div className="summary-label">Description:</div>
            <div className="summary-value">{config.description}</div>
          </div>
        )}
        <div className="summary-item">
          <div className="summary-label">Number of Teams:</div>
          <div className="summary-value">{config.num_teams}</div>
        </div>
        <div className="summary-item">
          <div className="summary-label">Budget per Team:</div>
          <div className="summary-value">{config.budget} €M</div>
        </div>
        <div className="summary-item">
          <div className="summary-label">Players per Team:</div>
          <div className="summary-value">{totalPlayers}</div>
        </div>
        <div className="summary-item">
          <div className="summary-label">Team Composition:</div>
          <div className="summary-value">
            {config.gk_count} GK, {config.def_count} DEF, {config.mid_count} MID, {config.fwd_count} FWD
          </div>
        </div>
        <div className="summary-item">
          <div className="summary-label">Total Players Required:</div>
          <div className="summary-value">{config.num_teams * totalPlayers}</div>
        </div>
        <div className="summary-item">
          <div className="summary-label">Default:</div>
          <div className="summary-value">{config.is_default ? 'Yes' : 'No'}</div>
        </div>
        
        {validationMessage && (
          <div className={`validation-message ${isValid ? 'valid' : 'invalid'}`}>
            {validationMessage}
          </div>
        )}
        
        <div className="summary-actions">
          <button
            className="edit-button"
            onClick={() => startEditing(config)}
            disabled={editingConfig !== null}
          >
            Edit
          </button>
          {!config.is_default && (
            <button
              className="delete-button"
              onClick={() => deleteConfiguration(config.id)}
              disabled={editingConfig !== null}
            >
              Delete
            </button>
          )}
        </div>
      </div>
    );
  };

  if (isLoading) {
    return <div className="configuration-management loading">Loading configurations...</div>;
  }

  return (
    <div className="configuration-management">
      <h2>Configuration Management</h2>
      
      {error && (
        <div className="error-message">
          {error}
          <button className="close-button" onClick={() => setError(null)}>×</button>
        </div>
      )}
      
      <div className="management-layout">
        <div className="left-panel">
          {renderConfigurationForm(newConfig)}
        </div>
        
        <div className="right-panel">
          <div className="config-list-header">
            <h3>Available Configurations</h3>
          </div>
          
          {editingConfig && (
            <div className="editing-overlay">
              {renderConfigurationForm(editingConfig, true)}
            </div>
          )}
          
          {configurations.length === 0 ? (
            <div className="no-configs">
              No configurations found. Add a configuration to get started.
            </div>
          ) : (
            <div className="config-list">
              <div className="config-selector">
                <label htmlFor="config-select">Select Configuration:</label>
                <select
                  id="config-select"
                  value={selectedConfig ? selectedConfig.id : ''}
                  onChange={(e) => selectConfiguration(parseInt(e.target.value, 10))}
                  disabled={editingConfig !== null}
                >
                  {configurations.map(config => (
                    <option key={config.id} value={config.id}>
                      {config.name} {config.is_default ? '(Default)' : ''}
                    </option>
                  ))}
                </select>
              </div>
              
              {selectedConfig && renderConfigurationSummary(selectedConfig)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConfigurationManagement;
