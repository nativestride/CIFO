import React, { useState, useEffect } from 'react';
import '../styles.css';

/**
 * PlayerManagement component for adding, editing, and removing players
 * in the Fantasy League optimization website.
 */
const PlayerManagement = ({ onPlayersChange }) => {
  // State for players
  const [players, setPlayers] = useState([]);
  // State for loading status
  const [isLoading, setIsLoading] = useState(true);
  // State for error message
  const [error, setError] = useState(null);
  // State for new player form
  const [newPlayer, setNewPlayer] = useState({
    name: '',
    position: 'GK',
    skill: 80,
    cost: 80
  });
  // State for editing player
  const [editingPlayer, setEditingPlayer] = useState(null);
  // State for filter
  const [filter, setFilter] = useState({
    position: 'all',
    searchTerm: ''
  });
  // State for position counts
  const [positionCounts, setPositionCounts] = useState({
    GK: 0,
    DEF: 0,
    MID: 0,
    FWD: 0
  });

  // Fetch players from API
  useEffect(() => {
    fetchPlayers();
  }, []);

  // Update position counts when players change
  useEffect(() => {
    const counts = {
      GK: 0,
      DEF: 0,
      MID: 0,
      FWD: 0
    };
    
    players.forEach(player => {
      if (counts[player.position] !== undefined) {
        counts[player.position]++;
      }
    });
    
    setPositionCounts(counts);
    
    // Notify parent component of player changes
    if (onPlayersChange) {
      onPlayersChange(players);
    }
  }, [players, onPlayersChange]);

  // Fetch players from API
  const fetchPlayers = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/players');
      const data = await response.json();
      
      if (data.success) {
        setPlayers(data.players);
      } else {
        setError(data.message || 'Failed to fetch players');
      }
    } catch (error) {
      console.error('Error fetching players:', error);
      setError('Failed to fetch players. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  // Add a new player
  const addPlayer = async () => {
    // Validate player data
    if (!newPlayer.name.trim()) {
      setError('Player name is required');
      return;
    }
    
    if (newPlayer.skill < 1 || newPlayer.skill > 100) {
      setError('Skill must be between 1 and 100');
      return;
    }
    
    if (newPlayer.cost < 1) {
      setError('Cost must be positive');
      return;
    }
    
    try {
      const response = await fetch('/api/players', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newPlayer)
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Add new player to state
        setPlayers([...players, data.player]);
        
        // Reset new player form
        setNewPlayer({
          name: '',
          position: 'GK',
          skill: 80,
          cost: 80
        });
        
        setError(null);
      } else {
        setError(data.message || 'Failed to add player');
      }
    } catch (error) {
      console.error('Error adding player:', error);
      setError('Failed to add player. Please try again later.');
    }
  };

  // Update a player
  const updatePlayer = async () => {
    if (!editingPlayer) return;
    
    // Validate player data
    if (!editingPlayer.name.trim()) {
      setError('Player name is required');
      return;
    }
    
    if (editingPlayer.skill < 1 || editingPlayer.skill > 100) {
      setError('Skill must be between 1 and 100');
      return;
    }
    
    if (editingPlayer.cost < 1) {
      setError('Cost must be positive');
      return;
    }
    
    try {
      const response = await fetch(`/api/players/${editingPlayer.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(editingPlayer)
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Update player in state
        setPlayers(players.map(player => 
          player.id === editingPlayer.id ? data.player : player
        ));
        
        // Reset editing player
        setEditingPlayer(null);
        
        setError(null);
      } else {
        setError(data.message || 'Failed to update player');
      }
    } catch (error) {
      console.error('Error updating player:', error);
      setError('Failed to update player. Please try again later.');
    }
  };

  // Delete a player
  const deletePlayer = async (playerId) => {
    if (!window.confirm('Are you sure you want to delete this player?')) {
      return;
    }
    
    try {
      const response = await fetch(`/api/players/${playerId}`, {
        method: 'DELETE'
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Remove player from state
        setPlayers(players.filter(player => player.id !== playerId));
        
        setError(null);
      } else {
        setError(data.message || 'Failed to delete player');
      }
    } catch (error) {
      console.error('Error deleting player:', error);
      setError('Failed to delete player. Please try again later.');
    }
  };

  // Handle new player form changes
  const handleNewPlayerChange = (e) => {
    const { name, value } = e.target;
    
    setNewPlayer({
      ...newPlayer,
      [name]: name === 'skill' || name === 'cost' ? parseInt(value, 10) : value
    });
  };

  // Handle editing player form changes
  const handleEditingPlayerChange = (e) => {
    const { name, value } = e.target;
    
    setEditingPlayer({
      ...editingPlayer,
      [name]: name === 'skill' || name === 'cost' ? parseInt(value, 10) : value
    });
  };

  // Start editing a player
  const startEditing = (player) => {
    setEditingPlayer({ ...player });
  };

  // Cancel editing
  const cancelEditing = () => {
    setEditingPlayer(null);
  };

  // Handle filter changes
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    
    setFilter({
      ...filter,
      [name]: value
    });
  };

  // Filter players
  const filteredPlayers = players.filter(player => {
    // Filter by position
    if (filter.position !== 'all' && player.position !== filter.position) {
      return false;
    }
    
    // Filter by search term
    if (filter.searchTerm && !player.name.toLowerCase().includes(filter.searchTerm.toLowerCase())) {
      return false;
    }
    
    return true;
  });

  // Render player form (add or edit)
  const renderPlayerForm = (player, isEditing = false) => {
    const handleChange = isEditing ? handleEditingPlayerChange : handleNewPlayerChange;
    const handleSubmit = isEditing ? updatePlayer : addPlayer;
    const buttonText = isEditing ? 'Update Player' : 'Add Player';
    
    return (
      <div className={`player-form ${isEditing ? 'editing-form' : 'new-form'}`}>
        <h3>{isEditing ? 'Edit Player' : 'Add New Player'}</h3>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-name`}>Name:</label>
          <input
            type="text"
            id={`${isEditing ? 'edit' : 'new'}-name`}
            name="name"
            value={player.name}
            onChange={handleChange}
            placeholder="Player Name"
          />
        </div>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-position`}>Position:</label>
          <select
            id={`${isEditing ? 'edit' : 'new'}-position`}
            name="position"
            value={player.position}
            onChange={handleChange}
          >
            <option value="GK">Goalkeeper (GK)</option>
            <option value="DEF">Defender (DEF)</option>
            <option value="MID">Midfielder (MID)</option>
            <option value="FWD">Forward (FWD)</option>
          </select>
        </div>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-skill`}>Skill (1-100):</label>
          <input
            type="number"
            id={`${isEditing ? 'edit' : 'new'}-skill`}
            name="skill"
            value={player.skill}
            onChange={handleChange}
            min="1"
            max="100"
          />
        </div>
        <div className="form-group">
          <label htmlFor={`${isEditing ? 'edit' : 'new'}-cost`}>Cost (€M):</label>
          <input
            type="number"
            id={`${isEditing ? 'edit' : 'new'}-cost`}
            name="cost"
            value={player.cost}
            onChange={handleChange}
            min="1"
          />
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

  // Render position counts
  const renderPositionCounts = () => {
    return (
      <div className="position-counts">
        <h3>Position Counts</h3>
        <div className="counts-grid">
          <div className="count-item">
            <div className="count-label">Goalkeepers:</div>
            <div className="count-value">{positionCounts.GK}</div>
          </div>
          <div className="count-item">
            <div className="count-label">Defenders:</div>
            <div className="count-value">{positionCounts.DEF}</div>
          </div>
          <div className="count-item">
            <div className="count-label">Midfielders:</div>
            <div className="count-value">{positionCounts.MID}</div>
          </div>
          <div className="count-item">
            <div className="count-label">Forwards:</div>
            <div className="count-value">{positionCounts.FWD}</div>
          </div>
          <div className="count-item">
            <div className="count-label">Total:</div>
            <div className="count-value">{players.length}</div>
          </div>
        </div>
      </div>
    );
  };

  if (isLoading) {
    return <div className="player-management loading">Loading players...</div>;
  }

  return (
    <div className="player-management">
      <h2>Player Management</h2>
      
      {error && (
        <div className="error-message">
          {error}
          <button className="close-button" onClick={() => setError(null)}>×</button>
        </div>
      )}
      
      <div className="management-layout">
        <div className="left-panel">
          {renderPlayerForm(newPlayer)}
          {renderPositionCounts()}
        </div>
        
        <div className="right-panel">
          <div className="player-list-header">
            <h3>Player List</h3>
            <div className="filter-controls">
              <div className="filter-group">
                <label htmlFor="position-filter">Position:</label>
                <select
                  id="position-filter"
                  name="position"
                  value={filter.position}
                  onChange={handleFilterChange}
                >
                  <option value="all">All Positions</option>
                  <option value="GK">Goalkeepers</option>
                  <option value="DEF">Defenders</option>
                  <option value="MID">Midfielders</option>
                  <option value="FWD">Forwards</option>
                </select>
              </div>
              <div className="filter-group">
                <label htmlFor="search-filter">Search:</label>
                <input
                  type="text"
                  id="search-filter"
                  name="searchTerm"
                  value={filter.searchTerm}
                  onChange={handleFilterChange}
                  placeholder="Search by name"
                />
              </div>
            </div>
          </div>
          
          {editingPlayer && (
            <div className="editing-overlay">
              {renderPlayerForm(editingPlayer, true)}
            </div>
          )}
          
          {filteredPlayers.length === 0 ? (
            <div className="no-players">
              No players found. {filter.position !== 'all' || filter.searchTerm ? 'Try adjusting your filters.' : 'Add some players to get started.'}
            </div>
          ) : (
            <div className="player-list">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Position</th>
                    <th>Skill</th>
                    <th>Cost (€M)</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPlayers.map(player => (
                    <tr key={player.id} className={`position-${player.position.toLowerCase()}`}>
                      <td>{player.id}</td>
                      <td>{player.name}</td>
                      <td>{player.position}</td>
                      <td>{player.skill}</td>
                      <td>{player.cost}</td>
                      <td className="actions">
                        <button
                          className="edit-button"
                          onClick={() => startEditing(player)}
                          disabled={editingPlayer !== null}
                        >
                          Edit
                        </button>
                        <button
                          className="delete-button"
                          onClick={() => deletePlayer(player.id)}
                          disabled={editingPlayer !== null}
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PlayerManagement;
