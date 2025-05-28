import React, { useState, useEffect } from 'react';
import '../styles.css';

/**
 * PlayerRoster component for displaying and filtering all available players
 * in the Fantasy League optimization problem.
 */
const PlayerRoster = ({ players }) => {
  const [filteredPlayers, setFilteredPlayers] = useState([]);
  const [filters, setFilters] = useState({
    position: 'all',
    minSkill: 0,
    maxSkill: 100,
    minCost: 0,
    maxCost: 2000,
    searchTerm: ''
  });
  
  // Apply filters when players or filters change
  useEffect(() => {
    if (!players || players.length === 0) return;
    
    const filtered = players.filter(player => {
      // Position filter
      if (filters.position !== 'all' && player.position !== filters.position) {
        return false;
      }
      
      // Skill range filter
      if (player.skill < filters.minSkill || player.skill > filters.maxSkill) {
        return false;
      }
      
      // Cost range filter
      if (player.cost < filters.minCost || player.cost > filters.maxCost) {
        return false;
      }
      
      // Search term filter
      if (filters.searchTerm && !player.name.toLowerCase().includes(filters.searchTerm.toLowerCase())) {
        return false;
      }
      
      return true;
    });
    
    setFilteredPlayers(filtered);
  }, [players, filters]);
  
  // Handle filter changes
  const handleFilterChange = (filterName, value) => {
    setFilters(prev => ({
      ...prev,
      [filterName]: value
    }));
  };
  
  // Get position-specific styles
  const getPositionStyle = (position) => {
    switch (position) {
      case 'GK':
        return { backgroundColor: '#ffcc00', color: '#000' };
      case 'DEF':
        return { backgroundColor: '#0066cc', color: '#fff' };
      case 'MID':
        return { backgroundColor: '#33cc33', color: '#fff' };
      case 'FWD':
        return { backgroundColor: '#cc3300', color: '#fff' };
      default:
        return { backgroundColor: '#999', color: '#fff' };
    }
  };

  return (
    <div className="player-roster">
      <h3>Complete Player Roster</h3>
      
      <div className="roster-filters">
        <div className="filter-group">
          <label>Position:</label>
          <select 
            value={filters.position} 
            onChange={(e) => handleFilterChange('position', e.target.value)}
          >
            <option value="all">All Positions</option>
            <option value="GK">Goalkeepers (GK)</option>
            <option value="DEF">Defenders (DEF)</option>
            <option value="MID">Midfielders (MID)</option>
            <option value="FWD">Forwards (FWD)</option>
          </select>
        </div>
        
        <div className="filter-group">
          <label>Skill Range:</label>
          <div className="range-inputs">
            <input 
              type="number" 
              min="0" 
              max="100" 
              value={filters.minSkill} 
              onChange={(e) => handleFilterChange('minSkill', parseInt(e.target.value, 10))}
            />
            <span>to</span>
            <input 
              type="number" 
              min="0" 
              max="100" 
              value={filters.maxSkill} 
              onChange={(e) => handleFilterChange('maxSkill', parseInt(e.target.value, 10))}
            />
          </div>
        </div>
        
        <div className="filter-group">
          <label>Cost Range:</label>
          <div className="range-inputs">
            <input 
              type="number" 
              min="0" 
              max="2000" 
              value={filters.minCost} 
              onChange={(e) => handleFilterChange('minCost', parseInt(e.target.value, 10))}
            />
            <span>to</span>
            <input 
              type="number" 
              min="0" 
              max="2000" 
              value={filters.maxCost} 
              onChange={(e) => handleFilterChange('maxCost', parseInt(e.target.value, 10))}
            />
          </div>
        </div>
        
        <div className="filter-group">
          <label>Search:</label>
          <input 
            type="text" 
            placeholder="Search by name..." 
            value={filters.searchTerm} 
            onChange={(e) => handleFilterChange('searchTerm', e.target.value)}
          />
        </div>
      </div>
      
      <div className="roster-stats">
        <span>Showing {filteredPlayers.length} of {players.length} players</span>
      </div>
      
      <div className="roster-table-container">
        <table className="roster-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Position</th>
              <th>Skill</th>
              <th>Cost</th>
            </tr>
          </thead>
          <tbody>
            {filteredPlayers.map(player => (
              <tr key={player.id}>
                <td>{player.name}</td>
                <td>
                  <span 
                    className="position-badge"
                    style={getPositionStyle(player.position)}
                  >
                    {player.position}
                  </span>
                </td>
                <td>{player.skill}</td>
                <td>{player.cost}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PlayerRoster;
