import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';

const TeamVisualization = ({ teams, players, playerMovements, configuration }) => {
  const [hoveredPlayer, setHoveredPlayer] = useState(null);
  const [movingPlayers, setMovingPlayers] = useState([]);

  // Process player movements
  useEffect(() => {
    if (playerMovements && playerMovements.length > 0) {
      // Get the latest movement
      const latestMovement = playerMovements[playerMovements.length - 1];
      
      if (latestMovement) {
        // Set moving players
        setMovingPlayers(latestMovement.map(move => move.player_id));
        
        // Clear moving players after animation
        const timer = setTimeout(() => {
          setMovingPlayers([]);
        }, 2000);
        
        return () => clearTimeout(timer);
      }
    }
  }, [playerMovements]);

  // Generate team colors
  const teamColors = [
    { primary: '#8e24aa', secondary: '#d81b60' }, // Purple/Pink
    { primary: '#1e88e5', secondary: '#00acc1' }, // Blue/Teal
    { primary: '#43a047', secondary: '#7cb342' }, // Green/Light Green
    { primary: '#fb8c00', secondary: '#ffb300' }, // Orange/Amber
    { primary: '#546e7a', secondary: '#78909c' }, // Blue Grey
  ];

  // Position player on pitch
  const positionPlayer = (player, index, totalPlayers) => {
    const position = player.position;
    let x, y;

    switch (position) {
      case 'GK':
        x = 50;
        y = 10;
        break;
      case 'DEF':
        x = 20 + (index * 60) / (totalPlayers - 1);
        y = 30;
        break;
      case 'MID':
        x = 20 + (index * 60) / (totalPlayers - 1);
        y = 60;
        break;
      case 'FWD':
        x = 20 + (index * 60) / (totalPlayers - 1);
        y = 85;
        break;
      default:
        x = 50;
        y = 50;
    }

    return { x, y };
  };

  // Group players by position for each team
  const getPositionGroups = (teamPlayers) => {
    const groups = {
      GK: [],
      DEF: [],
      MID: [],
      FWD: []
    };

    teamPlayers.forEach(player => {
      if (groups[player.position]) {
        groups[player.position].push(player);
      }
    });

    return groups;
  };

  // Get position class
  const getPositionClass = (position) => {
    switch (position) {
      case 'GK': return 'gk';
      case 'DEF': return 'def';
      case 'MID': return 'mid';
      case 'FWD': return 'fwd';
      default: return '';
    }
  };

  // Render empty state
  if (!teams || teams.length === 0) {
    return (
      <div className="team-visualization-empty">
        <p>No teams to display. Start optimization to see teams.</p>
        
        <div className="position-legend">
          <div className="legend-item">
            <div className="legend-color gk"></div>
            <span>GK - Goalkeeper</span>
          </div>
          <div className="legend-item">
            <div className="legend-color def"></div>
            <span>DEF - Defender</span>
          </div>
          <div className="legend-item">
            <div className="legend-color mid"></div>
            <span>MID - Midfielder</span>
          </div>
          <div className="legend-item">
            <div className="legend-color fwd"></div>
            <span>FWD - Forward</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="team-visualization">
      {teams.map((team, teamIndex) => {
        const positionGroups = getPositionGroups(team.players);
        const teamColor = teamColors[teamIndex % teamColors.length];
        
        return (
          <div className="team-card" key={team.id}>
            <div className="team-header" style={{ background: `linear-gradient(to right, ${teamColor.primary}, ${teamColor.secondary})` }}>
              <h3>Team {team.id}</h3>
              <div className="team-stats">
                <span>Skill: {team.total_skill}</span>
                <span>Cost: {team.total_cost}€M</span>
              </div>
            </div>
            <div className="pitch">
              <div className="pitch-markings">
                <div className="goal-area"></div>
                <div className="center-circle"></div>
                <div className="center-line"></div>
              </div>
              
              {Object.entries(positionGroups).map(([position, posPlayers]) => (
                posPlayers.map((player, playerIndex) => {
                  const { x, y } = positionPlayer(player, playerIndex, posPlayers.length);
                  const isMoving = movingPlayers.includes(player.id);
                  
                  return (
                    <div 
                      key={player.id}
                      className={`player ${getPositionClass(position)} ${isMoving ? 'moving' : ''}`}
                      style={{ 
                        left: `${x}%`, 
                        top: `${y}%`,
                      }}
                      onMouseEnter={() => setHoveredPlayer(player)}
                      onMouseLeave={() => setHoveredPlayer(null)}
                    >
                      {player.skill}
                      {hoveredPlayer === player && (
                        <div className="player-tooltip">
                          {player.name} - {player.position}<br />
                          Skill: {player.skill} | Cost: {player.cost}€M
                        </div>
                      )}
                    </div>
                  );
                })
              ))}
            </div>
          </div>
        );
      })}
      
      <div className="position-legend">
        <div className="legend-item">
          <div className="legend-color gk"></div>
          <span>GK - Goalkeeper</span>
        </div>
        <div className="legend-item">
          <div className="legend-color def"></div>
          <span>DEF - Defender</span>
        </div>
        <div className="legend-item">
          <div className="legend-color mid"></div>
          <span>MID - Midfielder</span>
        </div>
        <div className="legend-item">
          <div className="legend-color fwd"></div>
          <span>FWD - Forward</span>
        </div>
      </div>
    </div>
  );
};

TeamVisualization.propTypes = {
  teams: PropTypes.array,
  players: PropTypes.array,
  playerMovements: PropTypes.array,
  configuration: PropTypes.object
};

TeamVisualization.defaultProps = {
  teams: [],
  players: [],
  playerMovements: [],
  configuration: null
};

export default TeamVisualization;
