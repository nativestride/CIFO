import React, { useState, useEffect } from 'react';
import '../styles.css';

/**
 * AnimatedPlayerMovement component for visualizing player transfers
 * between teams during algorithm optimization.
 */
const AnimatedPlayerMovement = ({ 
  teams, 
  isRunning, 
  currentIteration, 
  playerMovements 
}) => {
  const [visibleMovements, setVisibleMovements] = useState([]);
  const [animatingPlayers, setAnimatingPlayers] = useState([]);
  
  // Update visible movements based on current iteration
  useEffect(() => {
    if (playerMovements.length > 0 && isRunning) {
      // Get movements that should be visible at current iteration
      const currentMovements = playerMovements.filter(
        movement => movement.iteration === currentIteration
      );
      
      if (currentMovements.length > 0) {
        setVisibleMovements(currentMovements);
        
        // Track which players are currently animating
        const playersMoving = currentMovements.map(m => m.playerId);
        setAnimatingPlayers(playersMoving);
        
        // Clear animation status after a delay
        const timer = setTimeout(() => {
          setAnimatingPlayers([]);
        }, 1500); // Animation duration
        
        return () => clearTimeout(timer);
      }
    }
  }, [currentIteration, playerMovements, isRunning]);

  // Find player by ID across all teams
  const findPlayerById = (playerId) => {
    for (const team of teams) {
      const player = team.players.find(p => p.id === playerId);
      if (player) {
        return { player, teamId: team.id };
      }
    }
    return null;
  };

  // Render player movement animations
  const renderPlayerMovements = () => {
    return visibleMovements.map((movement, index) => {
      const playerInfo = findPlayerById(movement.playerId);
      
      if (!playerInfo) return null;
      
      const { player } = playerInfo;
      
      // Calculate source and destination team positions
      // This would need actual DOM positions in a real implementation
      const sourceTeam = document.querySelector(`.team-card:nth-child(${movement.fromTeam + 1})`);
      const destTeam = document.querySelector(`.team-card:nth-child(${movement.toTeam + 1})`);
      
      if (!sourceTeam || !destTeam) return null;
      
      const sourceRect = sourceTeam.getBoundingClientRect();
      const destRect = destTeam.getBoundingClientRect();
      
      const startX = sourceRect.left + sourceRect.width / 2;
      const startY = sourceRect.top + sourceRect.height / 2;
      const endX = destRect.left + destRect.width / 2;
      const endY = destRect.top + destRect.height / 2;
      
      // Animation style
      const animationStyle = {
        position: 'absolute',
        left: 0,
        top: 0,
        width: '100%',
        height: '100%',
        zIndex: 1000 + index,
        pointerEvents: 'none'
      };
      
      // Player card style for animation
      const playerCardStyle = {
        position: 'absolute',
        left: `${startX}px`,
        top: `${startY}px`,
        transform: 'translate(-50%, -50%)',
        transition: 'all 1.5s cubic-bezier(0.25, 0.1, 0.25, 1.0)',
        animation: 'pulse 1.5s infinite',
        zIndex: 1000 + index,
        boxShadow: '0 0 20px var(--secondary-color)',
        background: 'white',
        borderRadius: '8px',
        padding: '8px',
        display: 'flex',
        alignItems: 'center',
        maxWidth: '200px'
      };
      
      // Trigger animation after a small delay
      setTimeout(() => {
        const playerElement = document.getElementById(`moving-player-${movement.playerId}`);
        if (playerElement) {
          playerElement.style.left = `${endX}px`;
          playerElement.style.top = `${endY}px`;
        }
      }, 50);
      
      return (
        <div key={`movement-${index}`} style={animationStyle}>
          <div 
            id={`moving-player-${movement.playerId}`}
            style={playerCardStyle}
            className="player-card moving"
          >
            <div className={`player-position ${player.position}`}>
              {player.position}
            </div>
            <div className="player-info">
              <div className="player-name">{player.name}</div>
              <div className="player-stats">
                <div className="player-skill">Skill: {player.skill}</div>
              </div>
            </div>
          </div>
        </div>
      );
    });
  };

  return (
    <div className="player-movement-animations">
      {isRunning && renderPlayerMovements()}
    </div>
  );
};

export default AnimatedPlayerMovement;
