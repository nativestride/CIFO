/**
 * IslandGA visualizer component for the educational website.
 *
 * This component provides an interactive visualization of how the Island Genetic Algorithm
 * works on the Fantasy League problem.
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, Button, Form, Row, Col, Badge, ProgressBar, Table, Tabs, Tab } from 'react-bootstrap';

const IslandGAVisualizer = () => {
  // State for visualization
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [currentStep, setCurrentStep] = useState(0);
  const [showExplanation, setShowExplanation] = useState(true);
  
  // State for algorithm parameters
  const [numIslands, setNumIslands] = useState(4);
  const [islandPopulationSize, setIslandPopulationSize] = useState(20);
  const [maxGenerations, setMaxGenerations] = useState(100);
  const [migrationFrequency, setMigrationFrequency] = useState(10);
  const [migrationRate, setMigrationRate] = useState(0.2);
  const [migrationTopology, setMigrationTopology] = useState('ring');
  const [crossoverRate, setCrossoverRate] = useState(0.8);
  const [mutationRate, setMutationRate] = useState(0.1);
  
  // State for algorithm execution
  const [steps, setSteps] = useState([]);
  const [islands, setIslands] = useState([]);
  const [globalBestSolution, setGlobalBestSolution] = useState(null);
  const [diversityHistory, setDiversityHistory] = useState([]);
  const [interIslandDiversity, setInterIslandDiversity] = useState([]);
  
  // Animation reference
  const animationRef = useRef(null);
  
  // Initialize with dummy data
  useEffect(() => {
    generateDummyData();
  }, []);
  
  // Generate dummy data for visualization
  const generateDummyData = () => {
    // Create initial islands
    const initialIslands = Array.from({ length: numIslands }, (_, islandIndex) => {
      // Create population for each island
      const population = Array.from({ length: islandPopulationSize }, (_, individualIndex) => {
        return {
          id: `island-${islandIndex}-individual-${individualIndex}`,
          fitness: 15 + Math.random() * 10,
          cost: 90 + Math.random() * 20,
          stdDev: 2 + Math.random() * 2,
          genes: Array.from({ length: 11 }, () => Math.floor(Math.random() * 100)),
          origin: islandIndex
        };
      });
      
      // Sort by fitness (lower is better)
      population.sort((a, b) => a.fitness - b.fitness);
      
      return {
        id: `island-${islandIndex}`,
        population: population,
        bestFitness: population[0].fitness,
        bestSolution: population[0],
        averageFitness: population.reduce((sum, ind) => sum + ind.fitness, 0) / population.length,
        diversity: 0.2 + Math.random() * 0.6
      };
    });
    
    setIslands(initialIslands);
    
    // Set global best solution
    const allSolutions = initialIslands.flatMap(island => island.population);
    allSolutions.sort((a, b) => a.fitness - b.fitness);
    setGlobalBestSolution(allSolutions[0]);
    
    // Generate steps
    const generatedSteps = [];
    let currentGen = 0;
    let currentIslands = JSON.parse(JSON.stringify(initialIslands));
    let globalBestFitness = allSolutions[0].fitness;
    let noImprovementCount = 0;
    
    // Track diversity
    let diversityValues = [];
    let interIslandDiversityValues = [];
    
    while (currentGen < maxGenerations && noImprovementCount < 20) {
      // Evolve each island independently
      const evolvedIslands = currentIslands.map(island => {
        return evolveIsland(island, currentGen);
      });
      
      // Check if migration should occur
      const isMigrationGen = currentGen > 0 && currentGen % migrationFrequency === 0;
      let migrations = [];
      
      if (isMigrationGen) {
        // Perform migration
        migrations = performMigration(evolvedIslands, migrationTopology, migrationRate);
        
        // Update islands after migration
        for (const migration of migrations) {
          const { fromIsland, toIsland, migrant } = migration;
          
          // Remove from source island (already done in performMigration)
          
          // Add to destination island
          evolvedIslands[toIsland].population.push(migrant);
          
          // Sort destination island population
          evolvedIslands[toIsland].population.sort((a, b) => a.fitness - b.fitness);
          
          // Update destination island best if needed
          if (migrant.fitness < evolvedIslands[toIsland].bestFitness) {
            evolvedIslands[toIsland].bestFitness = migrant.fitness;
            evolvedIslands[toIsland].bestSolution = migrant;
          }
          
          // Update destination island average
          evolvedIslands[toIsland].averageFitness = 
            evolvedIslands[toIsland].population.reduce((sum, ind) => sum + ind.fitness, 0) / 
            evolvedIslands[toIsland].population.length;
        }
      }
      
      // Find global best solution
      const allEvolvedSolutions = evolvedIslands.flatMap(island => island.population);
      allEvolvedSolutions.sort((a, b) => a.fitness - b.fitness);
      const newGlobalBestFitness = allEvolvedSolutions[0].fitness;
      
      // Check for improvement
      const improvement = newGlobalBestFitness < globalBestFitness;
      
      if (improvement) {
        globalBestFitness = newGlobalBestFitness;
        noImprovementCount = 0;
      } else {
        noImprovementCount++;
      }
      
      // Calculate diversity metrics
      const avgDiversity = evolvedIslands.reduce((sum, island) => sum + island.diversity, 0) / evolvedIslands.length;
      diversityValues.push(avgDiversity);
      
      // Calculate inter-island diversity
      const interDiversity = calculateInterIslandDiversity(evolvedIslands);
      interIslandDiversityValues.push(interDiversity);
      
      // Create step
      const step = {
        generation: currentGen,
        islands: JSON.parse(JSON.stringify(evolvedIslands)),
        globalBestFitness: newGlobalBestFitness,
        globalBestSolution: allEvolvedSolutions[0],
        improvement: improvement,
        noImprovementCount: noImprovementCount,
        isMigrationGen: isMigrationGen,
        migrations: migrations,
        avgDiversity: avgDiversity,
        interIslandDiversity: interDiversity
      };
      
      generatedSteps.push(step);
      
      // Update for next generation
      currentIslands = JSON.parse(JSON.stringify(evolvedIslands));
      currentGen++;
    }
    
    setSteps(generatedSteps);
    setDiversityHistory(diversityValues);
    setInterIslandDiversity(interIslandDiversityValues);
  };
  
  // Evolve a single island for one generation
  const evolveIsland = (island, generation) => {
    const evolvedIsland = { ...island };
    const population = [...evolvedIsland.population];
    
    // Select parents (tournament selection)
    const parents = [];
    const numParents = Math.floor(population.length * 0.5);
    
    for (let i = 0; i < numParents; i++) {
      const tournamentSize = 3;
      const tournament = [];
      
      for (let j = 0; j < tournamentSize; j++) {
        const randomIndex = Math.floor(Math.random() * population.length);
        tournament.push(population[randomIndex]);
      }
      
      tournament.sort((a, b) => a.fitness - b.fitness);
      parents.push(tournament[0]);
    }
    
    // Create offspring through crossover
    const offspring = [];
    
    for (let i = 0; i < parents.length - 1; i += 2) {
      const parent1 = parents[i];
      const parent2 = parents[i + 1] || parents[0];
      
      if (Math.random() < crossoverRate) {
        // One-point crossover
        const crossoverPoint = Math.floor(Math.random() * parent1.genes.length);
        
        const child1Genes = [
          ...parent1.genes.slice(0, crossoverPoint),
          ...parent2.genes.slice(crossoverPoint)
        ];
        
        const child2Genes = [
          ...parent2.genes.slice(0, crossoverPoint),
          ...parent1.genes.slice(crossoverPoint)
        ];
        
        offspring.push({
          id: `island-${island.id}-offspring-${i}-1`,
          genes: child1Genes,
          origin: parent1.origin
        });
        
        offspring.push({
          id: `island-${island.id}-offspring-${i}-2`,
          genes: child2Genes,
          origin: parent2.origin
        });
      } else {
        // No crossover, just copy parents
        offspring.push({
          id: `island-${island.id}-offspring-${i}-1`,
          genes: [...parent1.genes],
          origin: parent1.origin
        });
        
        offspring.push({
          id: `island-${island.id}-offspring-${i}-2`,
          genes: [...parent2.genes],
          origin: parent2.origin
        });
      }
    }
    
    // Apply mutation
    for (const child of offspring) {
      for (let i = 0; i < child.genes.length; i++) {
        if (Math.random() < mutationRate) {
          child.genes[i] = Math.floor(Math.random() * 100);
        }
      }
    }
    
    // Evaluate offspring
    for (const child of offspring) {
      // In a real implementation, this would calculate fitness based on genes
      // For this mockup, we'll use a random value that improves over generations
      const baseFitness = 15 - (10 * (1 - Math.exp(-generation / 30)));
      const fitness = baseFitness + (Math.random() * 2 - 1);
      
      child.fitness = fitness;
      child.cost = 90 + Math.random() * 20;
      child.stdDev = 2 + Math.random() * 2;
    }
    
    // Create new population (elitism + offspring)
    const eliteCount = Math.max(1, Math.floor(population.length * 0.1));
    const elites = population.slice(0, eliteCount);
    
    // Sort offspring
    offspring.sort((a, b) => a.fitness - b.fitness);
    
    // Take best offspring to fill remaining spots
    const newPopulation = [
      ...elites,
      ...offspring.slice(0, population.length - eliteCount)
    ];
    
    // Sort new population
    newPopulation.sort((a, b) => a.fitness - b.fitness);
    
    // Update island
    evolvedIsland.population = newPopulation;
    evolvedIsland.bestFitness = newPopulation[0].fitness;
    evolvedIsland.bestSolution = newPopulation[0];
    evolvedIsland.averageFitness = newPopulation.reduce((sum, ind) => sum + ind.fitness, 0) / newPopulation.length;
    
    // Update diversity (in a real implementation, this would be calculated based on population)
    evolvedIsland.diversity = Math.max(0.1, island.diversity * (0.95 + Math.random() * 0.1));
    
    return evolvedIsland;
  };
  
  // Perform migration between islands
  const performMigration = (islands, topology, rate) => {
    const migrations = [];
    const numMigrants = Math.max(1, Math.floor(islands[0].population.length * rate));
    
    if (topology === 'ring') {
      // Ring topology: island i sends migrants to island i+1
      for (let i = 0; i < islands.length; i++) {
        const fromIsland = i;
        const toIsland = (i + 1) % islands.length;
        
        // Select best individuals as migrants
        const migrants = islands[fromIsland].population.slice(0, numMigrants);
        
        // Record migrations
        for (const migrant of migrants) {
          migrations.push({
            fromIsland: fromIsland,
            toIsland: toIsland,
            migrant: { ...migrant, origin: fromIsland }
          });
        }
        
        // Remove migrants from source island
        islands[fromIsland].population = islands[fromIsland].population.slice(numMigrants);
      }
    } else if (topology === 'random_pair') {
      // Random pair topology: randomly select pairs of islands to exchange migrants
      const islandIndices = Array.from({ length: islands.length }, (_, i) => i);
      
      // Shuffle island indices
      for (let i = islandIndices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [islandIndices[i], islandIndices[j]] = [islandIndices[j], islandIndices[i]];
      }
      
      // Pair islands and exchange migrants
      for (let i = 0; i < islandIndices.length - 1; i += 2) {
        const fromIsland = islandIndices[i];
        const toIsland = islandIndices[i + 1];
        
        // Select best individuals as migrants from first island
        const migrants1 = islands[fromIsland].population.slice(0, numMigrants);
        
        // Record migrations
        for (const migrant of migrants1) {
          migrations.push({
            fromIsland: fromIsland,
            toIsland: toIsland,
            migrant: { ...migrant, origin: fromIsland }
          });
        }
        
        // Remove migrants from source island
        islands[fromIsland].population = islands[fromIsland].population.slice(numMigrants);
        
        // Select best individuals as migrants from second island
        const migrants2 = islands[toIsland].population.slice(0, numMigrants);
        
        // Record migrations
        for (const migrant of migrants2) {
          migrations.push({
            fromIsland: toIsland,
            toIsland: fromIsland,
            migrant: { ...migrant, origin: toIsland }
          });
        }
        
        // Remove migrants from source island
        islands[toIsland].population = islands[toIsland].population.slice(numMigrants);
      }
    } else if (topology === 'broadcast_best') {
      // Broadcast best topology: best island sends migrants to all other islands
      // Find best island
      let bestIslandIndex = 0;
      let bestFitness = islands[0].bestFitness;
      
      for (let i = 1; i < islands.length; i++) {
        if (islands[i].bestFitness < bestFitness) {
          bestFitness = islands[i].bestFitness;
          bestIslandIndex = i;
        }
      }
      
      // Select best individuals as migrants
      const migrants = islands[bestIslandIndex].population.slice(0, numMigrants);
      
      // Send to all other islands
      for (let i = 0; i < islands.length; i++) {
        if (i !== bestIslandIndex) {
          // Record migrations
          for (const migrant of migrants) {
            migrations.push({
              fromIsland: bestIslandIndex,
              toIsland: i,
              migrant: { ...migrant, origin: bestIslandIndex }
            });
          }
        }
      }
      
      // Remove migrants from source island
      islands[bestIslandIndex].population = islands[bestIslandIndex].population.slice(numMigrants);
    }
    
    return migrations;
  };
  
  // Calculate inter-island diversity
  const calculateInterIslandDiversity = (islands) => {
    // In a real implementation, this would calculate genetic diversity between islands
    // For this mockup, we'll use a random value
    return 0.3 + Math.random() * 0.5;
  };
  
  // Start animation
  const startAnimation = () => {
    setIsRunning(true);
    setIsPaused(false);
    
    if (currentStep >= steps.length) {
      setCurrentStep(0);
    }
    
    animateStep();
  };
  
  // Animate a single step
  const animateStep = () => {
    if (currentStep < steps.length) {
      // Update islands based on step
      const step = steps[currentStep];
      setIslands(step.islands);
      
      // Update global best solution if improved
      if (step.improvement) {
        setGlobalBestSolution(step.globalBestSolution);
      }
      
      // Schedule next step
      const delay = 2000 / speed;
      animationRef.current = setTimeout(() => {
        setCurrentStep(prevStep => prevStep + 1);
        
        if (currentStep + 1 < steps.length && !isPaused) {
          animateStep();
        } else {
          setIsRunning(false);
        }
      }, delay);
    } else {
      setIsRunning(false);
    }
  };
  
  // Pause animation
  const pauseAnimation = () => {
    setIsPaused(true);
    setIsRunning(false);
    
    if (animationRef.current) {
      clearTimeout(animationRef.current);
    }
  };
  
  // Step forward
  const stepForward = () => {
    if (currentStep < steps.length) {
      // Update islands based on step
      const step = steps[currentStep];
      setIslands(step.islands);
      
      // Update global best solution if improved
      if (step.improvement) {
        setGlobalBestSolution(step.globalBestSolution);
      }
      
      setCurrentStep(prevStep => prevStep + 1);
    }
  };
  
  // Reset animation
  const resetAnimation = () => {
    setIsRunning(false);
    setIsPaused(false);
    setCurrentStep(0);
    
    if (animationRef.current) {
      clearTimeout(animationRef.current);
    }
    
    // Reset islands and global best solution
    if (steps.length > 0) {
      setIslands(steps[0].islands);
      setGlobalBestSolution(steps[0].globalBestSolution);
    }
  };
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, []);
  
  // Regenerate data when parameters change
  useEffect(() => {
    resetAnimation();
    generateDummyData();
  }, [numIslands, islandPopulationSize, maxGenerations, migrationFrequency, migrationRate, migrationTopology, crossoverRate, mutationRate]);
  
  return (
    <div className="island-ga-visualizer">
      <Card className="mb-4">
        <Card.Header>
          <h3>Island Genetic Algorithm Visualization</h3>
        </Card.Header>
        <Card.Body>
          <Row className="mb-4">
            <Col md={6}>
              <h4>Algorithm Parameters</h4>
              <Form>
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Number of Islands:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={numIslands} 
                      onChange={(e) => setNumIslands(parseInt(e.target.value))}
                      min={2}
                      max={8}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Island Population Size:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={islandPopulationSize} 
                      onChange={(e) => setIslandPopulationSize(parseInt(e.target.value))}
                      min={10}
                      max={100}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Max Generations:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={maxGenerations} 
                      onChange={(e) => setMaxGenerations(parseInt(e.target.value))}
                      min={10}
                      max={500}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Migration Frequency:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={migrationFrequency} 
                      onChange={(e) => setMigrationFrequency(parseInt(e.target.value))}
                      min={1}
                      max={50}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Migration Rate:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={migrationRate} 
                      onChange={(e) => setMigrationRate(parseFloat(e.target.value))}
                      min={0.05}
                      max={0.5}
                      step={0.05}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Migration Topology:</Form.Label>
                  <Col sm={6}>
                    <Form.Select 
                      value={migrationTopology} 
                      onChange={(e) => setMigrationTopology(e.target.value)}
                      disabled={isRunning}
                    >
                      <option value="ring">Ring</option>
                      <option value="random_pair">Random Pair</option>
                      <option value="broadcast_best">Broadcast Best</option>
                    </Form.Select>
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Crossover Rate:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={crossoverRate} 
                      onChange={(e) => setCrossoverRate(parseFloat(e.target.value))}
                      min={0.1}
                      max={1.0}
                      step={0.1}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Mutation Rate:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={mutationRate} 
                      onChange={(e) => setMutationRate(parseFloat(e.target.value))}
                      min={0.01}
                      max={0.5}
                      step={0.01}
                      disabled={isRunning}
                    />
                  </Col>
                </Form.Group>
              </Form>
            </Col>
            
            <Col md={6}>
              <h4>Visualization Controls</h4>
              <div className="d-flex mb-3">
                <Button 
                  variant="success" 
                  onClick={startAnimation} 
                  disabled={isRunning || currentStep >= steps.length}
                  className="me-2"
                >
                  {isPaused ? 'Resume' : 'Start'}
                </Button>
                
                <Button 
                  variant="warning" 
                  onClick={pauseAnimation} 
                  disabled={!isRunning}
                  className="me-2"
                >
                  Pause
                </Button>
                
                <Button 
                  variant="info" 
                  onClick={stepForward} 
                  disabled={isRunning || currentStep >= steps.length}
                  className="me-2"
                >
                  Step
                </Button>
                
                <Button 
                  variant="danger" 
                  onClick={resetAnimation}
                  disabled={isRunning && !isPaused}
                >
                  Reset
                </Button>
              </div>
              
              <Form.Group className="mb-3">
                <Form.Label>Animation Speed:</Form.Label>
                <Form.Range 
                  min={0.5}
                  max={5}
                  step={0.5}
                  value={speed}
                  onChange={(e) => setSpeed(parseFloat(e.target.value))}
                />
                <div className="d-flex justify-content-between">
                  <small>Slow</small>
                  <small>Fast</small>
                </div>
              </Form.Group>
              
              <Card>
                <Card.Header>
                  <h5>Migration Topology: {migrationTopology.replace('_', ' ').toUpperCase()}</h5>
                </Card.Header>
                <Card.Body>
                  <div className="topology-visualization text-center p-3 bg-light rounded">
                    {/* In a real implementation, this would be a D3 or SVG visualization */}
                    <div style={{ fontSize: '0.9rem' }}>
                      {migrationTopology === 'ring' && (
                        <div>
                          <p>Island 0 → Island 1 → Island 2 → ... → Island {numIslands-1} → Island 0</p>
                          <div className="d-flex justify-content-center align-items-center">
                            {Array.from({ length: numIslands }, (_, i) => (
                              <div key={i} className="mx-2 p-2 border rounded bg-white">
                                Island {i}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {migrationTopology === 'random_pair' && (
                        <div>
                          <p>Random pairs of islands exchange migrants each migration event</p>
                          <div className="d-flex justify-content-center align-items-center flex-wrap">
                            {Array.from({ length: numIslands }, (_, i) => (
                              <div key={i} className="m-2 p-2 border rounded bg-white">
                                Island {i}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {migrationTopology === 'broadcast_best' && (
                        <div>
                          <p>Best island sends migrants to all other islands</p>
                          <div className="position-relative" style={{ height: '120px' }}>
                            <div className="position-absolute top-0 start-50 translate-middle-x p-2 border rounded bg-success text-white">
                              Best Island
                            </div>
                            <div className="d-flex justify-content-around align-items-end h-100">
                              {Array.from({ length: numIslands - 1 }, (_, i) => (
                                <div key={i} className="p-2 border rounded bg-white">
                                  Island {i}
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          <Row className="mb-4">
            <Col>
              <h4>Algorithm Progress</h4>
              <ProgressBar 
                now={(currentStep / (steps.length || 1)) * 100} 
                label={`${currentStep} / ${steps.length}`}
                variant="primary"
                className="mb-3"
              />
              
              <div className="d-flex justify-content-between mb-3">
                <div>
                  <strong>Current Generation:</strong> {currentStep < steps.length ? steps[currentStep].generation : 'N/A'}
                </div>
                <div>
                  <strong>Migration Event:</strong> {
                    currentStep < steps.length ? 
                    (steps[currentStep].isMigrationGen ? 
                      <Badge bg="success">Yes</Badge> : 
                      <Badge bg="secondary">No</Badge>) : 
                    'N/A'
                  }
                </div>
                <div>
                  <strong>Status:</strong> {
                    currentStep >= steps.length ? 
                    <Badge bg="success">Completed</Badge> : 
                    isRunning ? 
                    <Badge bg="primary">Running</Badge> : 
                    <Badge bg="secondary">Idle</Badge>
                  }
                </div>
              </div>
            </Col>
          </Row>
          
          <Row className="mb-4">
            <Col md={6}>
              <Card>
                <Card.Header>
                  <h5>Global Best Solution</h5>
                </Card.Header>
                <Card.Body>
                  <div className="mb-3">
                    <strong>Fitness:</strong> {globalBestSolution?.fitness.toFixed(4)}
                  </div>
                  <div className="mb-3">
                    <strong>Cost:</strong> {globalBestSolution?.cost?.toFixed(2)}
                  </div>
                  <div className="mb-3">
                    <strong>Standard Deviation:</strong> {globalBestSolution?.stdDev?.toFixed(4)}
                  </div>
                  <div className="mb-3">
                    <strong>Origin Island:</strong> {globalBestSolution?.origin !== undefined ? globalBestSolution.origin : 'N/A'}
                  </div>
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={6}>
              <Card>
                <Card.Header>
                  <h5>Diversity Metrics</h5>
                </Card.Header>
                <Card.Body>
                  <div className="mb-3">
                    <strong>Average Island Diversity:</strong> {
                      currentStep < steps.length ? 
                      steps[currentStep].avgDiversity.toFixed(4) : 
                      'N/A'
                    }
                  </div>
                  <div className="mb-3">
                    <strong>Inter-Island Diversity:</strong> {
                      currentStep < steps.length ? 
                      steps[currentStep].interIslandDiversity.toFixed(4) : 
                      'N/A'
                    }
                  </div>
                  <div className="mb-3">
                    <strong>Migration Impact:</strong> {
                      currentStep > 0 && currentStep < steps.length && steps[currentStep].isMigrationGen ? 
                      <Badge bg="success">Active</Badge> : 
                      <Badge bg="secondary">Inactive</Badge>
                    }
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          {currentStep < steps.length && (
            <Row className="mb-4">
              <Col>
                <Tabs defaultActiveKey="islands" id="island-ga-tabs" className="mb-3">
                  <Tab eventKey="islands" title="Islands Overview">
                    <Table striped bordered hover size="sm">
                      <thead>
                        <tr>
                          <th>Island</th>
                          <th>Best Fitness</th>
                          <th>Avg Fitness</th>
                          <th>Diversity</th>
                          <th>Population Size</th>
                        </tr>
                      </thead>
                      <tbody>
                        {steps[currentStep].islands.map((island, index) => (
                          <tr key={island.id}>
                            <td>Island {index}</td>
                            <td>{island.bestFitness.toFixed(4)}</td>
                            <td>{island.averageFitness.toFixed(4)}</td>
                            <td>{island.diversity.toFixed(4)}</td>
                            <td>{island.population.length}</td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </Tab>
                  
                  <Tab eventKey="migration" title="Migration">
                    {steps[currentStep].isMigrationGen ? (
                      <div>
                        <h5>Migration Events (Generation {steps[currentStep].generation})</h5>
                        <Table striped bordered hover size="sm">
                          <thead>
                            <tr>
                              <th>#</th>
                              <th>From Island</th>
                              <th>To Island</th>
                              <th>Migrant Fitness</th>
                            </tr>
                          </thead>
                          <tbody>
                            {steps[currentStep].migrations.map((migration, index) => (
                              <tr key={`migration-${index}`}>
                                <td>{index + 1}</td>
                                <td>Island {migration.fromIsland}</td>
                                <td>Island {migration.toIsland}</td>
                                <td>{migration.migrant.fitness.toFixed(4)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </Table>
                      </div>
                    ) : (
                      <div className="text-center p-4">
                        <p>No migration in this generation.</p>
                        <p>Next migration at generation {Math.ceil(steps[currentStep].generation / migrationFrequency) * migrationFrequency}</p>
                      </div>
                    )}
                  </Tab>
                  
                  <Tab eventKey="island-details" title="Island Details">
                    <Form.Group className="mb-3">
                      <Form.Label>Select Island:</Form.Label>
                      <Form.Select 
                        defaultValue={0}
                        onChange={(e) => {
                          // This would update the selected island in a real implementation
                        }}
                      >
                        {steps[currentStep].islands.map((island, index) => (
                          <option key={island.id} value={index}>Island {index}</option>
                        ))}
                      </Form.Select>
                    </Form.Group>
                    
                    <h5>Island 0 Population</h5>
                    <Table striped bordered hover size="sm">
                      <thead>
                        <tr>
                          <th>Rank</th>
                          <th>Individual</th>
                          <th>Fitness</th>
                          <th>Origin Island</th>
                        </tr>
                      </thead>
                      <tbody>
                        {steps[currentStep].islands[0].population.slice(0, 10).map((individual, index) => (
                          <tr key={individual.id} className={index === 0 ? 'table-success' : ''}>
                            <td>{index + 1}</td>
                            <td>{individual.id}</td>
                            <td>{individual.fitness.toFixed(4)}</td>
                            <td>{individual.origin}</td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                    {steps[currentStep].islands[0].population.length > 10 && (
                      <div className="text-center">
                        <small>Showing top 10 of {steps[currentStep].islands[0].population.length} individuals</small>
                      </div>
                    )}
                  </Tab>
                </Tabs>
              </Col>
            </Row>
          )}
          
          <Row>
            <Col>
              <Form.Check 
                type="switch"
                id="explanation-switch"
                label="Show Algorithm Explanation"
                checked={showExplanation}
                onChange={(e) => setShowExplanation(e.target.checked)}
                className="mb-3"
              />
              
              {showExplanation && (
                <Card bg="light">
                  <Card.Body>
                    <h5>How Island Genetic Algorithm Works</h5>
                    <p>
                      Island Genetic Algorithm is a parallel version of GA with multiple isolated populations (islands) that occasionally exchange individuals through migration.
                    </p>
                    <ol>
                      <li>Create multiple isolated populations (islands)</li>
                      <li>Evolve each island independently using standard GA</li>
                      <li>Periodically migrate individuals between islands according to a migration topology</li>
                      <li>Continue evolution until a stopping criterion is met</li>
                    </ol>
                    <h5>Migration Topologies</h5>
                    <ul>
                      <li><strong>Ring:</strong> Each island sends migrants to the next island in a circular arrangement</li>
                      <li><strong>Random Pair:</strong> Random pairs of islands exchange migrants</li>
                      <li><strong>Broadcast Best:</strong> The island with the best solution sends migrants to all other islands</li>
                    </ul>
                    <h5>Key Metrics</h5>
                    <ul>
                      <li><strong>Number of Islands:</strong> Number of separate populations</li>
                      <li><strong>Migration Events:</strong> Number of times individuals migrated between islands</li>
                      <li><strong>Migration Impact:</strong> Effect of migration on solution quality</li>
                      <li><strong>Migration Success Rate:</strong> Percentage of migrations that improved target islands</li>
                      <li><strong>Topology Efficiency:</strong> How effectively the migration topology facilitated improvement</li>
                      <li><strong>Inter-Island Diversity:</strong> Genetic diversity between different islands</li>
                    </ul>
                    <h5>Strengths and Weaknesses</h5>
                    <p><strong>Strengths:</strong> Maintains higher population diversity, parallelizable, often finds better solutions than standard GA</p>
                    <p><strong>Weaknesses:</strong> More complex to implement, requires more parameters to tune</p>
                    <h5>Why Migration Topology Matters</h5>
                    <p>
                      Migration topology determines how genetic information flows between islands, affecting:
                    </p>
                    <ul>
                      <li>Exploration vs. exploitation balance</li>
                      <li>Diversity maintenance</li>
                      <li>Convergence speed</li>
                      <li>Solution quality</li>
                    </ul>
                    <h5>Synchronous vs. Asynchronous Migration</h5>
                    <p>
                      <strong>Synchronous migration</strong> occurs at fixed intervals, with all islands exchanging individuals simultaneously. This is simpler to implement but may cause bottlenecks in parallel implementations.
                    </p>
                    <p>
                      <strong>Asynchronous migration</strong> allows islands to exchange individuals independently, without waiting for others. This can be more efficient in parallel environments but introduces more complexity and potential imbalance.
                    </p>
                  </Card.Body>
                </Card>
              )}
            </Col>
          </Row>
        </Card.Body>
      </Card>
    </div>
  );
};

export default IslandGAVisualizer;
