/**
 * GeneticAlgorithm visualizer component for the educational website.
 *
 * This component provides an interactive visualization of how the Genetic Algorithm
 * works on the Fantasy League problem.
 */

import React, { useState, useEffect, useRef } from 'react';
import { Card, Button, Form, Row, Col, Badge, ProgressBar, Table, Tabs, Tab } from 'react-bootstrap';

const GeneticAlgorithmVisualizer = () => {
  // State for visualization
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [currentStep, setCurrentStep] = useState(0);
  const [showExplanation, setShowExplanation] = useState(true);
  
  // State for algorithm parameters
  const [populationSize, setPopulationSize] = useState(50);
  const [maxGenerations, setMaxGenerations] = useState(100);
  const [crossoverRate, setCrossoverRate] = useState(0.8);
  const [mutationRate, setMutationRate] = useState(0.1);
  const [selectionMethod, setSelectionMethod] = useState('tournament');
  const [crossoverMethod, setCrossoverMethod] = useState('onePoint');
  
  // State for algorithm execution
  const [steps, setSteps] = useState([]);
  const [currentPopulation, setCurrentPopulation] = useState([]);
  const [bestSolution, setBestSolution] = useState(null);
  const [diversityHistory, setDiversityHistory] = useState([]);
  
  // Animation reference
  const animationRef = useRef(null);
  
  // Initialize with dummy data
  useEffect(() => {
    generateDummyData();
  }, []);
  
  // Generate dummy data for visualization
  const generateDummyData = () => {
    // Create initial population
    const initialPopulation = Array.from({ length: populationSize }, (_, i) => {
      return {
        id: `individual-${i}`,
        fitness: 15 + Math.random() * 10,
        cost: 90 + Math.random() * 20,
        stdDev: 2 + Math.random() * 2,
        genes: Array.from({ length: 11 }, () => Math.floor(Math.random() * 100))
      };
    });
    
    // Sort by fitness (lower is better)
    initialPopulation.sort((a, b) => a.fitness - b.fitness);
    
    setCurrentPopulation(initialPopulation);
    setBestSolution(initialPopulation[0]);
    
    // Generate steps
    const generatedSteps = [];
    let currentGen = 0;
    let currentBestFitness = initialPopulation[0].fitness;
    let noImprovementCount = 0;
    let diversity = calculateDiversity(initialPopulation);
    let diversityValues = [diversity];
    
    while (currentGen < maxGenerations && noImprovementCount < 20) {
      // Select parents
      const parents = selectParents(initialPopulation, selectionMethod);
      
      // Create offspring through crossover
      const offspring = performCrossover(parents, crossoverMethod, crossoverRate);
      
      // Apply mutation
      const mutatedOffspring = performMutation(offspring, mutationRate);
      
      // Evaluate offspring
      const evaluatedOffspring = evaluateOffspring(mutatedOffspring, currentGen);
      
      // Create new population
      const newPopulation = [...initialPopulation];
      
      // Replace worst individuals with offspring
      for (let i = 0; i < evaluatedOffspring.length; i++) {
        const replaceIndex = newPopulation.length - 1 - i;
        if (replaceIndex >= 0) {
          newPopulation[replaceIndex] = evaluatedOffspring[i];
        }
      }
      
      // Sort by fitness
      newPopulation.sort((a, b) => a.fitness - b.fitness);
      
      // Calculate diversity
      diversity = calculateDiversity(newPopulation);
      diversityValues.push(diversity);
      
      // Check for improvement
      const newBestFitness = newPopulation[0].fitness;
      const improvement = newBestFitness < currentBestFitness;
      
      if (improvement) {
        currentBestFitness = newBestFitness;
        noImprovementCount = 0;
      } else {
        noImprovementCount++;
      }
      
      // Create step
      const step = {
        generation: currentGen,
        parents: parents,
        offspring: offspring,
        mutatedOffspring: mutatedOffspring,
        evaluatedOffspring: evaluatedOffspring,
        newPopulation: newPopulation,
        bestFitness: newBestFitness,
        improvement: improvement,
        noImprovementCount: noImprovementCount,
        diversity: diversity
      };
      
      generatedSteps.push(step);
      
      // Update for next generation
      initialPopulation.splice(0, initialPopulation.length, ...newPopulation);
      currentGen++;
    }
    
    setSteps(generatedSteps);
    setDiversityHistory(diversityValues);
  };
  
  // Calculate population diversity
  const calculateDiversity = (population) => {
    // In a real implementation, this would calculate genetic diversity
    // For this mockup, we'll use a random value that decreases over time
    return 0.2 + Math.random() * 0.6;
  };
  
  // Select parents based on selection method
  const selectParents = (population, method) => {
    const parents = [];
    const numParents = Math.floor(population.length * 0.5);
    
    if (method === 'tournament') {
      // Tournament selection
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
    } else if (method === 'roulette') {
      // Roulette wheel selection
      const totalFitness = population.reduce((sum, individual) => sum + (1 / individual.fitness), 0);
      
      for (let i = 0; i < numParents; i++) {
        let value = Math.random() * totalFitness;
        let runningSum = 0;
        
        for (let j = 0; j < population.length; j++) {
          runningSum += (1 / population[j].fitness);
          
          if (runningSum >= value) {
            parents.push(population[j]);
            break;
          }
        }
      }
    } else {
      // Ranking selection
      const sortedPopulation = [...population].sort((a, b) => a.fitness - b.fitness);
      
      for (let i = 0; i < numParents; i++) {
        const rank = Math.floor(Math.random() * Math.random() * sortedPopulation.length);
        parents.push(sortedPopulation[rank]);
      }
    }
    
    return parents;
  };
  
  // Perform crossover based on crossover method
  const performCrossover = (parents, method, rate) => {
    const offspring = [];
    
    for (let i = 0; i < parents.length - 1; i += 2) {
      const parent1 = parents[i];
      const parent2 = parents[i + 1] || parents[0];
      
      if (Math.random() < rate) {
        if (method === 'onePoint') {
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
            id: `offspring-${i}-1`,
            genes: child1Genes,
            crossoverPoint: crossoverPoint,
            parents: [parent1.id, parent2.id]
          });
          
          offspring.push({
            id: `offspring-${i}-2`,
            genes: child2Genes,
            crossoverPoint: crossoverPoint,
            parents: [parent1.id, parent2.id]
          });
        } else if (method === 'twoPoint') {
          // Two-point crossover
          const point1 = Math.floor(Math.random() * parent1.genes.length);
          let point2 = Math.floor(Math.random() * parent1.genes.length);
          
          // Ensure point2 > point1
          if (point2 < point1) {
            [point1, point2] = [point2, point1];
          }
          
          const child1Genes = [
            ...parent1.genes.slice(0, point1),
            ...parent2.genes.slice(point1, point2),
            ...parent1.genes.slice(point2)
          ];
          
          const child2Genes = [
            ...parent2.genes.slice(0, point1),
            ...parent1.genes.slice(point1, point2),
            ...parent2.genes.slice(point2)
          ];
          
          offspring.push({
            id: `offspring-${i}-1`,
            genes: child1Genes,
            crossoverPoints: [point1, point2],
            parents: [parent1.id, parent2.id]
          });
          
          offspring.push({
            id: `offspring-${i}-2`,
            genes: child2Genes,
            crossoverPoints: [point1, point2],
            parents: [parent1.id, parent2.id]
          });
        } else {
          // Uniform crossover
          const child1Genes = [];
          const child2Genes = [];
          
          for (let j = 0; j < parent1.genes.length; j++) {
            if (Math.random() < 0.5) {
              child1Genes.push(parent1.genes[j]);
              child2Genes.push(parent2.genes[j]);
            } else {
              child1Genes.push(parent2.genes[j]);
              child2Genes.push(parent1.genes[j]);
            }
          }
          
          offspring.push({
            id: `offspring-${i}-1`,
            genes: child1Genes,
            uniformMask: Array.from({ length: parent1.genes.length }, () => Math.random() < 0.5),
            parents: [parent1.id, parent2.id]
          });
          
          offspring.push({
            id: `offspring-${i}-2`,
            genes: child2Genes,
            uniformMask: Array.from({ length: parent1.genes.length }, () => Math.random() < 0.5),
            parents: [parent1.id, parent2.id]
          });
        }
      } else {
        // No crossover, just copy parents
        offspring.push({
          id: `offspring-${i}-1`,
          genes: [...parent1.genes],
          noCrossover: true,
          parents: [parent1.id]
        });
        
        offspring.push({
          id: `offspring-${i}-2`,
          genes: [...parent2.genes],
          noCrossover: true,
          parents: [parent2.id]
        });
      }
    }
    
    return offspring;
  };
  
  // Perform mutation
  const performMutation = (offspring, rate) => {
    return offspring.map(child => {
      const mutatedGenes = [...child.genes];
      const mutations = [];
      
      for (let i = 0; i < mutatedGenes.length; i++) {
        if (Math.random() < rate) {
          const originalValue = mutatedGenes[i];
          mutatedGenes[i] = Math.floor(Math.random() * 100);
          mutations.push({ index: i, from: originalValue, to: mutatedGenes[i] });
        }
      }
      
      return {
        ...child,
        genes: mutatedGenes,
        mutations: mutations
      };
    });
  };
  
  // Evaluate offspring
  const evaluateOffspring = (offspring, generation) => {
    return offspring.map(child => {
      // In a real implementation, this would calculate fitness based on genes
      // For this mockup, we'll use a random value that improves over generations
      const baseFitness = 15 - (10 * (1 - Math.exp(-generation / 30)));
      const fitness = baseFitness + (Math.random() * 2 - 1);
      
      return {
        ...child,
        fitness: fitness,
        cost: 90 + Math.random() * 20,
        stdDev: 2 + Math.random() * 2
      };
    });
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
      // Update current population based on step
      const step = steps[currentStep];
      setCurrentPopulation(step.newPopulation);
      
      // Update best solution if improved
      if (step.improvement) {
        setBestSolution(step.newPopulation[0]);
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
      // Update current population based on step
      const step = steps[currentStep];
      setCurrentPopulation(step.newPopulation);
      
      // Update best solution if improved
      if (step.improvement) {
        setBestSolution(step.newPopulation[0]);
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
    
    // Reset population and best solution
    if (steps.length > 0) {
      const initialPopulation = steps[0].newPopulation;
      setCurrentPopulation(initialPopulation);
      setBestSolution(initialPopulation[0]);
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
  }, [populationSize, maxGenerations, crossoverRate, mutationRate, selectionMethod, crossoverMethod]);
  
  return (
    <div className="genetic-algorithm-visualizer">
      <Card className="mb-4">
        <Card.Header>
          <h3>Genetic Algorithm Visualization</h3>
        </Card.Header>
        <Card.Body>
          <Row className="mb-4">
            <Col md={6}>
              <h4>Algorithm Parameters</h4>
              <Form>
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Population Size:</Form.Label>
                  <Col sm={6}>
                    <Form.Control 
                      type="number" 
                      value={populationSize} 
                      onChange={(e) => setPopulationSize(parseInt(e.target.value))}
                      min={10}
                      max={200}
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
                      max={1000}
                      disabled={isRunning}
                    />
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
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Selection Method:</Form.Label>
                  <Col sm={6}>
                    <Form.Select 
                      value={selectionMethod} 
                      onChange={(e) => setSelectionMethod(e.target.value)}
                      disabled={isRunning}
                    >
                      <option value="tournament">Tournament</option>
                      <option value="roulette">Roulette Wheel</option>
                      <option value="ranking">Ranking</option>
                    </Form.Select>
                  </Col>
                </Form.Group>
                
                <Form.Group as={Row} className="mb-2">
                  <Form.Label column sm={6}>Crossover Method:</Form.Label>
                  <Col sm={6}>
                    <Form.Select 
                      value={crossoverMethod} 
                      onChange={(e) => setCrossoverMethod(e.target.value)}
                      disabled={isRunning}
                    >
                      <option value="onePoint">One Point</option>
                      <option value="twoPoint">Two Point</option>
                      <option value="uniform">Uniform</option>
                    </Form.Select>
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
                  <strong>No Improvement Count:</strong> {
                    currentStep < steps.length ? 
                    steps[currentStep].noImprovementCount : 'N/A'
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
                  <h5>Best Solution</h5>
                </Card.Header>
                <Card.Body>
                  <div className="mb-3">
                    <strong>Fitness:</strong> {bestSolution?.fitness.toFixed(4)}
                  </div>
                  <div className="mb-3">
                    <strong>Cost:</strong> {bestSolution?.cost.toFixed(2)}
                  </div>
                  <div className="mb-3">
                    <strong>Standard Deviation:</strong> {bestSolution?.stdDev.toFixed(4)}
                  </div>
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={6}>
              <Card>
                <Card.Header>
                  <h5>Population Statistics</h5>
                </Card.Header>
                <Card.Body>
                  <div className="mb-3">
                    <strong>Population Size:</strong> {currentPopulation.length}
                  </div>
                  <div className="mb-3">
                    <strong>Average Fitness:</strong> {
                      currentPopulation.length > 0 ? 
                      (currentPopulation.reduce((sum, ind) => sum + ind.fitness, 0) / currentPopulation.length).toFixed(4) : 
                      'N/A'
                    }
                  </div>
                  <div className="mb-3">
                    <strong>Diversity:</strong> {
                      currentStep < steps.length ? 
                      steps[currentStep].diversity.toFixed(4) : 
                      'N/A'
                    }
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          {currentStep < steps.length && (
            <Row className="mb-4">
              <Col>
                <Tabs defaultActiveKey="population" id="ga-tabs" className="mb-3">
                  <Tab eventKey="population" title="Population">
                    <Table striped bordered hover size="sm">
                      <thead>
                        <tr>
                          <th>Rank</th>
                          <th>Individual</th>
                          <th>Fitness</th>
                          <th>Cost</th>
                          <th>Std Dev</th>
                        </tr>
                      </thead>
                      <tbody>
                        {currentPopulation.slice(0, 10).map((individual, index) => (
                          <tr key={individual.id} className={index === 0 ? 'table-success' : ''}>
                            <td>{index + 1}</td>
                            <td>{individual.id}</td>
                            <td>{individual.fitness.toFixed(4)}</td>
                            <td>{individual.cost.toFixed(2)}</td>
                            <td>{individual.stdDev.toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                    {currentPopulation.length > 10 && (
                      <div className="text-center">
                        <small>Showing top 10 of {currentPopulation.length} individuals</small>
                      </div>
                    )}
                  </Tab>
                  
                  <Tab eventKey="selection" title="Selection">
                    {currentStep < steps.length && (
                      <div>
                        <h5>Selected Parents</h5>
                        <p>Selection Method: <Badge bg="info">{selectionMethod}</Badge></p>
                        <Table striped bordered hover size="sm">
                          <thead>
                            <tr>
                              <th>#</th>
                              <th>Parent ID</th>
                              <th>Fitness</th>
                            </tr>
                          </thead>
                          <tbody>
                            {steps[currentStep].parents.slice(0, 10).map((parent, index) => (
                              <tr key={`parent-${index}`}>
                                <td>{index + 1}</td>
                                <td>{parent.id}</td>
                                <td>{parent.fitness.toFixed(4)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </Table>
                        {steps[currentStep].parents.length > 10 && (
                          <div className="text-center">
                            <small>Showing 10 of {steps[currentStep].parents.length} parents</small>
                          </div>
                        )}
                      </div>
                    )}
                  </Tab>
                  
                  <Tab eventKey="crossover" title="Crossover">
                    {currentStep < steps.length && (
                      <div>
                        <h5>Crossover Results</h5>
                        <p>Crossover Method: <Badge bg="info">{crossoverMethod}</Badge> | Rate: <Badge bg="info">{crossoverRate}</Badge></p>
                        <Table striped bordered hover size="sm">
                          <thead>
                            <tr>
                              <th>#</th>
                              <th>Offspring ID</th>
                              <th>Parents</th>
                              <th>Crossover Details</th>
                            </tr>
                          </thead>
                          <tbody>
                            {steps[currentStep].offspring.slice(0, 10).map((child, index) => (
                              <tr key={`offspring-${index}`}>
                                <td>{index + 1}</td>
                                <td>{child.id}</td>
                                <td>{child.parents.join(', ')}</td>
                                <td>
                                  {child.noCrossover ? (
                                    <Badge bg="secondary">No Crossover</Badge>
                                  ) : crossoverMethod === 'onePoint' ? (
                                    <span>Point: {child.crossoverPoint}</span>
                                  ) : crossoverMethod === 'twoPoint' ? (
                                    <span>Points: {child.crossoverPoints.join(', ')}</span>
                                  ) : (
                                    <Badge bg="info">Uniform</Badge>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </Table>
                        {steps[currentStep].offspring.length > 10 && (
                          <div className="text-center">
                            <small>Showing 10 of {steps[currentStep].offspring.length} offspring</small>
                          </div>
                        )}
                      </div>
                    )}
                  </Tab>
                  
                  <Tab eventKey="mutation" title="Mutation">
                    {currentStep < steps.length && (
                      <div>
                        <h5>Mutation Results</h5>
                        <p>Mutation Rate: <Badge bg="info">{mutationRate}</Badge></p>
                        <Table striped bordered hover size="sm">
                          <thead>
                            <tr>
                              <th>#</th>
                              <th>Individual ID</th>
                              <th>Mutations</th>
                            </tr>
                          </thead>
                          <tbody>
                            {steps[currentStep].mutatedOffspring.slice(0, 10).map((individual, index) => (
                              <tr key={`mutated-${index}`}>
                                <td>{index + 1}</td>
                                <td>{individual.id}</td>
                                <td>
                                  {individual.mutations.length === 0 ? (
                                    <Badge bg="secondary">No Mutations</Badge>
                                  ) : (
                                    <Badge bg="warning">{individual.mutations.length} mutations</Badge>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </Table>
                        {steps[currentStep].mutatedOffspring.length > 10 && (
                          <div className="text-center">
                            <small>Showing 10 of {steps[currentStep].mutatedOffspring.length} individuals</small>
                          </div>
                        )}
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
                    <h5>How Genetic Algorithm Works</h5>
                    <p>
                      Genetic Algorithm is a population-based algorithm inspired by natural selection, using selection, crossover, and mutation operators.
                    </p>
                    <ol>
                      <li>Generate an initial population of random solutions</li>
                      <li>Evaluate the fitness of each solution</li>
                      <li>Select parents based on fitness (better solutions have higher chance)</li>
                      <li>Create offspring through crossover (combining parts of parents)</li>
                      <li>Apply mutation to introduce diversity</li>
                      <li>Replace the old population with the new generation</li>
                      <li>Repeat until a stopping criterion is met</li>
                    </ol>
                    <h5>Key Metrics</h5>
                    <ul>
                      <li><strong>Population Size:</strong> Number of solutions in each generation</li>
                      <li><strong>Crossover Success Rate:</strong> Percentage of crossovers that produced better offspring</li>
                      <li><strong>Mutation Impact:</strong> Effect of mutations on solution quality</li>
                      <li><strong>Selection Pressure:</strong> How strongly selection favors better solutions</li>
                      <li><strong>Population Diversity:</strong> Genetic diversity within the population</li>
                    </ul>
                    <h5>Strengths and Weaknesses</h5>
                    <p><strong>Strengths:</strong> Explores large search spaces effectively, handles complex problems well</p>
                    <p><strong>Weaknesses:</strong> Computationally intensive, requires careful parameter tuning</p>
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

export default GeneticAlgorithmVisualizer;
